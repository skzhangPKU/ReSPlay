import torch
import os
import numpy as np
import torch.optim as optim
from models.replaymomory import ReplayMemory,Transition
from imageio import imread
from config import *
import math
from models.uiencoder import Generator
from libs.AdbCommand import checkIfInstalled,dump_layout,call_adb
from itertools import count
from xml.etree import ElementTree as ET
from agent import exec_action,get_position_by_coord,select_action,checkRandomIndex,select_action_by_synthesis_strategy
from models.dqn import DQN
import torch.nn as nn
import random
import json
from torchvision import transforms
import re
import imgsim
import cv2
import datetime
import warnings
import time
import shutil
import globalVariable
from utils.common_util import init_steps_done_dict,read_json,get_screen,vec_distance,get_state,store_rr_to_path
from utils.image_util import match_img
from wasser_simi import earth_movers_distance
from text_extraction import get_equal_rate_1,image_to_words,match_text
from PIL import Image
from agent import get_reward_by_similarity
from util_info import getTotalM
from appium_helper import AppiumLauncher
from appium import webdriver
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['CUDA_VISIBLE_DEVICES']='0'

if SEED:
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False

memory = ReplayMemory(100000)
# memory.cuda()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

desired_caps = {'platformName': 'Android', # default Android
                 'platformVersion': '11',
                 'udid': '7ecb032b',
                 'deviceName': 'Xiaomi10',
                 'autoGrantPermissions': False,
                 'fullReset': False,
                 'resetKeyboard': True,
                 'androidInstallTimeout': 30000,
                 'isHeadless': False,
                 'automationName': 'uiautomator2',
                 'adbExecTimeout': 30000,
                 'appWaitActivity': '*',
                 'newCommandTimeout': 200}

driver = webdriver.Remote(f'http://127.0.0.1:4723/wd/hub', desired_caps)


def optimize_model():
    if len(memory) <= BATCH_SIZE:
        return
    policy_net = globalVariable.get('policy_net')
    target_net = globalVariable.get('target_net')
    optimizer = globalVariable.get('optimizer')
    batch_num = int(len(memory) / BATCH_SIZE)
    epoch_num = 2000
    print('==================training==================')
    for it in range(batch_num*epoch_num):
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        # concat
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state)), device=device, dtype=torch.bool).to(device)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)
        # calculate Q values
        dqn_net = policy_net(state_batch)
        # the Q value of actions
        state_action_values = dqn_net.gather(1, action_batch)
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        # Compute Bellman error
        bellman_error = expected_state_action_values.unsqueeze(1) - state_action_values
        # clip the bellman error between [-1 , 1]
        clipped_bellman_error = bellman_error.clamp(-1, 1)
        # Note: clipped_bellman_delta * -1 will be right gradient
        d_error = clipped_bellman_error * -1.0
        optimizer.zero_grad()
        state_action_values.backward(d_error.data)
        # Clear previous gradients before backward pass
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

def train_apk(pkName,activityName,params):
    pkPath = 'imageFile/'+pkName
    trace_list = os.listdir(pkPath)
    vtr = globalVariable.get('vtr')
    overall_record = {}
    for trace in trace_list:
        starttime = datetime.datetime.now()
        overall_record[trace] = {}
        action_dict = {}
        local_step = 0
        widgets = os.listdir(pkPath + '/' + trace + '/component')
        action_len = len(widgets)
        init_steps_done_dict(action_len)
        checkIfInstalled(driver,pkName,activityName)
        time.sleep(5)
        restart_flag = False
        trace_flag = False
        step_index = 0
        globalVariable.set("position_set",{})
        while step_index < action_len:
            match_flag = True
            text_flag = True
            first_flag = True
            second_flag = True
            count_action = 0
            if trace_flag:
                break
            position_set = globalVariable.get("position_set")
            if str(step_index) not in position_set:
                position_set[str(step_index)] = {}
            while True:
                print('==========================')
                if local_step % TARGET_UPDATE==0:
                    optimize_model()
                    print("optimize model")
                if restart_flag:
                    for item in range(step_index):
                        action_item = action_dict[str(item)]
                        exec_action(driver,action_item,params)
                        time.sleep(2)
                    restart_flag = False
                print('current step index, ',step_index)
                # current state start
                driver.get_screenshot_as_file(CACHE_FRONT)
                record_behind_path = pkPath + '/' + trace + '/screen/' + 'ss_' + (str(step_index + 2)) + '.png'
                shutil.copy(record_behind_path, SIM_DIR+'/record_behind.png')
                current_state_cpu = get_state(CACHE_FRONT,record_behind_path,SIM_DIR+'/replay_front.png')
                current_state = current_state_cpu.to(device)
                # select action
                component_path = pkPath + '/' + trace + '/component/'+'comp_'+(str(step_index+1))+'.png'
                print(component_path)
                if text_flag:
                    random_index = match_text(driver,component_path,params)
                    text_flag = False
                    if random_index == -2:
                        # select other strategies
                        random_index, ime_flag, pos = select_action_by_synthesis_strategy(current_state, step_index,component_path, match_flag,params)
                        if ime_flag:
                            continue
                        if random_index == -1:
                            trace_flag = True
                            break
                    else:
                        print('attempt attribute matching success')
                else:
                    if first_flag:
                        random_index = 2 * COLUMN
                        ime_flag = False
                        first_flag = False
                    elif second_flag:
                        random_index = 4 * COLUMN - 1
                        ime_flag = False
                        second_flag = False
                    else:
                        # select other strategies
                        random_index, ime_flag, pos = select_action_by_synthesis_strategy(current_state, step_index,component_path, match_flag,params)
                    if ime_flag:
                        continue
                    if random_index == -1:
                        trace_flag = True
                        break
                grid_position = torch.tensor([[random_index]], device=device, dtype=torch.long)
                exec_action(driver,random_index,params=params)
                time.sleep(5)
                count_action = count_action + 1
                local_step += 1
                # position exist start
                if str(random_index) in position_set[str(step_index)]:
                    if position_set[str(step_index)][str(random_index)]['reward'] >=1.0:
                        if not text_flag:
                            match_flag = True
                        break
                    elif position_set[str(step_index)][str(random_index)]['restart']:
                        restart_flag = True
                        if not text_flag:
                            if pos:
                                match_flag = False
                        # for closing popupWindow
                        exec_action(driver,COLUMN,params)
                        checkIfInstalled(driver,pkName,activityName)
                    continue
                else:
                    position_set[str(step_index)][str(random_index)]={}
                # position exist end
                print('random index', random_index)
                # next state start
                driver.get_screenshot_as_file(CACHE_BEHIND)
                record_behind2_path = pkPath + '/' + trace + '/screen/' + 'ss_' + (str(step_index + 3)) + '.png'
                next_state = get_state(CACHE_BEHIND,record_behind2_path,SIM_DIR+'/replay_behind.png')
                # next state end
                record_front_raw_path = pkPath + '/' + trace + '/screen/'+'ss_'+str(step_index+1)+'.png'
                shutil.copy(record_front_raw_path, 'simdir/record_front.png')
                reward_value = get_reward_by_similarity(params)
                self_distance = vec_distance('simdir/replay_front.png', 'simdir/replay_behind.png', canny=False)
                reward = torch.tensor([reward_value], device=device)
                # store the related information of position
                position_set[str(step_index)][str(random_index)]['reward'] = reward_value
                position_set[str(step_index)][str(random_index)]['restart'] = False
                # store transitions
                if reward_value>=1.0:
                    action_dict[str(step_index)] = random_index
                    match_flag = True
                    store_path = 'store/' + pkName + '/' + str(trace) + '/' + str(step_index)
                    if not os.path.exists(store_path):
                        os.makedirs(store_path)
                    store_rr_to_path(store_path)
                    print(action_dict)
                    break
                else:
                    position_set[str(step_index)][str(random_index)]['restart'] = True
                    if self_distance>0.1:
                        restart_flag = True
                        exec_action(driver, COLUMN, params)
                        checkIfInstalled(driver,pkName,activityName)
            step_index += 1
        endtime = datetime.datetime.now()
        overall_record[trace]["action"] = action_dict
        overall_record[trace]["time"] = (endtime - starttime).seconds
    policy_net = globalVariable.get('policy_net')
    torch.save(policy_net.state_dict(), 'save/policy_net_'+pkName+'.pth')
    print(overall_record)
    with open('save/'+pkName+'.json', 'w') as load_f:
        json.dump(overall_record,load_f)
    print('finished')

def initial_global_variables():
    globalVariable.init()
    vtr = imgsim.Vectorizer(device='cuda')
    globalVariable.set('vtr',vtr)
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    globalVariable.set('policy_net',policy_net)
    globalVariable.set('target_net',target_net)

if __name__ == '__main__':
    pk_dict = read_json('global.json')
    for pkName in pk_dict:
        activityName = pk_dict[pkName]
        params = read_json('configuration/global_device.json')
        initial_global_variables()
        train_apk(pkName,activityName,params)
        getTotalM('python.exe')
