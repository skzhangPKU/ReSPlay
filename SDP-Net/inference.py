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

def inference(pkName,activityName,params):
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
            count_action = 0
            if trace_flag:
                break
            position_set = globalVariable.get("position_set")
            if str(step_index) not in position_set:
                position_set[str(step_index)] = {}
            while True:
                print('==========================')
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
                random_index = select_action(current_state,step_index)
                exec_action(driver,random_index,params=params)
                time.sleep(5)
                count_action = count_action + 1
                local_step += 1
                print('random index', random_index)
                driver.get_screenshot_as_file(CACHE_BEHIND)
                record_behind2_path = pkPath + '/' + trace + '/screen/' + 'ss_' + (str(step_index + 3)) + '.png'
                record_front_raw_path = pkPath + '/' + trace + '/screen/'+'ss_'+str(step_index+1)+'.png'
                shutil.copy(record_front_raw_path, 'simdir/record_front.png')
                reward_value = get_reward_by_similarity(params)
                self_distance = vec_distance('simdir/replay_front.png', 'simdir/replay_behind.png', canny=False)
                if reward_value>=1.0:
                    action_dict[str(step_index)] = random_index
                    store_path = 'store/' + pkName + '/' + str(trace) + '/' + str(step_index)
                    if not os.path.exists(store_path):
                        os.makedirs(store_path)
                    store_rr_to_path(store_path)
                    print(action_dict)
                    break
                else:
                    if self_distance>0.1:
                        restart_flag = True
                        exec_action(driver, COLUMN, params)
                        checkIfInstalled(driver,pkName,activityName)
            step_index += 1
        endtime = datetime.datetime.now()
        overall_record[trace]["action"] = action_dict
        overall_record[trace]["time"] = (endtime - starttime).seconds
    print(overall_record)
    print('finished')

def initial_global_variables():
    globalVariable.init()
    vtr = imgsim.Vectorizer(device='cuda')
    globalVariable.set('vtr',vtr)
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    policy_net.load_state_dict(torch.load('save/policy_net.pth'))
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.RMSprop(policy_net.parameters(), lr=0.00025)
    globalVariable.set('policy_net',policy_net)
    globalVariable.set('target_net',target_net)
    globalVariable.set('optimizer', optimizer)

if __name__ == '__main__':
    pk_dict = read_json('global.json')
    for pkName in pk_dict:
        if pkName != 'com.yinxiang':
            continue
        activityName = pk_dict[pkName]
        params = read_json('configuration/global_device.json')
        # load trained models
        initial_global_variables()
        inference(pkName,activityName,params)
