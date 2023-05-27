import os
import time
import globalVariable
import random
import math
import torch
import pytesseract
import jieba
import numpy as np
from PIL import Image
from config import *
from wasser_simi import earth_movers_distance
from utils.common_util import vec_distance
from utils.image_util import fromimage,match_img
from text_extraction import image_to_words,get_equal_rate_1
from text_extraction import match_text
from appium.webdriver.common.touch_action import TouchAction
def get_reward_by_similarity(params):
    record_front = 'simdir/record_front.png'
    record_behind = 'simdir/record_behind.png'
    replay_front = 'simdir/replay_front.png'
    replay_behind = 'simdir/replay_behind.png'
    # The distance between GUI images
    image_distance = vec_distance(record_behind, replay_behind)
    img_record = Image.open(record_behind)
    img_replay = Image.open(replay_behind)
    img_record_word = image_to_words(img_record,params=params)
    img_replay_word = image_to_words(img_replay,params=params)
    img_record_str = ' '.join(img_record_word)
    img_replay_str = ' '.join(img_replay_word)
    # The distance between texts in GUI states
    string_similarity = get_equal_rate_1(img_replay_str,img_record_str)
    if string_similarity > params["text_similarity_threshold"]:
        text_reward = 1.0
    else:
        text_reward = 0.0
    if image_distance > 0.8:
        vision_reward = 1.0
    else:
        vision_reward = 0.0
    immediate_reward = 0.8*text_reward + 0.2*vision_reward
    return immediate_reward

def select_action(state,step_index):
    # fetch global variable
    steps_done_dict = globalVariable.get('steps_done_dict')
    policy_net = globalVariable.get('policy_net')
    position_set = globalVariable.get('position_set')
    # random policy
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done_dict[str(step_index)] / EPS_DECAY)
    sample = random.random()
    if sample > eps_threshold:
        with torch.no_grad():
            grid_position = policy_net(state).max(1)[1].view(1, 1).item()
    else:
        if str(step_index) in position_set:
            rawImg = Image.open(SIM_DIR+'/replay_front.png')
            while True:
                if len(position_set[str(step_index)]) >= COLUMN * ROW:
                    grid_position = -1
                    break
                grid_position = random.randrange(COLUMN * ROW)
                if str(grid_position) not in position_set[str(step_index)]:
                    pure_flag = rigion_pure_color(grid_position, rawImg)
                    if pure_flag:
                        position_set[str(step_index)][str(grid_position)] = {}
                        position_set[str(step_index)][str(grid_position)]['reward'] = 0.0
                        position_set[str(step_index)][str(grid_position)]['restart'] = False
                        continue
                    else:
                        print(grid_position)
                        break
        else:
            grid_position = random.randrange(COLUMN * ROW)
    steps_done_dict[str(step_index)] = steps_done_dict[str(step_index)] + 1
    return grid_position

def exec_action(driver,action,params=None):
    tapx,tapy = get_coord_by_position(action,params)
    # driver.tap([(tapx, tapy)],300)
    print((tapx, tapy))
    os.system('adb shell input tap %d %d' % (tapx, tapy))
    time.sleep(2)

def get_coord_by_position(action,params):
    index = action + 1
    pos_row, pos_col = int(index / COLUMN), index % COLUMN
    cell_width, cell_hight = REPLAY_RESOLUTION_X / COLUMN, REPLAY_RESOLUTION_Y / ROW
    if pos_col > 0:
        tapy = pos_row * cell_hight + 0.5 * cell_hight
        tapx = (pos_col - 1) * cell_width + 0.5 * cell_width # center selected
    else:
        tapy = (pos_row - 1) * cell_hight + 0.5 * cell_hight
        tapx = (COLUMN - 1) * cell_width + 0.5 * cell_width # center
    return tapx,tapy

def get_grid_by_position(action):
    index = action + 1
    pos_row, pos_col = int(index / COLUMN), index % COLUMN
    cell_width, cell_hight = REPLAY_RESOLUTION_X / COLUMN, REPLAY_RESOLUTION_Y / ROW
    if pos_col > 0:
        tapy = pos_row * cell_hight + 0.5 * cell_hight
        tapx = (pos_col - 1) * cell_width + 0.5 * cell_width
    else:
        tapy = (pos_row - 1) * cell_hight + 0.5 * cell_hight
        tapx = (COLUMN - 1) * cell_width + 0.5 * cell_width
    left, top = tapx - 0.5 * cell_width, tapy - 0.5 * cell_hight
    right, bottom = tapx + 0.5 * cell_width, tapy + 0.5 * cell_hight
    return left,top,right,bottom

def get_position_by_coord(abs_x,abs_y):
    for action in range(COLUMN*ROW):
        index = action+1
        pos_row, pos_col = int(index / COLUMN), index % COLUMN
        cell_width, cell_hight = REPLAY_RESOLUTION_X / COLUMN, REPLAY_RESOLUTION_Y / ROW
        if pos_col > 0:
            tapy = pos_row * cell_hight + 0.5 * cell_hight
            tapx = (pos_col - 1) * cell_width + 0.5 * cell_width
        else:
            tapy = (pos_row - 1) * cell_hight + 0.5 * cell_hight
            tapx = (COLUMN - 1) * cell_width + 0.5 * cell_width
        left_up_x = tapx - 0.5 * cell_width
        left_up_y = tapy - 0.5 * cell_hight
        right_bottom_x = tapx + 0.5 * cell_width
        right_bottom_y = tapy + 0.5 * cell_hight
        if abs_x >= left_up_x and abs_x < right_bottom_x and abs_y >= left_up_y and abs_y < right_bottom_y:
            return action

def checkRandomIndex(random_index,params=None):
    tapx,tapy = get_coord_by_position(random_index,params)
    left, top = tapx-300,tapy-100
    right, bottom = tapx+300,tapy+100
    if right > REPLAY_RESOLUTION_X:
        right = REPLAY_RESOLUTION_X
    if bottom > REPLAY_RESOLUTION_Y:
        bottom = REPLAY_RESOLUTION_Y
    replay_front = SIM_DIR+'/replay_front.png'
    img = Image.open(replay_front)
    img_gray = img.convert("L")
    img_tw0 = img_gray.point(lambda x: 255 if x > 220 else 0)
    img_crop = img_tw0.crop((left, top, right, bottom))
    code = pytesseract.image_to_string(img_crop, lang='eng').strip().replace('.', '')
    word_list = []
    str_list = list(jieba.cut(code))
    if 'Switch' in str_list or 'IME' in str_list:
        return True
    return False

def rigion_pure_color(action,rawImg):
    left,top,right,bottom = get_grid_by_position(action)
    crop  = rawImg.crop([left,top,right,bottom])
    crop_arr = fromimage(crop.convert('RGB'))
    first = np.mean(crop_arr[:, :, 0])
    second = np.mean(crop_arr[:,:,1])
    third = np.mean(crop_arr[:,:,2])
    MEAN = np.array((first, second, third), dtype=np.float32).reshape((1, 1, 3))
    res = crop_arr-MEAN.flat
    if np.all(res==0):
        return True
    return False

def select_action_by_synthesis_strategy(current_state,step_index,component_path,match_flag,params):
    ime_flag = False
    position_set = globalVariable.get('position_set')
    if match_flag:
        if params['img_binary_threshold'] < 0:
            pos = match_img(CACHE_FRONT, component_path, value=params["img_match_threshold"])
        else:
            pos = match_img(CACHE_FRONT, component_path, value=params["img_match_threshold"], thresholdFlag=True, params=params)
    else:
        pos = None
    if pos:
        random_index = get_position_by_coord(pos[0], pos[1])
        print('random index', random_index)
    else:
        random_index = select_action(current_state, step_index)
        ime_flag = checkRandomIndex(random_index,params)
        if ime_flag:
            if str(random_index) not in position_set[str(step_index)]:
                position_set[str(step_index)][str(random_index)] = {}
                position_set[str(step_index)][str(random_index)]['reward'] = 0.0
                position_set[str(step_index)][str(random_index)]['restart'] = False
                print('choose switch ime-choose again')
    return random_index,ime_flag,pos

