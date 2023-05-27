import re
import os
import cv2
import json
import torch
import numpy as np
import imgsim
import shutil
import globalVariable
import matplotlib.pyplot as plt
from config import *
from utils.image_util import imread,imresize,convert_canny
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

def read_json(file):
    with open(file, 'r') as load_f:
        load_dict = json.load(load_f)
    return load_dict

def get_screen(aug=False,path=None):
    MEAN_TORCH_BGR = np.array((103.53, 116.28, 123.675), dtype=np.float32).reshape((1, 3, 1, 1))
    STD_TORCH_BGR = np.array((57.375, 57.12, 58.395), dtype=np.float32).reshape((1, 3, 1, 1))
    img = imresize(imread(path), (320, 180))
    norm_img = (img[..., [2, 1, 0]] - MEAN_TORCH_BGR.flat) / STD_TORCH_BGR.flat
    norm_img = np.transpose(norm_img, (2, 0, 1))[np.newaxis, ...] # BGR
    norm_img = torch.autograd.Variable(torch.Tensor(norm_img)).cuda() # GBR
    return norm_img#bgr

def parseBounds(boundStr):
    result = re.match(BOUND_PATTERN, boundStr)
    if result:
        left = int(result.group(1))
        top = int(result.group(2))
        right = int(result.group(3))
        bottom = int(result.group(4))
        return left,top,right,bottom

def vec_distance(record_front_path,replay_front_path,canny=True):
    img1 = Image.open(record_front_path)
    patch = Image.new("RGBA",(REPLAY_RESOLUTION_X,80),"#FFFFFF")
    img1.paste(patch)
    img2 = Image.open(replay_front_path)
    img2.paste(patch)
    record_front = cv2.cvtColor(np.asarray(img1), cv2.COLOR_RGB2BGR)
    replay_front = cv2.cvtColor(np.asarray(img2), cv2.COLOR_RGB2BGR)
    replay_front = np.resize(replay_front, (record_front.shape[0], record_front.shape[1], record_front.shape[2]))
    if canny:
        record_front, replay_front = convert_canny(record_front, replay_front)
    front_distance = calculate_similarity(record_front, replay_front)
    return front_distance

def calculate_similarity(record,replay):
    vtr = globalVariable.get('vtr')
    with torch.no_grad():
        vec0 = vtr.vectorize(record)
        vec1 = vtr.vectorize(replay)
    if IMG_SIM_TYPE == 'Euclidean':
        dist = imgsim.distance(vec0, vec1)
    elif IMG_SIM_TYPE == 'Cosine':
        dist = cosine_similarity(vec0,vec1)
    return dist

def init_steps_done_dict(action_len):
    steps_done_dict = {}
    for j in range(action_len):
        steps_done_dict[str(j)] = 0
    globalVariable.set('steps_done_dict',steps_done_dict)

def get_state(replay_behind_path,record_behind2_path,sim_path):
    vtr = globalVariable.get('vtr')
    replay_behind = get_screen(aug=False, path=replay_behind_path)
    shutil.copy(replay_behind_path, sim_path)
    if os.path.exists(record_behind2_path):
        record_behind2 = get_screen(aug=False, path=record_behind2_path)
        segnet_next_state = torch.cat([replay_behind, record_behind2], 3)
        with torch.no_grad():
            next_state = vtr.my_model(segnet_next_state)
            next_state = next_state.cpu()
    else:
        next_state = None
    return next_state

def store_rr_to_path(store_path):
    shutil.copy('simdir/record_behind.png', store_path + '/record_behind.png')
    shutil.copy('simdir/record_front.png', store_path + '/record_front.png')
    shutil.copy('simdir/replay_behind.png', store_path + '/replay_behind.png')
    shutil.copy('simdir/replay_front.png', store_path + '/replay_front.png')