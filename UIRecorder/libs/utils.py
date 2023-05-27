import json
import os
from libs.config import currPath
from libs.config import pkName,boundPattern,originSceneImageRoot
from PIL import Image
import re

def write_record_json(dictObj):
    jsObj = json.dumps(dictObj)
    scenePkImageRoot = os.path.join(originSceneImageRoot, pkName)
    global_dict = read_gloabl_json()
    dir_Name = global_dict[pkName]
    scenePkImageRoot = os.path.join(scenePkImageRoot, str(dir_Name))
    record_json_path = os.path.join(scenePkImageRoot,'record.json')
    with open(record_json_path,'w') as f:
        f.write(jsObj)
    print('wirte record json file finished')

def write_gloabl_json(dictObj):
    jsObj = json.dumps(dictObj)
    jsonPath = os.path.join(currPath, 'global.json')
    with open(jsonPath,'w') as f:
        f.write(jsObj)
    print('wirte global file finished')

def read_gloabl_json():
    jsonPath = os.path.join(currPath, 'global.json')
    with open(jsonPath,'r') as f:
        res = json.load(f)
    return res

def modify_global_file():
    global_dict = read_gloabl_json()
    global_dict[pkName] = global_dict[pkName] + 1
    write_gloabl_json(global_dict)

def parseBounds(boundStr):
    result = re.match(boundPattern, boundStr)
    if result:
        left = int(result.group(1))
        top = int(result.group(2))
        right = int(result.group(3))
        bottom = int(result.group(4))
        return left,top,right,bottom

def cropImage(sceneFilePath, componentPkImageRoot, left, top, right, bottom):
    img = Image.open(sceneFilePath)
    im = img.crop((left, top, right, bottom))
    crop_name = os.path.basename(sceneFilePath).replace('ss','comp')
    crop_name_detail = os.path.join(componentPkImageRoot, crop_name)
    im.save(crop_name_detail)
    print('crop image finished')
    return crop_name_detail
