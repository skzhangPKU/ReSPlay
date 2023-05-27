import cv2
import numpy as np
from PIL import Image

def imread(path):
    im = Image.open(path)
    return np.array(im)

def toimage(arr):
    data = np.asarray(arr)
    shape = list(data.shape)
    strdata = data.tostring()
    shape = (shape[1], shape[0])
    image = Image.frombytes('RGBA', shape, strdata)
    return image

def fromimage(im):
    return np.array(im)

def imresize(arr, size,interp='bilinear'):
    im = toimage(arr)
    size = (size[1], size[0])
    func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
    imnew = im.resize(size, resample=func[interp])
    return fromimage(imnew)

def cv2_crop(im, box):
    return im.copy()[box[1]:box[3], box[0]:box[2], :]

def match_img(image,target,value,thresholdFlag=False,params=None):
    img_rgb = cv2.imread(image)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(target,0)
    try:
        w, h = template.shape[::-1]
    except:
        pass
    if thresholdFlag:
        ret0,img_gray = cv2.threshold(img_gray, params["img_binary_threshold"], 255, cv2.THRESH_BINARY)
        ret1,template = cv2.threshold(template, params["img_binary_threshold"], 255, cv2.THRESH_BINARY)
    template = cv2.resize(template,(150,int(150*template.shape[1]/template.shape[0])))
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = value
    res_max = np.amax(res)
    if res_max>=threshold:
        loc = np.where(res >= res_max)
    else:
        return None
    for pt in zip(*loc[::-1]):
        x = int(pt[0] + w/2)
        y = int(pt[1] + h/2)
        return (x,y)

def convert_canny(record_front,replay_front):
    record_front = cv2.Canny(record_front, 50, 150)
    record_front = np.expand_dims(record_front, 2)
    tmp_record = np.zeros_like(record_front)
    record_front = np.concatenate([record_front,tmp_record,tmp_record], axis=2)
    replay_front = cv2.Canny(replay_front, 50, 150)
    replay_front = np.expand_dims(replay_front, 2)
    tmp_replay = np.zeros_like(replay_front)
    replay_front = np.concatenate([replay_front,tmp_replay,tmp_replay], axis=2)
    return record_front,replay_front

def is_valid(file):
    valid = True
    try:
        Image.open(file).load()
    except OSError:
        valid = False
    return valid