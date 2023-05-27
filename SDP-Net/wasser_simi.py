import warnings
from skimage.metrics import structural_similarity
from skimage.transform import resize
from scipy.stats import wasserstein_distance
import numpy as np
import cv2
from PIL import Image
from config import REPLAY_RESOLUTION_X
import globalVariable
import imgsim
from utils.common_util import vec_distance
warnings.filterwarnings('ignore')

height = 2**10
width = 2**10

def fromimage(im, flatten=False, mode=None):
  if flatten:
    im = im.convert('F')
  a = np.array(im)
  return a

def imread(name, flatten=False, mode=None):
  im = Image.open(name)
  patch = Image.new("RGBA", (REPLAY_RESOLUTION_X, 80), "#FFFFFF")
  im.paste(patch)
  return fromimage(im, flatten=flatten, mode=mode)

def get_img(path, norm_size=True, norm_exposure=False):
  img = imread(path, flatten=True).astype(int)
  if norm_size:
    img = resize(img, (height, width), anti_aliasing=True, preserve_range=True)
  if norm_exposure:
    img = normalize_exposure(img)
  return img

def get_histogram(img):
  h, w = img.shape
  hist = [0.0] * 256
  for i in range(h):
    for j in range(w):
      hist[img[i, j]] += 1
  return np.array(hist) / (h * w)

def normalize_exposure(img):
  img = img.astype(int)
  hist = get_histogram(img)
  cdf = np.array([sum(hist[:i+1]) for i in range(len(hist))])
  sk = np.uint8(255 * cdf)
  height, width = img.shape
  normalized = np.zeros_like(img)
  for i in range(0, height):
    for j in range(0, width):
      normalized[i, j] = sk[img[i, j]]
  return normalized.astype(int)

def earth_movers_distance(path_a, path_b):
  img_a = get_img(path_a, norm_exposure=True)
  img_b = get_img(path_b, norm_exposure=True)
  hist_a = get_histogram(img_a)
  hist_b = get_histogram(img_b)
  return wasserstein_distance(hist_a, hist_b)

def initial_global_variables():
   globalVariable.init()
   vtr = imgsim.Vectorizer(device='cuda')
   globalVariable.set('vtr', vtr)