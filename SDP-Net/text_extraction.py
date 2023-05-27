import pytesseract
from PIL import Image
import unicodedata
import sys
import jieba
import enchant
import difflib
from config import REPLAY_RESOLUTION_X,REPLAY_RESOLUTION_Y,COLUMN,ROW
import datetime
from xml.etree import ElementTree as ET
from libs.AdbCommand import dump_layout
from utils.common_util import parseBounds
# from agent import get_position_by_coord
import uiautomator2 as u2
import time
from fuzzywuzzy import fuzz
from nltk.corpus import words

# driver = u2.connect()
d = enchant.Dict("en-US")
ele = None
max_score = -1

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

def match_text(driver,comp_path,params):
    global ele
    global max_score
    ele = None
    max_score = -1
    img_record = Image.open(comp_path)
    img_record_word = image_to_words_comp(img_record,params=params)
    img_str = ' '.join(img_record_word)
    if img_str.strip() == '':
        random_index = -2
        return random_index
    # xmlStr = driver.dump_hierarchy()
    xmlStr = driver.page_source
    xmlRoot = ET.fromstring(xmlStr)
    findLeafNodes(img_str, xmlRoot)
    if ele is not None:
        # special
        boundStr = ele.attrib['bounds']
        left, top, right, bottom = parseBounds(boundStr)
        pos = int((left+right)/2), int((top+bottom)/2)
        random_index = get_position_by_coord(pos[0], pos[1])
    else:
        random_index = -2
    return random_index

def getElementWH(ele):
    boundStr = ele.attrib['bounds']
    left,top,right,bottom = parseBounds(boundStr)
    return  (right-left),(bottom-top)

def minAreaFindNodeListener(element):
    global ele
    ele = element

def findLeafNodes(img_str, rootNode):
    global max_score
    foundInChild = False
    for node in rootNode:
        foundInChild |= findLeafNodes(img_str, node)
    if foundInChild:
        return True
    if 'text' not in rootNode.attrib:
        return False
    compare_str1 = img_str.lower()
    compare_str2 = rootNode.attrib['text'].lower().replace('& ','')
    current_score = get_equal_rate_1(compare_str1, compare_str2)
    if 'description' in rootNode.attrib:
        compare_str3 = rootNode.attrib['description'].lower().replace('& ', '').replace(',',' ')
        current_score2 = get_equal_rate_1(compare_str1, compare_str3)
        current_score = max(current_score, current_score2)
    if 'content-desc' in rootNode.attrib:
        compare_str4 = rootNode.attrib['content-desc'].lower().replace('& ', '').replace(',',' ')
        current_score3 = get_equal_rate_1(compare_str1, compare_str4)
        current_score = max(current_score, current_score3)
    if current_score>=0.3:
        if current_score>max_score:
            max_score = current_score
            minAreaFindNodeListener(rootNode)
        if current_score>0.9:
            return True
        return False
    else:
        return False


def isChinese(word):
    for ch in word:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

def get_equal_rate_1(str1, str2):
   DATE_LIST = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun','January','February','March','April','May','June','July','August','September','October','November','December',
                'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','Dec']
   for item in DATE_LIST:
       str1 = str1.replace(item+' ','').replace(item, '')
       str2 = str2.replace(item+' ','').replace(item, '')
   return difflib.SequenceMatcher(None, str1, str2).quick_ratio()

def image_to_words_comp(img,lang='eng',params=None):
    img_gray = img.convert("L")
    img_tw0 = img_gray.point(lambda x: 255 if x > params['txt_extract_binary_threshold'] else 0)
    if lang == 'eng':
        code = pytesseract.image_to_string(img_tw0, lang='eng').strip().replace('.', '')
        word_list = []
        str_list = list(jieba.cut(code))
        len_str_list = len(str_list)
        for item in str_list:
            if item == ' ' or item == '.' or item == ',':
                continue
            if item.replace('%', '').isdigit() or len(item) == 1:
                continue
            if len_str_list > 1:
                if d.check(item):
                    word_list.append(item)
            else:
                word_list.append(item)
    elif lang == 'chi_sim':
        code = pytesseract.image_to_string(img, lang='chi_sim').strip().replace('。', '')
        word_list = []
        ch_str_list = list(jieba.cut(code))
        for item in ch_str_list:
            if item == ' ' or item == ',' or item == '"' or item == '\n':
                continue
            if isChinese(item):
                word_list.append(item)
    return word_list

def image_to_words(img,lang='eng',params=None):
    patch = Image.new("RGBA",(REPLAY_RESOLUTION_X,80),"#FFFFFF")
    img.paste(patch)
    img_gray = img.convert("L")
    img_tw0 = img_gray.point(lambda x: 255 if x > params['txt_extract_binary_threshold'] else 0)
    if lang=='eng':
        code = pytesseract.image_to_string(img_tw0,lang='eng').strip().replace('.','')
        word_list = []
        str_list = list(jieba.cut(code))
        for item in str_list:
            if item == ' ' or item == '.' or item == ',':
                continue
            if item.replace('%','').isdigit() or len(item)==1:
                continue
            if d.check(item):
                word_list.append(item)
    elif lang=='chi_sim':
        code = pytesseract.image_to_string(img, lang='chi_sim').strip().replace('。', '')
        word_list = []
        ch_str_list = list(jieba.cut(code))
        for item in ch_str_list:
            if item == ' ' or item == ',' or item == '"' or item == '\n':
                continue
            if isChinese(item):
                word_list.append(item)
    return word_list

def textocr():
    img = Image.open('compare/siamense_12.png')
    img2 = Image.open('imageFile/com.amazon.mShop.android.shopping/4/screen/ss_13.png')
    params = {}
    params['txt_binary_threshold'] = 225
    img_word1 = image_to_words(img,params=params)
    img_word2 = image_to_words(img2,params=params)
    img_str = ' '.join(img_word1)
    img_str2 = ' '.join(img_word2)
    print(img_str)
    print(img_str2)
    print(get_equal_rate_1(img_str, img_str2))

def readtxt():
    with open("C:\\Users\\andrew\\Desktop\\remark.txt",'r') as file:
        list_item = file.readlines()
    kk = ''
    for item in list_item:
        if item[0]=='[' or item == '\n':
            continue
        kk = kk + item.strip()+' '
    with open('a2.txt','w') as file:
        file.write(kk)

if __name__ == '__main__':
    textocr()
    print('finished')