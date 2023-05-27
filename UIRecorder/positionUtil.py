import time
import operator
import os
import numpy
from libs.config import *
import collections
import re
import math
import uiautomator2 as u2
from libs.GetDeviceInfo import getDeviceInfo
from libs.AdbCommand import checkPackageInstall
from libs.AdbCommand import start_activity,force_stop,screencap,clear
from libs.AdbCommand import dump_layout,mkdirs,pull,delete
from subprocess import PIPE, Popen
from threading  import Thread
from xml.etree.ElementTree import parse
from xml.etree import ElementTree as ET
from libs.utils import read_gloabl_json,write_gloabl_json,parseBounds,cropImage
try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty  # python 3.x

ele = None
nele= None
edist= 0

def getElementWH(ele):
    boundStr = ele.attrib['bounds']
    result = re.match(boundPattern, boundStr)
    if result:
        left = int(result.group(1))
        top = int(result.group(2))
        right = int(result.group(3))
        bottom = int(result.group(4))
        return (right-left),(bottom-top)

def nearestFindNodeListener(element,dist_ele):
    global nele
    global edist
    if nele == None:
        nele = element
        edist = dist_ele
    elif dist_ele<edist:
        nele = element
        edist = dist_ele

def minAreaFindNodeListener(element):
    global ele
    if ele == None:
        ele = element
    else:
        ele_width,ele_hight = getElementWH(ele)
        element_w, element_h = getElementWH(element)
        if (element_w*element_h)<(ele_width*ele_hight):
            ele = element

def distance_rect(px,py,left,top,right,bottom):
    dx = max(left-px,0,px-right)
    dy = max(top-py,0,py-bottom)
    return math.sqrt(dx*dx+dy*dy)

def findLeafMostNodesAtPoint(tpx, tpy, rootNode):
    foundInChild = False
    for node in rootNode:
        foundInChild |= findLeafMostNodesAtPoint(tpx,tpy,node)
    if foundInChild:
        return True
    if 'bounds' in rootNode.attrib:
        if len(rootNode.attrib['bounds'])>=10:
            boundStr = rootNode.attrib['bounds']
            left, top, right, bottom = parseBounds(boundStr)
            if (bottom-top)*(right-left)<(res_x*res_y/3):
                if tpx >= left and tpx <= right and tpy>=top and tpy<=bottom:
                    # add collection
                    minAreaFindNodeListener(rootNode)
                    return True
                else:
                    if len(list(rootNode))==0 and (right-left)<=res_x/5 and (bottom-top)<=res_y/10:
                        dist = distance_rect(tpx,tpy,left,top,right,bottom)
                        nearestFindNodeListener(rootNode,dist)
                    return False
            else:
                return False
        else:
            return False
    else:
        return False

def findElement(px,py,pageSourceDetail):
    with open(pageSourceDetail, "r",encoding='utf-8') as f:
        xmlStr = f.read()
    xmlRoot = ET.fromstring(xmlStr)
    findLeafMostNodesAtPoint(px, py, xmlRoot)

def checkIfInstalled():
    while True:
        ret = checkPackageInstall(pkName, deviceName)
        if ret is True:
            print('%s has installed..' % pkName)
            # clear(pkName,deviceName)
            force_stop(pkName,deviceName)
            time.sleep(0.5)
            start_activity(pkName+'/'+activityName,deviceName)
            time.sleep(0.5)
            break
        else:
            print('sleep 0.5s to wait %s install finish' % pkName)
            time.sleep(0.5)

def getScriptFileRoot():
    currPath = os.getcwd()
    scriptFileRoot = os.path.join(currPath, 'scriptFile', deviceName)
    if os.path.isdir(scriptFileRoot) is False:
        os.makedirs(scriptFileRoot)
    return scriptFileRoot

def enqueue_output(out, queue):
    for line_byte in iter(out.readline, b''):
        line = str(line_byte, encoding="utf-8")
        if 'ABS_MT_POSITION_X' in line or 'ABS_MT_POSITION_Y' in line:
            line = line.replace('\r\n', '')
            queue.put(line)
    out.close()

def get_resolution():
    deviceInfo = getDeviceInfo(deviceName)
    # deviceInfo = getDeviceInfo(ipAddress)
    resolutioin = deviceInfo['deviceResolution']
    splitInfo = resolutioin.split('x')
    max_width = int(splitInfo[0].split()[0])
    max_height = int(splitInfo[1].split()[0])
    return (max_width, max_height)

def write_constant(scriptFile):
    scriptFile.write('#!/usr/bin/env python\r')
    scriptFile.write('# coding: utf-8\r')
    scriptFile.write('\r')
    scriptFile.write('import os\r')
    scriptFile.write('import time\r')
    scriptFile.write('\r')
    scriptFile.write('if __name__ == "__main__":\r')

def existDIR(dir_name):
    if os.path.isdir(dir_name) is False:
        os.makedirs(dir_name)

def initialPath():
    scenePkImageRoot = os.path.join(originSceneImageRoot, pkName)
    existDIR(scenePkImageRoot)
    global_dict = read_gloabl_json()
    dir_Name = global_dict[pkName]
    scenePkImageRoot = os.path.join(scenePkImageRoot, str(dir_Name))
    screenshotPkImageRoot = os.path.join(scenePkImageRoot, screenPath)
    existDIR(screenshotPkImageRoot)
    # widget screenshot path
    componentPkImageRoot = os.path.join(scenePkImageRoot, componentPath)
    existDIR(componentPkImageRoot)
    # pageSource path
    pageSourcePkXmlPath = os.path.join(scenePkImageRoot, pageSourcePath)
    existDIR(pageSourcePkXmlPath)
    return componentPkImageRoot,screenshotPkImageRoot,pageSourcePkXmlPath

def pageSourceCap(pageSourceDetail,deviceId,driver):
    xml = driver.dump_hierarchy()
    with open(pageSourceDetail, "w",encoding='utf-8') as f:
        f.write(xml)
    return xml

def reset_global():
    global ele
    global nele
    global edist
    ele = None
    nele = None
    edist = 0

def enhanced_record_value(recordValue,componentPkImageRoot,sceneFilePath,pageSourceDetail,driver):
    # get coordinates
    px,py = recordValue['clickPosition']
    findElement(px,py,pageSourceDetail)
    global ele
    global nele
    if ele is not None:
        boundStr = ele.attrib['bounds']
        left, top, right, bottom = parseBounds(boundStr)
        # crop component image
        crop_name_detail = cropImage(sceneFilePath,componentPkImageRoot,left, top, right, bottom)
        # create dicts
        step_dict = {'screenshot': sceneFilePath, 'componentshot': crop_name_detail, 'widget_coordinate': boundStr,
                     'click_coordinate': str(px) + ',' + str(py),'clickType':recordValue['clickType'],
                     'clickStartTime':recordValue['clickStartTime'],'screenType':recordValue['screenType'],'index':ele.attrib['index'],
                     'resource-id':ele.attrib['resource-id'],'text':ele.attrib['text'],'class':ele.attrib['class'],'package':ele.attrib['package'],
                     'content-desc':ele.attrib['content-desc']}
    else:
        boundStr = nele.attrib['bounds']
        left, top, right, bottom = parseBounds(boundStr)
        crop_name_detail = cropImage(sceneFilePath, componentPkImageRoot, left, top, right, bottom)
        # create dicts
        step_dict = {'screenshot': sceneFilePath, 'componentshot': crop_name_detail, 'widget_coordinate': boundStr,
                     'click_coordinate': str(px) + ',' + str(py), 'clickType': recordValue['clickType'],
                     'clickStartTime': recordValue['clickStartTime'], 'screenType': recordValue['screenType'],'index':nele.attrib['index'],
                     'resource-id': nele.attrib['resource-id'], 'text': nele.attrib['text'],'class':nele.attrib['class'],'package':nele.attrib['package'],
                     'content-desc': nele.attrib['content-desc']}
    reset_global()
    return step_dict

def get_screen_xml(clickNum,driver):
    time.sleep(3)
    componentPkImageRoot,screenshotPkImageRoot,pageSourcePkXmlPath = initialPath()
    sceneFilePath = os.path.join(screenshotPkImageRoot, 'ss_%s.png' % str(clickNum))
    # driver.screenshot(sceneFilePath)
    screencap(sceneFilePath,deviceName)
    pageSourceDetail = os.path.join(pageSourcePkXmlPath, 'ps_%s.xml' % str(clickNum))
    pageSourceCap(pageSourceDetail, deviceName, driver)
    print('screen and xml have been writen to file')
    return componentPkImageRoot,sceneFilePath,pageSourceDetail

