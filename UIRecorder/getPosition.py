import signal
import numpy
import os
import subprocess
import operator
import sys
from libs.GetDeviceInfo import getDeviceInfo
from libs.AdbCommand import checkPackageInstall
from positionUtil import *
from libs.config import *
import uiautomator2 as u2
from libs.utils import modify_global_file,write_record_json
import logging
import datetime

def signal_handler(signal, frame):
    global finishFlag
    finishFlag = True
    print('You pressed Ctrl+C!')

def getevent_position(resolution):
    geteventCmd = 'adb -s %s shell getevent -lt /dev/input/event5' % deviceName  #Huawei Mate8, oppo
    p = Popen(geteventCmd, shell=True, stdout=PIPE)
    positionRecord = collections.OrderedDict()
    q = Queue()
    t = Thread(target=enqueue_output, args=(p.stdout, q))
    t.daemon = True # thread dies with the program
    t.start()
    # get event position
    noOutput = False
    hasLine = False
    max_width = resolution[0]
    while noOutput is False or hasLine is False:
        if finishFlag is True:
            break
        try:
            line = q.get_nowait() # or q.get(timeout=.1)
        except Empty:
            noOutput = True
            if hasLine is False:
                time.sleep(0.5)
        else:
            hasLine = True
            noOutput = False
            splitLine = line.split()
            timeValue = splitLine[1][0:-1]
            positionValue = int(splitLine[4], 16)
            if timeValue not in positionRecord:
                positionRecord[timeValue] = []
            positionRecord[timeValue].append(positionValue)
    p.kill()
    p.wait()
    clickOp = collections.OrderedDict()
    if len(positionRecord) == 1:
        startTime = list(positionRecord.keys())[0]
        positionValue = positionRecord[startTime]
        clickOp['clickType'] = 'tap'
        clickOp['clickStartTime'] = startTime
        clickOp['clickPosition'] = positionValue
        clickOp['screenType'] = 'portrait'
    elif len(positionRecord) > 1:
        startTime = list(positionRecord.keys())[0]
        startPosition = positionRecord[startTime]
        endTime = list(positionRecord.keys())[-1]
        endPosition = positionRecord[endTime]
        startVect = numpy.array(startPosition)
        endVect = numpy.array(endPosition)
        eucDist = round(numpy.linalg.norm(endVect-startVect), 3)
        if eucDist < 20:
            clickOp['clickType'] = 'tap'
            clickOp['clickStartTime'] = startTime
            clickOp['clickPosition'] = startPosition
            clickOp['screenType'] = 'portrait'
        else:
            clickOp['clickType'] = 'swipe'
            clickOp['clickStartTime'] = startTime
            clickOp['clickEndTime'] = endTime
            clickOp['clickStartPosition'] = startPosition
            clickOp['clickEndPosition'] = endPosition
            clickOp['eucDist'] = eucDist
            clickOp['screenType'] = 'portrait'
        print(clickOp)
    # print('end of getevent position....')
    return clickOp

def generate_script(scriptFileRoot, resolution, driver):
    signal.signal(signal.SIGINT, signal_handler)
    clickRecord = []
    clickNum = 0
    sleepFlag = True
    record_dict = {}
    # get event position
    u2.logger.setLevel(logging.ERROR)
    while finishFlag is False:
        clickNum += 1
        if sleepFlag:
            time.sleep(3)
            sleepFlag = False
        else:
            time.sleep(2)
        componentPkImageRoot,sceneFilePath,pageSourceDetail = get_screen_xml(clickNum,driver)
        recordValue = getevent_position(resolution)
        # time.sleep(3)
        if len(recordValue)>0:
            if recordValue['clickType']=='tap':
                step_dict = enhanced_record_value(recordValue,componentPkImageRoot,sceneFilePath,pageSourceDetail,driver)
                step_dict['resolution'] = resolution
                record_dict[str(clickNum)]=step_dict
            elif recordValue['clickType']=='swipe':
                step_dict = {'screenshot': sceneFilePath, 'click_start_coordinate': str(recordValue['clickStartPosition'][0])+','+str(recordValue['clickStartPosition'][1]),
                             'click_end_coordinate': str(recordValue['clickEndPosition'][0])+','+str(recordValue['clickEndPosition'][1]),'clickType': recordValue['clickType'],
                             'clickStartTime': recordValue['clickStartTime'], 'screenType': recordValue['screenType'],
                             'resolution': resolution}
                record_dict[str(clickNum)] = step_dict
    write_record_json(record_dict)
    print('finished')

if __name__ == '__main__':
    driver = u2.connect(deviceName)
    # scriptFileRoot = getScriptFileRoot()
    scriptFileRoot = None
    resolution = get_resolution()
    checkIfInstalled()
    time.sleep(5)
    record_start = datetime.datetime.now()
    generate_script(scriptFileRoot, resolution, driver)
    record_end = datetime.datetime.now()
    print((record_end - record_start).seconds,' seconds recording')
    modify_global_file()

