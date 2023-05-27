import os
import platform
from time import time
import subprocess

def call_adb(command, deviceId = None):
    command_result = ''
    if deviceId is None:
        command_text = 'adb %s' % command
    else:
        command_text = 'adb -s %s %s' % (deviceId, command)
    results = os.popen(command_text, "r")
    while 1:
        line = results.readline()
        if line == '':
            break
        command_result += line
    results.close()
    return command_result

# get devices
def attached_devices():
    result = call_adb("devices")
    devices = result.partition('\n')[2].replace('\n', '').split('\tdevice')
    deviceList = []
    for device in devices:
        if len(device) > 2:
            deviceList.append(device)
    return deviceList

# check device connect or not
def check_connected(deviceId):
    result = get_state(deviceId)
    if result == 'device':
        return True
    else:
        return False

#get state
def get_state(deviceId):
    result = call_adb("get-state", deviceId)
    result = result.strip(' \t\n\r')
    return result or None

#push file to phone
def push(local, remote, deviceId):
    result = None
    if ls(remote, deviceId) is True:
        result = call_adb("push %s %s" % (local, remote), deviceId)
    return result

#pull file from phone
def pull(remote , local, deviceId):
    result = None
    if ls(remote, deviceId) is True:
        result = call_adb("pull %s %s" % (remote, local), deviceId)
    return result

#delete file from phone
def delete(filePath, deviceId):
    result = call_adb("shell rm -rf %s" % filePath, deviceId)
    return repr(result)

#mkdir
def mkdirs(dirPath, deviceId):
    result = None
    #mkdir if dir is not exist
    if ls(dirPath, deviceId) is False:
        result = call_adb("shell mkdir -p %s" % dirPath, deviceId)
    return result

#list files or dir
def ls(path, deviceId):
    result = call_adb("shell ls -l %s" % path, deviceId)
    if "No such file or directory" in result:
        return False
    else:
        return True

#clear app data
def clear(packageName, deviceId):
    result = call_adb("shell pm clear %s" % packageName, deviceId)
    return result

#force stop apk
def forceStop(packageName, deviceId):
    result = call_adb("shell am force-stop %s" % packageName, deviceId)
    return result

#capture screen
def screencap(saveFile, deviceId):
    sysstr = platform.system()
    if sysstr == "Linux":
        call_adb("shell screencap -p  | sed \'s/\\r$//\' >  %s" % saveFile, deviceId)
    elif sysstr == "Darwin" or sysstr=="Windows":
        tmpDir = '/data/local/tmp'
        mkdirs(tmpDir, deviceId)
        # remoteFile = os.path.join(tmpDir, "phoneHomePage.png")
        remoteFile = tmpDir+'/'+'phoneHomePage.png'
        call_adb("shell screencap -p %s" % remoteFile, deviceId)
        pull(remoteFile, saveFile, deviceId)
        delete(remoteFile, deviceId)

#record video
def screenrecord(saveFile, deviceId):
    result = call_adb("shell screenrecord %s" % saveFile, deviceId)
    return result

def run(cmd, deviceId):
    return subprocess.check_output(('adb -s %s %s' % (deviceId, cmd)).split(' '))

#get pageSource(xml) of current UI
def dump_layout(dump_file, deviceId):
    startDumpTime = time()
    #call_adb('shell uiautomator dump %s' % dump_file, deviceId)
    run('shell uiautomator dump %s' % dump_file, deviceId)
    endDumpTime = time()
    dumpTime = round(endDumpTime-startDumpTime, 4)
    return dumpTime

def open_app_detail(package, deviceId):
    # adb shell am start -a ACTION -d DATA
    intent_action = 'android.settings.APPLICATION_DETAILS_SETTINGS'
    intent_data = 'package:%s' % package
    call_adb('shell am start -a %s -d %s' % (intent_action, intent_data), deviceId)

def force_stop(package, deviceId):
    # adb shell am force-stop <package>
    call_adb('shell am force-stop %s' % package, deviceId)

def start_activity(activity, deviceId):
    # adb shell am start -n <activity>
    call_adb('shell am start -n %s' % activity, deviceId)


def clear_data(package, deviceId):
    # adb shell pm clear <package>
    call_adb('shell pm clear %s' % package, deviceId)

def keyboard_input(text, deviceId):
    # adb shell input text <string>
    call_adb('shell input text %s' % text, deviceId)

def keyboard_back(deviceId):
    # adb shell input keyevent 4
    call_adb('shell input keyevent 4', deviceId)

def uninstall(package, deviceId):
    #adb -s <deviceId> uninstall <package>
    call_adb('uninstall %s' % package, deviceId)

def keyboard_home(deviceId):
    #adb shellinput keyevent 3
    call_adb('shell input keyevent 3', deviceId)

def checkPackageInstall(pkName, deviceId):
    result = call_adb('shell pm list package | grep %s' % pkName, deviceId)
    result = result.replace('\r\n', '').strip()
    if result == '':
        return False
    else:
        packageName = result[8:]  #8 is length of package:
        if pkName in packageName:
            return True
        else:
            return False
