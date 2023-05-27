import psutil
import time
import os

def getMemSize(pid):
    process = psutil.Process(pid)
    memInfo = process.memory_info()
    size = memInfo.rss / 1024 / 1024
    return size

def getTotalM(processName):
    totalM = 0
    for i in psutil.process_iter():
        if i.name() == processName:
            totalM += getMemSize(i.pid)
    print('Used Memory %.2f MB' % totalM)
    return totalM

if __name__ == "__main__":
    getTotalM('python.exe')