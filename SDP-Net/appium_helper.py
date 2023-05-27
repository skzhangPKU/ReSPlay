import os
import subprocess
import time
import pprint
import platform
from multiprocessing import Process
import signal

class AppiumLauncher:

    def __init__(self, port):
        self.port = port
        self.system = platform.system()
        self.start_appium()

    def terminate(self):
        # self.process.terminate()
        # os.system('taskkill /f /im node.exe')
        if "node" in os.popen('tasklist /FI "IMAGENAME eq node.exe"').read():
            subprocess.Popen('taskkill /f /im node.exe', shell=True,stdout=subprocess.PIPE)
            time.sleep(4.0)

    def start_appium(self):
        if self.system == 'Windows':
            # self.process = subprocess.Popen(['appium', '-p', f'{self.port}', '--log-level', 'error:error'],
            #                                 creationflags=0x00000008)
            try:
                self.terminate()
            except Exception as e:
                pass
            self.process = subprocess.Popen('appium -p 4723 --log-level error:error',shell=True,encoding='utf-8')
        else:
            self.process = subprocess.Popen(['appium', '-p', f'{self.port}', '--log-level', 'error:error'])
        # os.system('adb start-server')
        time.sleep(4.0)

    def restart_appium(self):
        self.terminate()
        self.start_appium()