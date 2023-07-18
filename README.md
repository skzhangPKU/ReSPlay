## ReSPlay
---
A cross-platform record-and-replay tool for mobile apps

All the evaluation artifacts are available here, which include tools, apps we used, and instructions about how to run the tool on mobile phones. Details can be found in our paper. It is a useful record-and-replay tool, which leverages a most robust visual feature, GUI sequences, to guide replaying more accurately.

#### Framework:
![Framework](https://github.com/skzhangPKU/ReSPlay/blob/master/Figures/overview.png)
---

## All Experimental Apps
---
The selected apps are Keep, Booking, Amazon Shopping, Evernote, App Music, Kindle, AdGuard, HERE WeGo, Tricount, Wikipedia, Monkey, and openHAB. The categories they belong to include health & fitness, travel, shopping, productivity, music, books, personalization, tools, finance, news, development, and lifestyle.
Experimental apps are available to download from [this link](https://drive.google.com/file/d/161DLXEDe7S4WPCPOzCVDpiZQBMzdoeTR/view?usp=sharing).


## Environment settings
---
  * Python 3.9.6
  * ADB
  * Appium
  * Tesseract
  
**Step One: ADB Install**
1. Get the Latest SDK Platform-tools From Android Studio's [SDK Manager](https://developer.android.com/studio/intro/update#sdk-manager) or From the [Sdkmanager](https://developer.android.com/studio/command-line/sdkmanager) Command-line Tool. Once you’ve downloaded the Platform Tools package, extract the contents of the .zip file to a folder (like “C:\Android\platform-tools”).

2. Configure the PATH Variable. The PATH variable is a master list of where to look for command line tools. For details, please refer to [this link](https://lifehacker.com/the-easiest-way-to-install-androids-adb-and-fastboot-to-1586992378).

3. Enable USB Debugging on Mobile Phones.

4. Test ADB (if Needed).
The third and fourth steps can refer to [this link](https://www.howtogeek.com/125769/how-to-install-and-use-abd-the-android-debug-bridge-utility/).

**Step Two: Appium Install**

&nbsp;&nbsp;&nbsp;&nbsp;The installation process can refer to [this link](http://appium.io/docs/en/2.0/quickstart/install/).

**Step Three: Dependency Library Installation**

&nbsp;&nbsp;&nbsp;&nbsp; Run the following command to install the Python libraries:
   ```sh
   pip install -r requirements.txt
   ```
  
**Step Four: Setup App**

&nbsp;&nbsp;&nbsp;&nbsp; Install the app on the mobile device:

   ```sh
   adb install XXX.apk
   ```

 **Step Five: Pretrained Models**
 
 Our pretrained models are optional to download from [this link](https://drive.google.com/file/d/1kuqT7qRvBzn4kCiiJMlaSMSLfwIq3XQl/view?usp=sharing).
 

## Record (UIRecorder)
---
The recoding phase aims to synchronously extract critical information from input events to provide support for sebsequent analysis. Specifically, ReSPlay records GUI screenshots, layout files, and widget screenshots. 

Users such as developers or testers perform a series of operations on mobile devices. Each operation will be responded to by the device's sensors in real-time and sent to the kernel in the form of event streams. The absolute position information for each operation is then identified and extracted. For instance, such events are stored in an external device file `/dev/input/event*` for the Android platform. 

The operated widget is found based on the recorded hierarchies and extracted coordinates using a recursive method. Widget screenshots are cropped from GUI screenshots based on widget coordinates.

**The process of the recording phase is as follows.**

1. Check and modify the `config` file.

Minor amendments to the config file are required, which include `deviceName`, `pkName`, `activityName`, `res_x`, and `res_y`. `res_x` and `res_y` indicate the device resolution in the x and y dimensions. `pkName` and `activityName` represent package name and lunchable activity name of apps.

To retrieve the `deviceName`, `res_x`, `res_y`, `pkName`, and `activityName`, run the following commands:
```
deviceName: adb devices
resolution: adb shell wm size
package and activity name: adb shell dumpsys window | findstr "mCurrentFocus"
```
2. Start the recording process.
```
python getPosition.py
```

**After the abovementioned processes, some necessary contents are automatically parsed and stored in a nested directory.**

The directories named with the package name contain five folders/scenarios, each of which includes widget screenshots (see Figure 1), layout files (see Figure 2), and GUI screenshots (see Figure 3).


#### Figure 1:
![figure2](https://github.com/skzhangPKU/ReSPlay/blob/master/Figures/widget_screenshots.jpg)

#### Figure 2:
![figure3](https://github.com/skzhangPKU/ReSPlay/blob/master/Figures/layout_files.jpg)

#### Figure 3:
![figure4](https://github.com/skzhangPKU/ReSPlay/blob/master/Figures/GUI_screenshots.jpg)

Specifically, the scenarios in our experiments are available [here](https://drive.google.com/file/d/1d785NGGPRIJ1Au82KZk121h2PiB8p5Se/view?usp=sharing).

## Replay (SDP-Net)
---
#### Step One: Run monkeyrunner to simulate random clicks by developers and produce contents in the same format as the recording phase.

&nbsp;&nbsp;&nbsp;&nbsp;The specific operation procedures can refer to the [previous work](https://github.com/fxlysm/PYAndroid_Test/blob/50dbd3aa6aa3e2514230bc82f7e2839fc969ebd5/com/monkey/RandomMonkey.py).

#### Step Two: Start offline training.

1. Modify global variables in the `config` file, such as `REPLAY_RESOLUTION_X`  and `REPLAY_RESOLUTION_Y`.
   
   It can be obtained by the method mentioned above. `adb shell wm size`.

2. Start the Appium server, which will interact with mobile devices on various platforms such as Android and iOS.

   To open Appium, type Appium followed by the IP address and the server port number, as shown in Figure below.
   
   Now, Appium is running and the REST HTTP is listening on the IP address (In this example, localhost 0.0.0.0 and the server port 4723).


<div align="center">
	<img src="https://github.com/skzhangPKU/ReSPlay/blob/master/Figures/appium_server_start1.png" width="250">
 
 <img src="https://github.com/skzhangPKU/ReSPlay/blob/master/Figures/appium_server_start2.png" width="250">
</div>

3. Run the training script, which will output an offline trained model. The trained model will be used to search for potential event traces and replay them on the target device.
```
python train.py
```

&nbsp;&nbsp;&nbsp;&nbsp;The Appium server interacts with the mobile device successfully when it receives a response with status 200 (see the Figure below).
<div align="center">
	<img src="https://github.com/skzhangPKU/ReSPlay/blob/master/Figures/train_appium_server.png" width="250">
</div>

&nbsp;&nbsp;&nbsp;&nbsp;Our tool will output the contents like the Figure below when it runs successfully.
<div align="center">
<img src="https://github.com/skzhangPKU/ReSPlay/blob/master/Figures/ReSPlay_train.png" width="250">
</div>

#### Step Three: Start the replaying phase.

&nbsp;&nbsp;&nbsp;&nbsp;The first step is to move recorded `traces` in the recording phase to `imageFile` directory of SDP-Net. Then, run the following commands.

```
python inference.py
```
&nbsp;&nbsp;&nbsp;&nbsp;During this phase, ReSPlay will automatically record the GUI screenshots on the replaying device, which facilitates a visual check on whether a replay is correct.
The result is shown in the Figure below.

<img src="https://github.com/skzhangPKU/ReSPlay/blob/master/Figures/replay_example.png" width="600">

## Plan for Improvement
---

1. Consider how to effectively replay event traces on apps that are quite different on different platforms.

2. Consider using the source code to improve recordand-replay for apps developed by cross-platform solutions like
Flutter or ReactNative.
