#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@author: EECE AR HUD TEAM 2018
Created on Fri Jul  6 14:39:50 2018

"""









"""
This is a script that can be used to retrain the YOLOv2 model for our own dataset.
"""
########################### MAIN CODE PART IMPORT #####################################
from IPython import get_ipython # to be used as an interface with Ipython console 
import time
import os,cv2,sys
import numpy as np
import PIL #to be used for preoprocessing of data
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yad2k.models.keras_yolo import (yolo_body,yolo_eval, yolo_head, yolo_loss)
from yad2k.utils.draw_boxes import draw_boxes
from PIL import Image
from queue import Queue #to be used for queuing to get rid of image after showing it in real live video
import socket
########################### NAVIGATION PART IMPORT ####################################
import urllib.request
import json
import re as regex
import mpu
############################### AR PART IMPORT ########################################
from PIL import ImageFont
from PIL import ImageDraw 
from datetime import datetime
from threading import Thread
from flask import Flask, render_template, Response
import http.client
"""
########################################################################################
########################## INITIALIZATION OF GLOBAL VARIABLES ##########################
########################################################################################
"""
################################### IP Comfiguration ###########################
webcamIP = "10.42.0.166"
serverDataIP = "10.42.0.1"
VR_IP = "192.168.42.132"
#AIzaSyDk6T9Ap6FuWMxQlmodB3MQszzN9upWZxw


API_KEY = "AIzaSyBwMEXvIKqsn5EShsfN6TUzv1u71YB3p74"
####### BUFFER FOR SHARING DATA BETWEEN AR, NAVIGATION AND DETECTION PARTS #############
buffer=[False,None,  0  , False,     None ,          None,      False,  0,      False,  (0,0),  False,(0,0),[]]

"""
IsDirection=buffer[0]-->flag for the arrow on the ground
direction=buffer[1]-->type of the arrow on the ground[go,left,right]
distance=buffer[2]--> int for the distance of the given direction (print in a message below the arrow)
arrived=buffer[3]--> Flag if arrived for final destination 
NextRouteDirection=buffer[4]-->string for navigation notifcation[first notification bar] if (=none) no navigation ,if ((up,right,left,uturn)) put the corrsponding image 
next_Route=buffer[5]-->(string ) name of the next route
IsCalling=buffer[6]-->flag if the coming mobile name is caller ID
MobileName=buffer[7]--> name coming from mobile caller ID or song name ,if none no mobile notifcation
IsSign=buffer[8]--> flag if sign detected
SignType=buffer[9]--> detected sign type
IsCar=buffer[10] -->flag for cars
CarPos=buffer[11]--> car position [todo positionsssss]
IsPed=buffer[12]-->flag for pedstrinas 
PedPos=buffer[13]--> pedstrian position [todo postionsss]
cat_sp=buffer[14] --> car speed
answered=buffer[17] --> True if the call is answered
music_state= buffer[18]--> pause or play
"""
############################### NAVIGATION VARIABLES ###################################

Mode="driving"
isNeedRerouting=False
destinationPoint ="0.0%2c0.0"
startingPoint ="0.0%2c0.0"
currentLocation="0.0%2c0.0"
FirstTimeNav=True
distance2=None
TextToSpeechFlag=1

################################# AR VARIABLES #########################################

#********************** used to display server data for a given time ******************#
folderPath="AR"          # folder containing arrows
length_of_box=12         # length of characters in notification box 
name_counter=-1          # index of first character in song name to be dispayed in each time which increases after a specific delay to slide
counter_Delay=0          # initial value of delay for each stream of characters to be displayed before sliding one character left
name_counter_R=-1
counter_Delay_R=0
switch_flag=0            # 0 is initial value, 1 when music is on and 2 in case of calling
CallImage_counter=0      # counter for flashing call image for a specific timer
start = datetime.now()   # time of answering a call which is the start value of call counter 
last_answered=False      # initial value for call state, True if answered then increasing call couter 
Ring=True                # initial state for sliding a caller name 
First_Time=True          # indication of first value for next route and then start sliding notification box
TrafficSignFlag=0
Sign_Or_Min=False

PedCounter0=0
PedCounter1=0
PedCounter2=0

PedCounterImage0=0
PedCounterImage1=0
PedCounterImage2=0
toggle0=True
toggle1=True
toggle2=True
#**************** used to display a given sign type for a given time ******************#
Idle=False               # state for detecting a traffic sign, True if detecting none
SleepTime=0              # time for displaying a message of detected traffic sign
MaxTime=100              # max time for displaying a detected traffic sign message """TODO find relation bet MaxTime and car Speed  """
DisplayedSignType=0      # sign type to be displayed [1:No Stop , 2:Curve Left , 3:Pedstrians Crossing , 4:Bump , 5:U-turn , 6: ... , 7:Split , 8:Devided Road , 9:Bump and Pedstrians Crossing]
NumberOfSigns=0    

#********************** To avoid oscillations in cars and pedestrian ******************#
KeepPositionTime=4
CarCounter=0
PedCounter=0
ChangeCar=True
ChangePed=True
CalibrationImg=True
CarCounter=[0,0,0]
PedCounter=[0,0,0]

OldPosCar=[(),(),()]   #centres of displayed  boxes "old ones"
OldPosPed=[(),(),()]

OldCarBoxDim=[[],[],[]]   #boxes of last displayed frame
OldpedBoxDim=[[],[],[]]



w=480                   # desired image width to be resized and displayed in  
h=360                  # desired image hight to be resized and displayed in
index1=int(110/1920*w)   # initial index for cropping next route notification box
index2=int(110/1920*w)   # initial index for cropping music and calling notification box
save=False               # resize and save all images needed for specified image size
Try=True                 # only resize needed images needed for specified image size without saving
T_server=[]              # transparent image for recieved server data
T_process=[]             # transparent image for detection data
navigationCounter=-1
##################################### only for testing ####################################
arrived=False            # initial value for navigation, True when arriving to desired destination
direc=None               # initial value for direction of next route
next_route=""            # initial value for string of next route
IsCalling=False          # initial value, False when music is on and True if calling
Mob_Name=None            # name of caller or song and None for Idle state
ans=False                # initial value for answering a call
speed=1                  # initial value for speedometer

###################################### MAIN VARIABLES ######################################

DrawFrames=[]
DisplayQueue = Queue();  # a queue to hold next image to be shown in video mode
frames_captured=[]       # frames captured by webcam (each frame captured is deleted after passing to draw function )
ProcessingTime = 0.5;    # initial processing time for the first frame
YOLO_ANCHORS = np.array(((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),(7.88282, 3.52778), (9.77052, 9.16828)))
weights_file_path='saved_weights/BestWeights/trained_epoch_65_acc_45.h5'
classes_path='saved_weights/BestWeights/Temp_classes.txt'
VideoPath='AR/TestVideos/Test5.MOV'
AR_Mode=True             # Falg for displaying AR data on frames
stream=True                             # True if stream and False if WebCam

ret=None                 # initial value for correctly capturing frame by WebCam 
im=None                  # the current captured frame to be sent to detection process
video=None               # initial value for video captured from a videoPath or live WebCam
im2=[]                   # the captured frame after preprocessing to be sent for dislaying in DisplayInterval Function
GetFramesFlag=False      # Flag is set to True when capturing frist frame from WebCam  
count_occurrence=np.zeros(9)
frames_counted=np.zeros(9)
begin=np.zeros(9)
number_of_frames_averaged=10
min_occurrence=5

"""
########################################################################################
############################### NAVIGATION FUNCCTIONS ##################################
########################################################################################
"""
########################################################################################
########################################################################################
########################################################################################
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

def gen():
    goSleep = False
    while True:         
        if (goSleep):
            time.sleep(1/100)
            goSleep = False    
        else:
            time.sleep(1/500)

        Image = open("images/{}.jpg".format(0),"rb").read();

        find = Image.find(b'\xff\xd9');
        if (find != -1):         
            goSleep = True
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + Image + b'\r\n\r\n')
        else:
            goSleep = False
            


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

########################################################################################
########################################################################################
########################################################################################
def getRequestSream(ip=serverDataIP,port=80,page='serverData.html'):
    if (port == 80):       
        connection = http.client.HTTPConnection(ip,port)
    elif (port == 443):
        connection = http.client.HTTPSConnection(ip,port)
        
    connection.request('GET', '/' + page)
    response = connection.getresponse()
    answer = response.read()
    return answer

def checkRerouting(currentLoc,nextLoc):
    global destinationPoint,distance2
    startingPoint2 = currentLoc; 
    requestData = getRequestSream(ip="maps.googleapis.com",port=443,page="maps/api/directions/json?origin={}&destination={}&" \
		"mode={}&key={}".format(startingPoint2,destinationPoint,"driving",API_KEY)).decode('utf-8');
    jsonData = json.loads(requestData)
    destLat  = jsonData["routes"][0]["legs"][0]["steps"][0]["end_location"]["lat"];
    destLong = jsonData["routes"][0]["legs"][0]["steps"][0]["end_location"]["lng"];
    distance2 = jsonData["routes"][0]["legs"][0]["steps"][0]["distance"]["text"];
    nextLocation = str(destLat) + "," + str(destLong);
    if (nextLocation == nextLoc):
        return False;
    else:
        return True;


    

"""
##############################################################################################
################################### Animation Function #######################################
##############################################################################################
"""
def Animation(name=" ",IsCall=True,Ring=True,Route=False):
      """
      this function takes caller name or song name and return the wanted characters to be displayed at that time for sliding effect. 

      Parameters:
      -----------
      name: string
            caller or song name

      IsCall: bool,default True
              True if call and False if song

      Ring: bool,default True
            in case of calling, True if Ringing mode and False after answering the call

      Return:
      -------
      name:string
           name to be displayed in notification box in this time  
      """
      global background
      global length_of_box
      global name_counter,counter_Delay,name_counter_R,counter_Delay_R
      if Route==True:
                counter_Delay_R+=1
                if(counter_Delay_R==8):
                    counter_Delay_R=0
                    name_counter_R+=1
                if name_counter_R==len (name):
                   name_counter_R=0
                return name[name_counter_R:name_counter_R+length_of_box+1]
          
      if IsCall and Ring:
         name=str(name+" is calling")
      while True :
        if(len(name)<=length_of_box):

                return name
        else:            
                counter_Delay+=1
                if(counter_Delay==8):
                    counter_Delay=0
                    name_counter+=1
                if name_counter==len (name):
                   name_counter=0
                return name[name_counter:name_counter+length_of_box+1]
  

"""
##############################################################################################
################################# DrawProcess Function #######################################
##############################################################################################
"""
def DrawProcess():
    """
    this function updates a global tranparent image for AR detection data[cars,traffic signs,pedsrians] after running session on the frame to be composited with Frame and displayed

    parameters:
    -----------
        None 

    Return:
    -------
        None
    """
    global buffer,T_process 
    global Idle,SleepTime,DisplayedSignType
    global ChangeCar,ChangePed,KeepPositionTime,CarCounter,PedCounter,TempPosCar,TempPosPed,CalibrationImg ### Dina
    global IP
    global w,h
    global save,Try
    global TrafficSignFlag,Sign_Or_Min
    global PedCounter0,PedCounter1,PedCounter2
    global PedCounterImage0,PedCounterImage1,PedCounterImage2
    global toggle0,toggle1,toggle2
    
    TransparantImage = Image.new('RGBA',(w,h),(255,255,255,0)) # create a transparent image

    # getting needed data from global buffer
    IsSign=buffer[6]                          
    SignType=buffer[7]
    IsCar=buffer[8]
    CarPos=buffer[9]
    IsPed=buffer[10]
    PedPos=buffer[11]
################################################################################################
############################## for Traffic sign detection ######################################
################################################################################################
    if Idle:                                  # if True(sign is already displayed), open the same traffic sign image to be displayed again  
        SignImage=Image.open(os.path.join(folderPath,"TrafficSigns/{}.png".format(DisplayedSignType)))
        SignImage=SignImage.convert("RGBA")
        if save==True:                        # resize and save if save mode
            SignImage=SignImage.resize((int(454/1920*w),int(340/1080*h)),Image.ANTIALIAS)
            SignImage.save("TrafficSigns/{}.png".format(SignType))
        elif Try==True:                       # only resize if try mode
            SignImage=SignImage.resize((int(454/1920*w),int(340/1080*h)),Image.ANTIALIAS)
        TransparantImage.paste(SignImage, (600,750),SignImage)
        SleepTime+=1
        if(SleepTime==MaxTime):               # stop displaying traffic image after a specific time
            Idle=False
            SleepTime=0
            Sign_Or_Min=True

    if(IsSign and not Idle and TrafficSignFlag):                  # if IsSign is true display the sign if Idle is False as it's first time to be detected 
        if SignType==78:
           SignType=7
        Sign_Or_Min=True
        Idle=True
        DisplayedSignType=SignType
        SignImage=Image.open(os.path.join(folderPath,"TrafficSigns/{}.png".format(SignType)))
        SignImage=SignImage.convert("RGBA")
        if save==True:                        # resize and save if save mode
            SignImage=SignImage.resize((int(454/1920*w),int(340/1080*h)),Image.ANTIALIAS)
            SignImage.save("TrafficSigns/{}.png".format(SignType))
        elif Try==True:                       # only resize if try mode
            SignImage=SignImage.resize((int(454/1920*w),int(340/1080*h)),Image.ANTIALIAS)
        TransparantImage.paste(SignImage, (int(600/1920*w),int(750/1080*h)),SignImage)
################################################################################################
#################################### for car detection #########################################
################################################################################################
    if(IsCar):                                # if IsCar is true display,loop over boxes and display the warning sign
        for i in range(len(CarPos)):
            if(CarPos[i]!=()):
                CarImage=Image.open(os.path.join(folderPath,"TrafficSigns/Car2.png"))
                CarImage=CarImage.convert("RGBA")
                if save==True:                # resize and save if save mode
                    CarImage=CarImage.resize((int(80/1920*w),int(80/1080*h)),Image.ANTIALIAS)
                    CarImage.save(os.path.join(folderPath,"TrafficSigns/Car2.png"))
                elif Try==True:               # only resize if try mode
                    CarImage=CarImage.resize((int(80/1920*w),int(80/1080*h)),Image.ANTIALIAS) 
                TransparantImage.paste(CarImage, CarPos[i],CarImage)
################################################################################################
################################# for pedstrian detection ######################################
################################################################################################
    if(IsPed):                             # if IsPed is true display,loop over boxes and display the warning sign
        for i in range(len(PedPos)) :
            if(PedPos[i]!=()):
              if(PedPos[i][0]<int(w/3)):
                PedImage=Image.open(os.path.join(folderPath,"TrafficSigns/ped{}.png".format(PedCounterImage0)))
                PedImage=PedImage.convert("RGBA")
#                PedImage=PedImage.resize((70,70),Image.ANTIALIAS)  
                PedImage=PedImage.resize((int(100/1920*w),int(100/1080*h)),Image.ANTIALIAS)  
                TransparantImage.paste(PedImage, PedPos[i],PedImage)
                PedCounter0+=1
                if(PedCounter0 % 30==0 and toggle0==True):
                  PedCounter0=0
                  if PedCounterImage0==1:
                     toggle0=False
                  else:
                     PedCounterImage0+=1

                elif(PedCounter0 % 30==0 and toggle0==False):
                  PedCounter0=0
                  if PedCounterImage0==0:
                     toggle0=True
                  else:
                     PedCounterImage0-=1


              elif(PedPos[i][0]>w/3 and PedPos[i][0]<2*w/3):
                PedImage=Image.open(os.path.join(folderPath,"TrafficSigns/ped{}.png".format(PedCounterImage1)))
                PedImage=PedImage.convert("RGBA")
#                PedImage=PedImage.resize((70,70),Image.ANTIALIAS)  
                PedImage=PedImage.resize((int(100/1920*w),int(100/1080*h)),Image.ANTIALIAS)  
                TransparantImage.paste(PedImage, PedPos[i],PedImage)
                PedCounter1+=1
                if(PedCounter1 % 30==0 and toggle1==True):
                  PedCounter1=0
                  if PedCounterImage1==1:
                     toggle1=False
                  else:
                     PedCounterImage1+=1

                elif(PedCounter1 % 30==0 and toggle1==False):
                  PedCounter1=0
                  if PedCounterImage1==0:
                     toggle1=True
                  else:
                     PedCounterImage1-=1

              if(PedPos[i][0]>2*w/3):
                PedImage=Image.open(os.path.join(folderPath,"TrafficSigns/ped{}.png".format(PedCounterImage2)))
                PedImage=PedImage.convert("RGBA")
#                PedImage=PedImage.resize((70,70),Image.ANTIALIAS)  
                PedImage=PedImage.resize((int(100/1920*w),int(100/1080*h)),Image.ANTIALIAS)  
                TransparantImage.paste(PedImage, PedPos[i],PedImage)
                PedCounter2+=1
                if(PedCounter2 % 30==0 and toggle2==True):
                  PedCounter2=0
                  if PedCounterImage2==1:
                     toggle2=False
                  else:
                     PedCounterImage2+=1

                elif(PedCounter2 % 30==0 and toggle2==False):
                  PedCounter2=0
                  if PedCounterImage2==0:
                     toggle2=True
                  else:
                     PedCounterImage2-=1
    # update the global process transparent image
    T_process=TransparantImage   
    
    

"""
##############################################################################################
################################### DrawDirection Function ###################################
##############################################################################################
"""
def DrawDirection():
    """
    This function run on a seperat thread to update a gloabl transparent image for AR server data[next navigation Route, call, music, car speed] to be composited with Frame and displayed

    parameters:
    -----------
        None 

    Return:
    -------
        None
    """  
    global buffer,T_server,length_of_box
    global index1
    global index2
    global switch_flag
    global CallImage_counter 
    global start
    global last_answered
    global Ring
    global First_Time
    global name_counter
    global Idle,SleepTime,DisplayedSignType
    global ChangeCar,ChangePed,KeepPositionTime,CarCounter,PedCounter,TempPosCar,TempPosPed,CalibrationImg 
    global IP
    global destinationPoint,startingPoint,currentLocation,FirstTimeNav
    global w,h
    global save,Try
    global TextToSpeechFlag,TrafficSignFlag
    global OldCarBoxDim,OldPosCar,navigationCounter,Sign_Or_Min
    # initialization values for server data before recieving the current data
    cat_sp=0
    IsCalling=False
    MobileName=None
    answered=False
    music_state= "pause"
    global destinationPoint,isNeedRerouting,currentLocation,startingPoint,buffer,distance2,text_to_speech_phrase,text_to_speech_flag
    text_to_speech_phrase=""
    # imporing the used font for writing notification
    font = ImageFont.truetype("AR/fonts/Coval-Heavy.ttf", 14)
    font1 = ImageFont.truetype("AR/fonts/digital-7 (italic).ttf", 30)

    while True:
        Sign_Or_Min=False
        time.sleep(1/30);
                # get navigation data from global buffer
        IsDirection=True
        direction=buffer[1]
        distance=buffer[2]
        arrived=buffer[3]
        NextRouteDirection=buffer[4]
        next_Route=buffer[5]
        CarPos=OldPosCar
        OldCarBox=OldCarBoxDim

        # read server data
        try:
            page_source = getRequestSream().decode('utf-8')
        
            if(len(page_source)>0):
                Long,Lat,speed,trackName,callerID,destLong,destLat,callState,settings = page_source.split(':')[1].split(",")     
        except:            
            input("Error in get stream Draw Function");
            continue;
           
#        destinationPoint="30.025953,31.223558"
#        startingPoint="30.022543,31.211342"
#        print (trackName)
#        print("Lat and Long =",Lat,",",Long)
        destinationPoint=str(destLat)+"%2c"+str(destLong)
        if FirstTimeNav==True and destinationPoint!="0.0%2c0.0":
            startingPoint = str(Lat)+"%2c"+str(Long);
#            print("startingPoint=",startingPoint)
#            input()
            FirstTimeNav=False
        else:
            currentLocation= str(Lat)+"%2c"+str(Long);
#            print(currentLocation)
        if  destinationPoint=="0.0%2c0.0" or  startingPoint=="0.0%2c0.0"   or  currentLocation=="0.0%2c0.0":
#            print("inside if condition.")
            NextRouteDirection=None
        # update settings flags
        Flags=settings.split("#")
        NavigationFlag=int(float(Flags[0]))
        MusicFlag=int(float(Flags[1]))
        PhoneCallFlag=int(float(Flags[2]))
        TrafficSignFlag=int(float(Flags[3]))
        TextToSpeechFlag=int(float(Flags[4]))    
        SpeedFlag=int(float(Flags[7]))
        TempFlag=int(float(Flags[8]))
        FuelFlag=int(float(Flags[9]))
        cat_sp=int(float(speed))
        
        IsDirection=NavigationFlag

        if (callerID!="Idle" and PhoneCallFlag):        # there is a comming call, so get the caller name and set the calling state to True
           IsCalling=True
           MobileName=callerID
           if callState=="Busy":              # state of answering a call, then start a time counter for the call 
               answered=True
              
        elif trackName!="null" and callState=="Idle" and MusicFlag:               # song is being played, then get the Track name and set the music state to "play" 
            IsCalling=False
            MobileName=trackName
            music_state="play"
        elif trackName=="null" and callState=="Idle" :               # song is being played, then get the Track name and set the music state to "play" 
            MobileName=None

        
        


        # create the transparent image 
        TransparantImage = Image.new('RGBA',(w,h),(255,255,255,0))
        d = ImageDraw.Draw(TransparantImage)


#############################################################################################
#############################################################################################
        if(True):
            for i in range (len(CarPos)):
             if(OldCarBox[i]!=[]):
              if(((OldCarBox[i][1] + OldCarBox[i][3])/2)>int(810/1920*w) and ((OldCarBox[i][1] + OldCarBox[i][3])/2) <int(1050/1920*w) and (OldCarBox[i][2] <int(900/1080*h)) and (OldCarBox[i][2] >int(850/1080*h))):
                MinImage=Image.open(os.path.join(folderPath,"Pictures/3.png"))
                MinImage=MinImage.convert("RGBA")
                MinImage=MinImage.resize((int(650/1920*w),int(150/1080*h)),Image.ANTIALIAS)
###                MinImage=MinImage.resize((1100,650),Image.ANTIALIAS)
                TransparantImage.paste(MinImage, (int(OldCarBox[i][1])-int((OldCarBox[i][3]-OldCarBox[i][1])*0.2),int(OldCarBox[i][2])),MinImage)
###                image_copy.paste(MinImage, (int(OldCarBox[i][1])-int((OldCarBox[i][3]-OldCarBox[i][1])*0.7),int(OldCarBox[i][2])-int((OldCarBox[i][2]-OldCarBox[i][0])*0.7)),MinImage)
                Sign_Or_Min=True
              elif(((OldCarBox[i][1] + OldCarBox[i][3])/2)>int(810/1920*w) and ((OldCarBox[i][1] + OldCarBox[i][3])/2) <int(1050/1920*w) and (OldCarBox[i][2] <int(850/1080*h)) and (OldCarBox[i][2] >int(800/1080*h))):

                MinImage=Image.open(os.path.join(folderPath,"Pictures/4.png"))
                MinImage=MinImage.convert("RGBA")
                MinImage=MinImage.resize((int(680/1920*w),int(200/1080*h),Image.ANTIALIAS))
###                MinImage=MinImage.resize((880,400),Image.ANTIALIAS)
                
                TransparantImage.paste(MinImage, (int(OldCarBox[i][1])-int((OldCarBox[i][3]-OldCarBox[i][1])*1.1),int(OldCarBox[i][2])),MinImage)
###                image_copy.paste(MinImage, (int(OldCarBox[i][1])-int((OldCarBox[i][3]-OldCarBox[i][1])*1.5),int(OldCarBox[i][2])-int((OldCarBox[i][2]-OldCarBox[i][0])*1.35)),MinImage)
                Sign_Or_Min=True
              elif(((OldCarBox[i][1] + OldCarBox[i][3])/2)>int(810/1920*w) and ((OldCarBox[i][1] + OldCarBox[i][3])/2) <int(1050/1920*w) and (OldCarBox[i][2] <int(950/1920*w)) and (OldCarBox[i][2] >int(900/1080*h))):

                MinImage=Image.open(os.path.join(folderPath,"Pictures/2.png"))
                MinImage=MinImage.convert("RGBA")
                MinImage=MinImage.resize((int(700/1920*w),int(100/1080*h)),Image.ANTIALIAS)
###                MinImage=MinImage.resize((1300,550),Image.ANTIALIAS)
                TransparantImage.paste(MinImage, (int(OldCarBox[i][1])-int((OldCarBox[i][3]-OldCarBox[i][1])*0.18),int(OldCarBox[i][2])),MinImage)
                Sign_Or_Min=True
###                image_copy.paste(MinImage, (int(OldCarBox[i][1])-int((OldCarBox[i][3]-OldCarBox[i][1])*0.85),int(OldCarBox[i][2])-int((OldCarBox[i][2]-OldCarBox[i][0])*.6)),MinImage)#######################################################################################

#############################################################################################
################################## arrow on the ground and its message ######################
#############################################################################################
        if(IsDirection and Sign_Or_Min==False and NavigationFlag):
         if((direction == "left" or direction == "uturnl") and distance<31 and distance!=-1):    # deh bs l7ad mnzabt kol al arrows
            direction = "left"
            position = (int(570/1920*w), int(730/1080*h))  #position to diplay the arrow
            Noti_Image=Image.open(os.path.join(folderPath,"PhotoShop/call_ImageCounter2.png"))
            TransparantImage.paste(Noti_Image, (int(720/1920*w),int(27/1080*h)),Noti_Image)  # hna kda al arrow at7at 3la al sora :D
            d.text((int(830/1920*w), int(55/1080*h)),"{}M".format(distance),(255,255,255),font=font) 
         elif((direction == "right" or direction == "uturnr") and distance<31 and distance!=-1):    # deh bs l7ad mnzabt kol al arrows
            direction = "right"
            position = (int(900/1920*w), int(750/1080*h))  #position to diplay the arrow
            Noti_Image=Image.open(os.path.join(folderPath,"PhotoShop/call_ImageCounter2.png"))
            TransparantImage.paste(Noti_Image, (int(720/1920*w),int(27/1080*h)),Noti_Image)  # hna kda al arrow at7at 3la al sora :D
            d.text((int(790/1920*w), int(55/1080*h)),"{}M".format(distance),(255,255,255),font=font) 
         else:
            direction="go"
### bta3 re7ab            position = (150, 600)  #position to diplay the arrow
            position = (int(600/1920*w), int(450/1080*h))  #position to diplay the arrow
         ArrowImage = Image.open(os.path.join(folderPath,"PhotoShop/{}.png".format(direction))) # open the arrow of the given direction
 
### bta3 re7ab        ArrowImage=ArrowImage.rotate(8)
         ArrowImage=ArrowImage.convert("RGBA")

         TransparantImage.paste(ArrowImage, position,ArrowImage)  # hna kda al arrow at7at 3la al sora :D

#        d.text((200, 920),"{}M to go {}".format(distance,direction),(255,255,255),font=font) 
####################################### noti for next route #################################
#############################################################################################       
        if(NextRouteDirection!=None and NavigationFlag):        # if we have next route direction in navigation    

            if First_Time==True:             # navigation started, then notification box start sliding
                Noti_Image=Image.open(os.path.join(folderPath,"PhotoShop/upper1.png"))
                """
                if save==True:               # resize and save if save mode
                    Noti_Image=Noti_Image.resize((int(772/1920*w),int(240/1080*h)),Image.ANTIALIAS)
                    Noti_Image.save(os.path.join(folderPath,"PhotoShop/upper1.png"))
                elif Try==True:              # only resize if try mode
                    Noti_Image=Noti_Image.resize((int(772/1920*w),int(240/1080*h)),Image.ANTIALIAS)
                """
                
                """crop a slide from notification box to be displayed with increasing the cropped size each time till the end of box then display the text notification"""      
                crop_rectangle = (int(2/1920*w), 0, index1, int(240/1080*h)) 
                index1+=int(20/1920*w)
                cropped_im = Noti_Image.crop(crop_rectangle)
                TransparantImage.paste(cropped_im, (int(10/1920*w),0),cropped_im) 
                if index1 > int(770/1920*w):
                    First_Time=False
            else:                            # sliding notification box finished, so pasting full notification box 
                Noti_Image=Image.open(os.path.join(folderPath,"PhotoShop/upper1.png"))
                """
                if save==True:  
                    Noti_Image=Noti_Image.resize((int(772/1920*w),int(240/1080*h)),Image.ANTIALIAS)
                    Noti_Image.save(os.path.join(folderPath,"PhotoShop/upper1.png"))
                elif Try==True:
                    Noti_Image=Noti_Image.resize((int(772/1920*w),int(240/1080*h)),Image.ANTIALIAS)
                """

                TransparantImage.paste(Noti_Image, (int(10/1920*w),0),Noti_Image) 
                index1=int(800/1920*w)
            if arrived:                      # final destination, importing destination image 
                DestImage=Image.open(os.path.join(folderPath,"arrows/Destination.png"))
                """
                if save==True:
                   DestImage=DestImage.resize((int(130/1920*w),int(100/1080*h)),Image.ANTIALIAS)
                   DestImage.save(os.path.join(folderPath,"arrows/Destination.png"))
                elif Try==True:
                   DestImage=DestImage.resize((int(130/1920*w),int(100/1080*h)),Image.ANTIALIAS)
               """

                TransparantImage.paste(DestImage, (0,int(20/1080*h)),DestImage)
                if index1>int(770/1920*h):   # pasting text that end with "..." if length of text bigger than length of box
                   if(len(next_Route)>length_of_box): 
                       next_Route=Animation(name=next_Route,IsCall=IsCalling,Ring=Ring,Route=True)
                       d.text((int(280/1920*w),int(50/1080*h)), next_Route, font=font, fill=(255,255,255,255))
                   else:
                       d.text((int(280/1920*w),int(50/1080*h)), next_Route, font=font, fill=(255,255,255,255))

            else:                            # not arrived yet, use next route direction 
                output = Image.open(os.path.join(folderPath,'arrows/{}2.png'.format(NextRouteDirection)))
                """
                if save==True:
                    output=output.resize((int(80/1920*w),int(80/1080*h)),Image.ANTIALIAS)
                    output.save(os.path.join(folderPath,'arrows/{}2.png'.format(NextRouteDirection)))
                elif Try==True:
                    output=output.resize((int(80/1920*w),int(80/1080*h)),Image.ANTIALIAS)
                """

                TransparantImage.paste(output, (int(25/1920*w),int(31/1080*h)),output) 
                if index1>int(770/1920*w):   # pasting text that end with "..." if length of text bigger than length of box
                    if(len(next_Route)>length_of_box): 
                       next_Route=Animation(name=next_Route,IsCall=IsCalling,Ring=Ring,Route=True)
                       d.text((int(280/1920*w),int(50/1080*h)), next_Route, font=font, fill='white')
                    else:
                       d.text((int(280/1920*w),int(50/1080*h)), next_Route, font=font, fill='white')
                               
##############################################################################################
################################ noti for mobile data ########################################   
##############################################################################################
        if(MobileName!=None):                # there is a mobile notification to be displayed
            if(IsCalling):                   # there is a comming call

               # first time to get this notification, then initialize the index of cropped notification and set state to call state
               if (switch_flag==0 or switch_flag==1):   
                   index2=int(110/1920*w)
                   switch_flag=2
                   name_counter=-1
               Noti_Image=Image.open(os.path.join(folderPath,"PhotoShop/upper1.png"))
               """
               if save==True:
                  Noti_Image=Noti_Image.resize((int(772/1920*w),int(240/1080*h)),Image.ANTIALIAS)
                  Noti_Image.save(os.path.join(folderPath,"PhotoShop/upper1.png"))
               elif Try==True:
                  Noti_Image=Noti_Image.resize((int(772/1920*w),int(240/1080*h)),Image.ANTIALIAS)
                  Noti_Image.save(os.path.join(folderPath,"PhotoShop/upper1.png"))              ######################
                  print("Mobile name 744")
               """
               # display the cropped part of notification box
               crop_rectangle = (int(2/1920*w), 0, index2, int(240/1080*h))
               cropped_im = Noti_Image.crop(crop_rectangle)
               TransparantImage.paste(cropped_im, (int(10/1920*w),int(110/1080*h)),cropped_im)
               im = Image.open(os.path.join(folderPath,"Pictures/Unknown_Caller1.png"))
               """
               if save==True:
                  im=im.resize((int(90/1920*w),int(90/1080*h)),Image.ANTIALIAS)
                  im.save(os.path.join(folderPath,"Pictures/Unknown_Caller1.png"))
               elif Try==True:
                  im=im.resize((int(90/1920*w),int(90/1080*h)),Image.ANTIALIAS) #############################
               """
               TransparantImage.paste(im, (int(22/1920*w),int(138/1080*h)),im)

#############################################################################################
############################# if user answered .. start counting time #######################   
#############################################################################################
               if last_answered==True:       # call is on and continue counting
                  now = datetime.now()
                  Noti_Image=Image.open(os.path.join(folderPath,"PhotoShop/call_ImageCounter2.png"))
                  """
                  if save==True:
                      Noti_Image=Noti_Image.resize((int(300/1920*w),int(100/1080*h)),Image.ANTIALIAS)
                      Noti_Image.save(os.path.join(folderPath,"PhotoShop/call_ImageCounter2.png"))
                  elif Try==True:
                      Noti_Image=Noti_Image.resize((int(300/1920*w),int(100/1080*h)),Image.ANTIALIAS)
                      Noti_Image.save(os.path.join(folderPath,"PhotoShop/call_ImageCounter2.png")) #################
                      print("call_ImageCounter2 773")
                  """
                  TransparantImage.paste(Noti_Image, (int(720/1920*w),int(137/1080*h)),Noti_Image)  
                  minutes, seconds = divmod(((now - start).total_seconds()), 59)
                  d.text((int(790/1920*w),int(163/1080*h)), "%02d:%02d" % (minutes, round(seconds)), font=font, fill='white') 
                  Ring=False    

               elif answered==True:          # first time to answer the call, get initial time for answering
                  last_answered=True
                  start = datetime.now()
               else:
                  Ring=True

#############################################################################################    
################################ flash the green call image #################################   
#############################################################################################
               # flash the displaying of call image in ringing state but it's always displayed after answering
               if CallImage_counter>=20 or answered== True: 
                   CallImage=Image.open(os.path.join(folderPath,"Pictures/phone_call3.png"))
                   """
                   if save==True:
                      CallImage=CallImage.resize((int(40/1920*w),int(40/1080*h)),Image.ANTIALIAS)
                      CallImage.save(os.path.join(folderPath,"Pictures/phone_call3.png"))
                   elif Try==True:
                      CallImage=CallImage.resize((int(40/1920*w),int(40/1080*h)),Image.ANTIALIAS)
                      CallImage.save(os.path.join(folderPath,"Pictures/phone_call3.png")) #############################
                      print("phone_call3 798")
                   """
                   TransparantImage.paste(CallImage, (int(10/1920*w),int(120/1080*h)),CallImage)
                   if CallImage_counter>=40:
                       CallImage_counter=0
               CallImage_counter+=1

#############################################################################################    
#################################### song is playing ########################################   
#############################################################################################
            elif MusicFlag==True:                           # initialize call couter and last answer flags for next calling
               last_answered=False           
               CallImage_counter=0

               # first time to get this notification, then initialize the index of cropped notification and set state to media state
               if (switch_flag==0 or switch_flag==2):
                  index2=int(110/1920*w)
                  switch_flag=1
                  name_counter=-1

               Noti_Image=Image.open(os.path.join(folderPath,"PhotoShop/upper1.png"))
               """
               if save==True:
                  Noti_Image=Noti_Image.resize((int(772/1920*w),int(240/1080*h)),Image.ANTIALIAS)
                  Noti_Image.save(os.path.join(folderPath,"PhotoShop/upper1.png"))
               elif Try==True:
                  Noti_Image=Noti_Image.resize((int(772/1920*w),int(240/1080*h)),Image.ANTIALIAS)
                  Noti_Image.save(os.path.join(folderPath,"PhotoShop/upper1.png"))  #########################
                  print("upper1 826")
               """
               # display the cropped part of notification box
               crop_rectangle = (int(2/1920*w), 0, index2, int(240/1080*h))
               cropped_im = Noti_Image.crop(crop_rectangle)
               TransparantImage.paste(cropped_im, (int(10/1920*w),int(110/1080*h)),cropped_im)
               im = Image.open("AR/Pictures/{}.png".format(music_state))
               """
               if save==True:
                   im=im.resize((int(85/1920*w),int(85/1080*h)),Image.ANTIALIAS)
                   im.save("AR/Pictures/{}.png".format(music_state))
               elif Try==True:
                   im=im.resize((int(85/1920*w),int(85/1080*h)),Image.ANTIALIAS)
                   im.save("AR/Pictures/{}.png".format(music_state)) ####################
                   print("music_state 840")
               """
               TransparantImage.paste(im, (int(21/1920*w),int(138/1080*h)),im)

            # display notification text after finishing sliding notification box
            if index2>=int(770/1920*w):
               name=Animation(name=MobileName,IsCall=IsCalling,Ring=Ring)        
               d.text((int(280/1920*w),int(160/1080*h)), name, font=font, fill='white') 
            index2+=int(20/1920*w)
        # no notification from mobil to be displayed, initialize switch flag, last state of answering a call and call counter
        else:
            last_answered=False
            CallImage_counter=0
            switch_flag=0   
     
#############################################################################################
################################## speedometer blue logo ####################################
#############################################################################################
        if (SpeedFlag):
            SP_Image=Image.open(os.path.join(folderPath,"PhotoShop/c4.png"))
            """
            if save==True:
                SP_Image=SP_Image.resize((int(280/1920*w),int(295/1080*h)),Image.ANTIALIAS)
                SP_Image.save(os.path.join(folderPath,"PhotoShop/c4.png"))
            elif Try==True:
                SP_Image=SP_Image.resize((int(280/1920*w),int(295/1080*h)),Image.ANTIALIAS)
                SP_Image.save(os.path.join(folderPath,"PhotoShop/c4.png")) ######################
                print("c4 866")
            """
            TransparantImage.paste(SP_Image, (int(190/1920*w),int(785/1080*h)),SP_Image) 
            d.text((int(300/1920*w),int(900/1080*h)), "{}".format(cat_sp), font=font1, fill=(255,255,255,255)) 

#############################################################################################
################################# Speedometer image #########################################
#############################################################################################
            if cat_sp>=6:
                cat_sp-=6
            else:
                cat_sp=0
            SpeedoImage=Image.open(os.path.join(folderPath,"cropped/frame_{}.png".format(cat_sp)))
            SpeedoImage=SpeedoImage.convert("RGBA")
            """
            if save==True:
                SpeedoImage=SpeedoImage.resize((int(274/1920*w),int(274/1080*h)),Image.ANTIALIAS)
                SpeedoImage.save(os.path.join(folderPath,"cropped/frame_{}.png".format(cat_sp)))
            elif Try==True:
                SpeedoImage=SpeedoImage.resize((int(274/1920*w),int(274/1080*h)),Image.ANTIALIAS)
                SpeedoImage.save(os.path.join(folderPath,"cropped/frame_{}.png".format(cat_sp))) ###################
                print("cat_sp")
            """
            TransparantImage.paste(SpeedoImage, (int(193/1920*w),int(780/1080*h)),SpeedoImage) 

#############################################################################################
################################### fuel and temp image #####################################
#############################################################################################
        if (TempFlag and FuelFlag):
            FuelImage=Image.open(os.path.join(folderPath,"cropped/FUEL_1.png"))
            FuelImage=FuelImage.convert("RGBA")
            """
            if save==True:
                FuelImage=FuelImage.resize((int(120/1920*w),int(180/1080*h)),Image.ANTIALIAS)
                FuelImage.save(os.path.join(folderPath,"cropped/FUEL_1.png"))
            elif Try==True:
                FuelImage=FuelImage.resize((int(120/1920*w),int(180/1080*h)),Image.ANTIALIAS)
                FuelImage.save(os.path.join(folderPath,"cropped/FUEL_1.png")) ###########################
                print("FUEL_1 898")
            """
            TransparantImage.paste(FuelImage, (int(60/1920*w),int(850/1080*h)),FuelImage)

        # update the gloabal server transparent image
        T_server=TransparantImage


#        print(startingPoint,"   ",destinationPoint,"   ",currentLocation)
#        print("\n\n")
 


"""
###############################################################################################
##################################### Write Buffer ############################################
###############################################################################################
"""
def WriteBuffer(out_Boxes,out_classes,classes_to_be_shown):
    """
    this function split frame into 3 ranges and set the data[signFlag, signType,carFlag, carBox, pedFalg, pedBox] in the global buffer due to its range sothat there is only one detected car or pedstrian in each range to be diplayed and one traffic sign message to be shown for a specific time.

    Parameters:
    -----------
    out_Boxes: list of arrays
        each array contain [x1,y1,x2,y2] of a specific object

    out_classes: list
        classes of objects that have been detected

    classes_to_be_shown: list
        averaged traffic sign classes to be displayed in the frame

    Return:
    -------
        None                  
    """

    global OldPosCar,OldPosPed,OldCarBoxDim,OldpedBoxDim,ChangeCar,ChangePed,CarCounter,PedCounter,bottom_max
    global Idle,SleepTime,DisplayedSignType,MaxTime,NumberOfSigns
    y=416/h
    x=416/w
    Range=-1
    carFlag=False
    pedFlag=False
    signFlag=False
    
    clip=0.35
    
    carBox=[(),(),()]
    pedBox=[(),(),()]
    
    signType=-1
    bottom_max=[-1,-1,10000000]

    NewBoxDim=[[],[],[]]
    NewBoxDimPed=[[],[],[]]
    
    for i,className in enumerate(out_classes):
        top, left, bottom, right = out_Boxes[i]
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(416, np.floor(bottom + 0.5).astype('int32'))
        right = min(416, np.floor(right + 0.5).astype('int32'))
        
        centerY=int((((bottom-top)/2)+top)/y)
        centerX=int((((left-right)/2)+right)/x)
        
#        if ((right-left <=60) and (bottom-top <=60)):continue
        
        top=top/y
        left=left/x
        bottom=bottom/y
        right=right/x
        
        
        
        if(centerX <(int(1920/3))):
            Range=0
        elif(centerX>(int(1920/3)) and centerX<2*(int(1920/3))):
             Range=1
        elif(centerX > 2*(int(1920/3))):  
            Range=2
        
        if className == 0:
            carFlag=True
            if (Range==1):
              if bottom > bottom_max[Range]:
                carBox[Range]=(centerX,centerY)
                NewBoxDim[Range]=[top, left, bottom, right]
                bottom_max[Range]=bottom
            elif (Range==0):
              if right > bottom_max[Range]:
                carBox[Range]=(centerX,centerY)
                NewBoxDim[Range]=[top, left, bottom, right]
                bottom_max[Range]=right
            elif (Range==2):
              if left < bottom_max[Range]:
                carBox[Range]=(centerX,centerY)
                NewBoxDim[Range]=[top, left, bottom, right]
                bottom_max[Range]=left    
                
        elif className == 6:
            pedFlag=True
            if (Range==1):
              if bottom > bottom_max[Range]:
                pedBox[Range]=(centerX,centerY)
                NewBoxDimPed[Range]=[top, left, bottom, right]
                bottom_max[Range]=bottom
            if (Range==0):
              if right > bottom_max[Range]:
                pedBox[Range]=(centerX,centerY)
                NewBoxDimPed[Range]=[top, left, bottom, right]
                bottom_max[Range]=right
            if (Range==2):
              if left < bottom_max[Range]:
                pedBox[Range]=(centerX,centerY)
                NewBoxDimPed[Range]=[top, left, bottom, right]
                bottom_max[Range]=left 

#            if bottom > bottom_max[Range]:
#                pedBox[Range]=(centerX,centerY)
#                NewBoxDimPed[Range]=[top, left, bottom, right]
#                bottom_max[Range]=bottom
 ######################################################################################
    for i in range (3):
        if(OldPosCar[i]== () or carBox[i]==()):
            CarCounter[i]=CarCounter[i]+1
            if (CarCounter[i]==10 or OldPosCar[i]== () ):
                CarCounter[i]=0
                OldCarBoxDim[i]=NewBoxDim[i]
                OldPosCar[i]=carBox[i]
        else:
            if(carBox[i][0]<(OldCarBoxDim[i][1]+clip*(OldCarBoxDim[i][3]-OldCarBoxDim[i][1])) or carBox[i][0]>(OldCarBoxDim[i][3]-clip*(OldCarBoxDim[i][3]-OldCarBoxDim[i][1])) or carBox[i][1]<(OldCarBoxDim[i][0]+clip*(OldCarBoxDim[i][2]-OldCarBoxDim[i][0])) or carBox[i][1]>(OldCarBoxDim[i][2]-clip*(OldCarBoxDim[i][2]-OldCarBoxDim[i][0])) ):
                OldCarBoxDim[i]=NewBoxDim[i]
                OldPosCar[i]=carBox[i]
            CarCounter[i]=0

#    if (OldPosCar[1] !=() and OldPosCar[0] !=()):
#        if (((OldPosCar[1][0] - OldPosCar[0][0] ) < 120)): OldPosCar[1]=() 
#    if (OldPosCar[2] !=() and OldPosCar[1] !=()):
#        if (((OldPosCar[2][0] - OldPosCar[1][0] ) < 120)): OldPosCar[1]=() 

    for i in range (3):
        if(OldPosPed[i]== () or pedBox[i]==()):
            PedCounter[i]=PedCounter[i]+1
            if (PedCounter[i]==0 or OldPosPed[i]== () ):
                PedCounter[i]=0
                OldpedBoxDim[i]=NewBoxDimPed[i]
                OldPosPed[i]=pedBox[i]
        else:     
            if(pedBox[i][0]<(OldpedBoxDim[i][1]+clip*(OldpedBoxDim[i][3]-OldpedBoxDim[i][1])) or pedBox[i][0]>(OldpedBoxDim[i][3]-clip*(OldpedBoxDim[i][3]-OldpedBoxDim[i][1])) or pedBox[i][1]<(OldpedBoxDim[i][0]+clip*(OldpedBoxDim[i][2]-OldpedBoxDim[i][0])) or pedBox[i][1]>(OldpedBoxDim[i][2]-clip*(OldpedBoxDim[i][2]-OldpedBoxDim[i][0])) ):
                OldpedBoxDim[i]=NewBoxDimPed[i]
                OldPosPed[i]=pedBox[i]
            PedCounter[i]=0
            
########################## to choose one traffic Sign#####################################
    
    NumberOfDetectedsigns=len(classes_to_be_shown)
    if(SleepTime!=MaxTime):
        if(NumberOfSigns==2):
            signFlag=True
            signType=DisplayedSignType
            SleepTime+=1
        elif(NumberOfSigns==1 and NumberOfDetectedsigns>=1):
            for i,className in enumerate(classes_to_be_shown):
                if (className!=DisplayedSignType and className!=0 and className!=6 ):      
                    NumberOfSigns=2
                    SleepTime=-1
                    if(DisplayedSignType < className):
                        DisplayedSignType = DisplayedSignType*10+className
                    
                    elif(DisplayedSignType > className):
                        DisplayedSignType = DisplayedSignType+className*10 
            signFlag=True
            SleepTime+=1
            signType=DisplayedSignType
            
        elif(NumberOfSigns==1 and NumberOfDetectedsigns==0):
            signFlag=True
            signType=DisplayedSignType
            SleepTime+=1
            
        elif(NumberOfSigns==0 and NumberOfDetectedsigns>=1):
            for i,className in enumerate(classes_to_be_shown):
                if(NumberOfSigns<2  and className!=0 and className!=6):
                    NumberOfSigns+=1
                    if DisplayedSignType==0:
                        DisplayedSignType=className
                        signFlag=True
                        
                    elif(DisplayedSignType < className):
                        DisplayedSignType = DisplayedSignType*10+className
                    
                    elif(DisplayedSignType > className):
                        DisplayedSignType = DisplayedSignType+className*10        
            
            signType=DisplayedSignType
        
        
    else:
        SleepTime=0
        DisplayedSignType=0
        signFlag=False
        NumberOfSigns=0
#    print(OldPosCar)
#    print(pedFlag)
#    print(OldPosPed)
    # update buffer data
    buffer[6]=signFlag
    buffer[7]=signType
    buffer[8]=carFlag
    buffer[9]=OldPosCar
    buffer[10]=pedFlag
    buffer[11]=OldPosPed
    buffer[12]=OldCarBoxDim

"""
##########################################################################################
################################## Create_Model Function #################################
##########################################################################################
"""
def create_model(anchors, class_names, load_pretrained=True):
    '''
    returns the body of the model and the model
    
    # Params:
    load_pretrained: whether or not to load the pretrained model or initialize all weights
    
    # Returns:
    model_body: YOLOv2 with new output layer
    model: YOLOv2 with custom loss Lambda layer
    '''

    detectors_mask_shape = (13, 13, 5, 1) 
    matching_boxes_shape = (13, 13, 5, 5)

    # Create model input layers. 
    image_input = Input(shape=(416, 416, 3))  
    boxes_input = Input(shape=(None, 5))      #true label
    detectors_mask_input = Input(shape=detectors_mask_shape) # a return from get_detectors function
    matching_boxes_input = Input(shape=matching_boxes_shape) # a return from get_detectors function

    # Create model body.
    # Note:Model here is created without last layer as we will train last layer again every time we have new dataset. 
    yolo_model = yolo_body(image_input, len(anchors), len(class_names))
    
    """
    The next line creates model between input layer and the layer before the last layer but how it creates the model with only knowing the first layer and last layer of model ?
    each layer has a pointer to the preeceding layer so it creates model from the last layer going upwards to the first layer through pointers 
    """
    topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)
    
    if load_pretrained:
        # Save topless yolo:
        # Saving occurs only once to save model .without last layer.
        #every time after this time we saved in, we will only load data.
        topless_yolo_path = os.path.join('model_data', 'yolo_topless.h5') #get path of topless_yolo
        if not os.path.exists(topless_yolo_path):         # ask if topless_yolo exists ? if exists then only load it ,if not exists then enter the if conditional to load the full yolo model  
            print("CREATING TOPLESS WEIGHTS FILE")        #so that you can extract the topless_yolo model.
            yolo_path = os.path.join('model_data', 'yolo.h5') #yolo.h5
            model_body = load_model(yolo_path)
            model_body = Model(model_body.inputs, model_body.layers[-2].output)
            model_body.save_weights(topless_yolo_path)       
        topless_yolo.load_weights(topless_yolo_path)

    final_layer = Conv2D(len(anchors)*(5+len(class_names)), (1, 1), activation='linear')(topless_yolo.output)
  

    model_body = Model(image_input, final_layer)

    # Place model loss on CPU to reduce GPU memory usage.
    """
    model_loss is a layer to represent the loss of our model
    so it feeds the final layer outputs to yolo_loss function to calculate loss and then passes it to Model func. to generate
    new model from the input to the last layer which is now the loss layer.
    """
    
#    with tf.device('/gpu:0'):
    model_loss = Lambda(yolo_loss,output_shape=(1, ),name='yolo_loss',
        arguments={'anchors': anchors,
                   'num_classes': len(class_names)})([
                       model_body.output, boxes_input,
                       detectors_mask_input, matching_boxes_input
                   ])

    model = Model([model_body.input, boxes_input, detectors_mask_input,matching_boxes_input], model_loss)

    return model_body, model        
        
"""
##########################################################################################
################################### Get_Classes Function #################################
##########################################################################################
"""
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

"""
##########################################################################################
################################## Process_Data Function #################################
##########################################################################################
"""
def process_data(images, boxes=None):
    '''processes the data'''
    images = [PIL.Image.fromarray(i) for i in images]
    orig_size = np.array([images[0].width, images[0].height])
    orig_size = np.expand_dims(orig_size, axis=0)

    # Image preprocessing.
    processed_images = [i.resize((416, 416), PIL.Image.BICUBIC) for i in images]
    processed_images = [np.array(image, dtype=np.float) for image in processed_images]
    processed_images = [image/255. for image in processed_images]

    if boxes is not None:
        # Box preprocessing.
        # Original boxes stored as 1D list of class, x_min, y_min, x_max, y_max.
        boxes = [box.reshape((-1, 5)) for box in boxes]

        # Get box parameters as x_center, y_center, box_width, box_height, class.
        boxes_xy = [0.5 * (box[:, 3:5] + box[:, 1:3]) for box in boxes]
        boxes_wh = [box[:, 3:5] - box[:, 1:3] for box in boxes]
        boxes_xy = [boxxy / orig_size for boxxy in boxes_xy]
        boxes_wh = [boxwh / orig_size for boxwh in boxes_wh]
        
        boxes = [np.concatenate((boxes_xy[i], boxes_wh[i], box[:, 0:1]), axis=1) for i, box in enumerate(boxes)]

        # find the max number of boxes
        max_boxes = 0
        for boxz in boxes:
            if boxz.shape[0] > max_boxes:
                max_boxes = boxz.shape[0]

        # add zero pad for training
        for i, boxz in enumerate(boxes):
            if boxz.shape[0]  < max_boxes:
                zero_padding = np.zeros( (max_boxes-boxz.shape[0], 5), dtype=np.float32)
                boxes[i] = np.vstack((boxz, zero_padding))

        return np.array(processed_images), np.array(boxes)
    else:
        return np.array(processed_images)

"""
##########################################################################################
################################### Model_Body_Processing Function #######################
##########################################################################################
"""
def model_body_processing(model_body, class_names, anchors):
    '''
    function to be called once for loading weights and preparing the boxes,scores and classes 
    according to anchor boxes values,score threshold and iou threshold.
    This is evaluated by non_max_suppression function.
    '''
    global input_image_shape
    global w,h
    
    yolo_outputs = yolo_head(model_body.output, anchors, len(class_names))    
    boxes, scores, classes = yolo_eval(yolo_outputs, input_image_shape, score_threshold=0.5, iou_threshold=0.5)    
    return boxes,scores,classes

"""
##########################################################################################
###################################### GetFrames Function ################################
##########################################################################################
"""
def GetFrames():
    '''
    Function to be run on separate thread to get frames from the camera and store them in global
    frames_captured.
    time.sleep is used to determine number of frames taken per second 
    [example if(Processing time =0.1) then fps=1/0.1=10 frames per second]
    Processing time variable is equivalent to processing time taken to get and draw bounding box plus
    an offset (tolerance).
    '''
    global ProcessingTime,frames_captured,ret,DisplayQueue,im,video,im2,GetFramesFlag

    while True:
        time.sleep(1/35)
        ret, im = video.read()
        im=cv2.cvtColor((im), cv2.COLOR_RGB2BGR)
        tempImage=np.asarray(im)
        tempImage=np.expand_dims(im,axis=0)
        tempImage=process_data(tempImage)
        tempImage= Image.fromarray(np.floor(tempImage[0] * 255 + 0.5).astype('uint8'))
        tempImage=tempImage.resize((w,h),Image.ANTIALIAS)  
        im2 = tempImage.convert("RGBA")  
        GetFramesFlag=True
    
"""
##########################################################################################
###################################### VideoDraw Function ################################
##########################################################################################
"""
def getNextRoute( ):
    global destinationPoint,isNeedRerouting,currentLocation,startingPoint,buffer,distance2,text_to_speech_phrase,text_to_speech_flag
    text_to_speech_phrase="idle@idle"
    getMeOut = False;
    currentLocation = startingPoint;
   
    while(True):
        
        time.sleep(1);
        getMeOut = False
#        print("START OF WHILE TRUEEEEEEEEEEEEEE")
        if destinationPoint== "0.0%2c0.0" or startingPoint=="0.0%2c0.0" or currentLocation =="0.0%2c0.0":
            continue
#        print("################### : ", currentLocation)
        if (isNeedRerouting):
            isNeedRerouting = False
#            print("###### calculate new route #######")
#        try:  
            
        requestData = getRequestSream(ip="maps.googleapis.com",port=443,page="maps/api/directions/json?origin={}&destination={}&" \
		"mode={}&key={}".format(startingPoint,destinationPoint,"driving",API_KEY)).decode('utf-8'); 
        jsonData = json.loads(requestData)   
#        print(jsonData)
#        break
        stepsCount = len(jsonData["routes"][0]["legs"][0]["steps"]);
#        print("get step count.. ",stepsCount)
           
#        except:
#            print("Error in get request stream in Navigation 1");
#            continue
        
        
        
#        try:
#        if (isFirstTimeToRoute):
#            isFirstTimeToRoute = False
#            firstStepLat = jsonData["routes"][0]["legs"][0]["steps"][0]["start_location"]["lat"];
#            firstStepLong = jsonData["routes"][0]["legs"][0]["steps"][0]["start_location"]["lng"];          
#            distanceToFirstStep =  int(mpu.haversine_distance((float(currentLocation.split("%2c")[0]),float(currentLocation.split("%2c")[1])) , (firstStepLat,firstStepLong)) * 1000)
#            print("get first distance",distanceToFirstStep)
#            while (distanceToFirstStep > 1):     
#                distanceToFirstStep =  int(mpu.haversine_distance((float(currentLocation.split("%2c")[0]),float(currentLocation.split("%2c")[1])) , (firstStepLat,firstStepLong)) * 1000) 
#                print("distance to first step= ",distanceToFirstStep)
#        except:
#            print("Error in first time");
            
            
            
            
                   
        buffer[1]="go"
        for nxt in range(stepsCount):
#            print("in for loop .. next =",nxt)
            if (getMeOut): 
                
#                print("in if Get me out")
                break;
            if (isNeedRerouting):
#                input("in if need rerouting")
                break
            if destinationPoint== "0.0%2c0.0" or startingPoint=="0.0%2c0.0" or currentLocation =="0.0%2c0.0":
#                input("in 1")
                break
            try:
#                print(jsonData)
                distance = jsonData["routes"][0]["legs"][0]["steps"][nxt]["distance"]["text"];
                destLat  = jsonData["routes"][0]["legs"][0]["steps"][nxt]["end_location"]["lat"]; 
                destLong = jsonData["routes"][0]["legs"][0]["steps"][nxt]["end_location"]["lng"];
                isManeuver = str(jsonData["routes"][0]["legs"][0]["steps"][nxt])
                if ("maneuver" in isManeuver):       
                    maneuver = isManeuver["maneuver"]
#                    print("maneuver  ",maneuver)
                else:
                     
    
                     maneuver = 'null';  
#                     print("maneuver else  ",maneuver) 
            except:
#                print("Error in 2")
                break

            try:
                
                instruction = regex.sub('<[^<]+?>', '#', jsonData["routes"][0]["legs"][0]["steps"][nxt]["html_instructions"]); 
                text=regex.sub('<[^<]+?>', ' ', jsonData["routes"][0]["legs"][0]["steps"][nxt]["html_instructions"]); 
                Result = instruction.split('#')
                
                left=["turn-slight-left","turn-sharp-left","turn-left","ramp-left","fork-left"]
                right=["turn-slight-right","turn-sharp-right","turn-right","ramp-right","fork-right"]
                go=["straight"]
                uturn_left=[ "uturn-left","roundabout-left"]      
                uturn_right=[ "uturn-right","roundabout-right"]                                  
                if(maneuver in left): nextroutedirec="left"
                if(maneuver in right): nextroutedirec="right"
                if((maneuver in go) or maneuver=="null" ): nextroutedirec="go"
                if(maneuver in uturn_left): nextroutedirec="uturnl"
                if(maneuver in uturn_right): nextroutedirec="uturnr"
    
                            #distance
                buffer[4]=nextroutedirec            #NextRouteDirection
                buffer[5]=Result[3]            #next_Route  
                temp=urllib.parse.quote_plus("direction@After {} {}".format(distance,text))
            except:
#                print("Error in 3");
                break;

            if( urllib.parse.quote_plus(text_to_speech_phrase) != temp and TextToSpeechFlag==1):
#                print("in 1406")
                text_to_speech_phrase = "direction@After {} {}".format(distance,text)
                try:
#                    print("get request stream")
                    getRequestSream(page="textToSpeech.php?data={}".format(text_to_speech_phrase))
                except:
#                    print("Error in get request stream Text to speech Navigation function");
                    continue;
     

            nextLocation = str(destLat) + "," + str(destLong); 
            distanceToNext =  int(mpu.haversine_distance((float(currentLocation.split("%2c")[0]),float(currentLocation.split("%2c")[1])) , (destLat,destLong)) * 1000)                  
            while (distanceToNext > 5): 
#                    print(distanceToNext)
                    buffer[2]= distanceToNext 
                    distanceToNext =  int(mpu.haversine_distance((float(currentLocation.split("%2c")[0]),float(currentLocation.split("%2c")[1]) ), (destLat,destLong)) * 1000)                  
                    if destinationPoint== "0.0%2c0.0" or startingPoint=="0.0%2c0.0" or currentLocation =="0.0%2c0.0":
                        getMeOut = True
                        break;
                    try:
                        isNeedRerouting = checkRerouting(currentLocation,nextLocation);
                    except:
#                        print("########## 3")
                        break
#                    print(isNeedRerouting)
                    if (isNeedRerouting):
                        startingPoint = currentLocation;
                        getMeOut = True
                        break
                                      

 
            if (not isNeedRerouting and nxt==stepsCount):
#                print(nxt,"        ",stepsCount)
                buffer[3]=True
                break

def videoDraw(model_body, class_names, anchors):
    '''
    Main function of drawing that controls all actions 
    '''
    global video,ProcessingTime,input_image_shape,frames_captured,DrawFrame,ret,im,stream
    global w,h
    input_image_shape = K.placeholder(shape=(2, ))    
    boxes,scores,classes=model_body_processing(model_body, class_names, anchors) #called once to avoid adding nodes to graph

    ########## CALIBRATION FOR SESS RUN AS FIRST RUN TAKES MUCH TIME #######
    calibrated=cv2.imread('Calibration.bmp')   
    calibrated=cv2.resize(calibrated,(416,416))
    calibrated = cv2.cvtColor(calibrated, cv2.COLOR_BGR2RGB) 
    calibrated = np.reshape(calibrated,(-1,416,416,3))
    draw(boxes,scores,classes,model_body, class_names, calibrated)   
    ######### END OF CALIBRATION ###########
    
    ##### start capturing from web_cam (frame is captured with size 416x416) ####
    if stream==True:
        video = cv2.VideoCapture('http://{}:8080/video'.format(webcamIP))

    else: 
        video = cv2.VideoCapture(1)
    video.set(3,h);
    video.set(4,w);

    #### multithreading between : 1) server data transparent image thread 2)Displaying video after composition with AR transparent images 3) Capturing video frames ####
    #pool=ThreadPool(processes=3)
    #pool.apply_async(DrawDirection,[])
    #pool.apply_async(display_interval)
    #pool.apply_async(GetFrames, [])
#    pool.apply_async(getNextRoute)
    
    thread1 = Thread(target = DrawDirection)
    thread2 = Thread(target = display_interval)
    thread3 = Thread(target = GetFrames)
    thread4 = Thread(target = getNextRoute)
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    while (True):
        time.sleep(1/30);
        if(ret):
           frames_captured_p=np.asarray(im) #frames captured are passed to another temp variable (frames captured_p)
           frames_captured_p=np.expand_dims(frames_captured_p, axis=0)
           frames_captured_p=process_data(frames_captured_p)#pass frames captured to be preprocessed
           draw(boxes,scores,classes,model_body, class_names, frames_captured_p[:]) #pass frames captured to draw function to add bounding boxes 
"""
##########################################################################################
###################################### draw Function #####################################
##########################################################################################
"""
def draw(boxes,scores,classes,model_body, class_names,image_data):
    '''
    this function apply processing to images, run session to produce boxes and classes of detected objects then draw boxes on frame in case of no AR mode or average classes, update buffer and update the global process transparent image in case of AR mode

    Parameters:
    -----------
    ################ i don't remmeber them !

    return:
    -------
        None
    '''
    global input_image_shape
    global buffer,im,T_process,ret
    global next_route,IsCalling,Mob_Name,ans,music_state,direc,arrived,speed
    while ((not ret) and im !=None):
        continue
    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

    # run session, get boxes and classes 
    out_boxes, out_scores, out_classes = sess.run([boxes, scores, classes],
            feed_dict={
                model_body.input: image_data[:],
                input_image_shape: [416, 416]
                ,K.learning_phase():0 #testing phase
            })
        
    if AR_Mode:            # get the averaged classes of traffic signs then update the global transparent process image
       classes_to_be_shown=average_classes(out_classes)
       WriteBuffer(out_boxes,out_classes,classes_to_be_shown)
       DrawProcess()
    else:                  # draw boxes on objects in the frame
       image_with_boxes = draw_boxes(image_data[:][0], out_boxes, out_classes,class_names, out_scores) 

"""
##########################################################################################
###################################### display_interval Function #########################
##########################################################################################
"""
def display_interval():
    '''
    Function to be run on separate thread to display frames after composition with server transparent image and process transparent image

    Parameters:
    -----------
         None

    Return:
    -------
         None
    '''
    global ret,T_server,T_process,im2
    global w,h,GetFramesFlag
    # initialize images
    T_server=Image.new('RGBA',(w,h),(255,255,255,0))
    T_process=Image.new('RGBA',(w,h),(255,255,255,0))
    counter = 0;
    while (True):
        time.sleep(1/30)
        if (GetFramesFlag==True): # CHECK IF FIRST FRAME IS CAPTURED
            #hnnnnnnnnnnnnnnnnnnnnnnna cooooooooooode
#            print(im2.shape,T_server.size)
            out1 = Image.alpha_composite(im2, T_server)
            out2= Image.alpha_composite(out1, T_process)
            out2=cv2.cvtColor(np.array(out2), cv2.COLOR_RGB2BGR)           
            cv2.imwrite("images/{}.jpg".format(0),out2);
            
            #print("Time:",time.time()-w)
#            cv2.imshow('r',(out2))
#            if cv2.waitKey(1) and 0xFF == ord('q'):break;
#    cv2.destroyAllWindows()

"""
##########################################################################################
################################ Average classes Function ################################
##########################################################################################
"""
def average_classes(out_classes):
    """
    Average function to eliminate False alarms of traffic signs
   
    Parameters:
    -----------
    out_classes: list
         output classes of detection objects in the frame 
    """    
    global begin,count_occurrence,frames_counted,number_of_frames_averaged,min_occurrence
    show_class=[]
    out_classes=list(np.array(out_classes))

    for z in out_classes:
        begin[z]=1
        count_occurrence[z]= count_occurrence[z]+1 
    frames_counted=[frames_counted[f]+1 if t==1 else frames_counted[f] for f,t in enumerate(begin)]
    frames_counted=list(np.asarray(frames_counted,dtype=np.int32))
    show_class=[show_class for show_class,x in enumerate(count_occurrence) if x > min_occurrence]
    show_class=list(np.asarray(show_class,dtype=np.int32))

    if (any(r==number_of_frames_averaged for r in frames_counted)):        
        indices_frames_counted=[]
        indices_frames_counted_temp=list(np.nonzero(np.array(frames_counted) >= number_of_frames_averaged))
        for i in range (len(indices_frames_counted_temp[0])):
            indices_frames_counted.append(indices_frames_counted_temp[0][i])

        for z in indices_frames_counted:
            frames_counted[z]=0 
            count_occurrence[z]=0
            begin[z]=0
            frames_counted=list(np.asarray(frames_counted,dtype=np.int32))

            count_occurrence=list(np.asarray(count_occurrence,dtype=np.int32))
            begin=list(np.asarray(begin,dtype=np.int32))

    return show_class


##########################################################################################
##########################################################################################            
##########################################################################################            
##########################################################################################

"""
##########################################################################################
######################################### MAIN ###########################################
##########################################################################################
"""
def videoDrawThread():
    class_names = get_classes(classes_path)          # load the classes names
    anchors = YOLO_ANCHORS                           
    model_body, model = create_model(anchors, class_names) 
    model.load_weights(weights_file_path)
    videoDraw(model_body,class_names,anchors) 
    
def _main():
    #####################################################
    #########ؤ############################################
#    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#    s.connect(('8.8.8.8', 1))  # connect() for UDP doesn't send packets
#    local_ip_address = s.getsockname()[0]
#    
    thread = Thread(target = videoDrawThread)
    thread.start();
    app.run(host=VR_IP,port=2000,threaded=True, debug=True)
   

    #####################################################
    ####################################################    
                         
if __name__ == '__main__':
   try:
       _main()
   except KeyboardInterrupt: #catch CTRL+C press
       print("                   ###### LIVE VIDEO RELEASED ########      ")
       video.release() #release camera resources after pressing CTRL+C
       get_ipython().magic('%reset -sf') #delete all variables after pressing CTRL+C
