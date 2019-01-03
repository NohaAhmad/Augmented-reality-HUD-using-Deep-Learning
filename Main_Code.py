#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 22:19:28 2018

@author: EECE AR HUD TEAM 2018
"""
"""
#####################################################################################
###################################### IMPORT LIBRARIES #############################
#####################################################################################
"""
from IPython import get_ipython # to be used as an interface with Ipython console 
import time
import os,cv2
from multiprocessing.pool import ThreadPool #to be used for Multithreading
import numpy as np
import PIL #to be used for preoprocessing of data
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yad2k.models.keras_yolo import (yolo_body,yolo_eval, yolo_head, yolo_loss)
from yad2k.utils.draw_boxes import draw_boxes
from matplotlib import pyplot as plt
import sys
from AR.AR_semiFinal2 import DrawDirection, WriteBuffer,gauge,Cat_edit
#import imutils
from PIL import Image
from queue import Queue #to be used for queuing to get rid of image after showing it in real live video
import urllib.request
import socket

"""
#####################################################################################
###################################### GLOBAL VARIABLES #############################
#####################################################################################
"""
##################################### JUST FOR TESTING ##############################
#next_route="Alhosary"
#IsCalling=False
#Mob_Name="el sood 3yonooo .. ya wlaa"
#ans=False
#music_state="play"
#direc="right"
#SignType=1
#arrived=False
#buffer=[False ,None,  0  , arrived  ,    direc ,     next_route,     IsCalling,    Mob_Name,      True,  SignType,      False,  (0,0),  False,(0,0),  50,      5,       10, ans, music_state]


#next_route=""
#IsCalling=False
#Mob_Name=None
#ans=False
#music_state="play"
#direc="right"
#arrived=False
#speed=1
q=0 # counter used for testing in draw function 

################################## MAIN VARIABLES ####################################

####### BUFFER FOR SHARING DATA BETWEEN AR, NAVIGATION AND DETECTION PARTS #############
buffer=[False,None,  0  , False,     None ,          None,      False,  0,      False,  (0,0),  False,(0,0)]

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

IP=[l for l in ([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")][:1], [[(s.connect(('8.8.8.8', 53)), s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) if l][0][0]
imageQueue = Queue(); #a queue to hold next image to be shown.(The image is dequued to temp variable (a variable in display_interval func) to be shown)
frames_captured=[] #frames captured by webcam (each frame captured is deleted after passing to draw function )  
ProcessingTime = 0.5; #initial processing time for the first frame
YOLO_ANCHORS = np.array(((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),(7.88282, 3.52778), (9.77052, 9.16828)))
weights_file_path='saved_weights/BestWeights/trained_epoch_65_acc_45.h5'
classes_path='saved_weights/BestWeights/Temp_classes.txt'
VideoPath='AR/TestVideos/Test5.MOV'
AR_Mode=True
Live=False
ret=None
################################## AVERAGE VARIABLES ##################################
count_occurrence=np.zeros(9)
frames_counted=np.zeros(9)
start=np.zeros(9)
number_of_frames_averaged=6
min_occurrence=3



"""
#####################################################################################
###################################### MAIN FUNCIONS ################################
#####################################################################################
"""
def _main():
    class_names = get_classes(classes_path) #loads the classes names
    anchors = YOLO_ANCHORS
    model_body, model = create_model(anchors, class_names) 
    model.load_weights(weights_file_path)
    videoDraw(model_body,class_names,anchors) 

"""
#####################################################################################
############################## create_model FUNCIONS ################################
#####################################################################################
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
    image_input = Input(shape=(416, 416, 3))  #
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
        # TODO: Replace Lambda with custom Keras layer for loss.
    model_loss = Lambda(yolo_loss,output_shape=(1, ),name='yolo_loss',
        arguments={'anchors': anchors,
                   'num_classes': len(class_names)})([
                       model_body.output, boxes_input,
                       detectors_mask_input, matching_boxes_input
                   ])

    model = Model([model_body.input, boxes_input, detectors_mask_input,matching_boxes_input], model_loss)

    return model_body, model        
        
"""
#####################################################################################
############################## get_classes FUNCTION #################################
#####################################################################################
"""   
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

"""
#####################################################################################
############################## process_data FUNCTION ################################
#####################################################################################
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
#####################################################################################
########################### model_body_processing FUNCTION ##########################
#####################################################################################
"""

def model_body_processing(model_body, class_names, anchors):
    '''
    function to be called once for loading weights and preparing the boxes,scores and classes 
    according to anchor boxes values,score threshold and iou threshold.
    This is evaluated by non_max_suppression function.
    '''
    global input_image_shape
        
    yolo_outputs = yolo_head(model_body.output, anchors, len(class_names))    
    boxes, scores, classes = yolo_eval(yolo_outputs, input_image_shape, score_threshold=0.5, iou_threshold=0.5)    
    return boxes,scores,classes 


"""
#####################################################################################
################################# GetFrames FUNCTION ################################
#####################################################################################
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
    global ProcessingTime,frames_captured,ret,DisplayQueue
    if Live:
        print("                   ###### STARTING LIVE VIDEO ########      \n\n")  
        while(True):
            ret, im = video.read()
            time.sleep(ProcessingTime)
            im=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            frames_captured.append(im)
    else:
        while(video.isOpened()):
            ret, im = video.read()
            time.sleep(ProcessingTime)
            im=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            frames_captured.append(im)

"""
#####################################################################################
################################# videoDraw FUNCTION ################################
#####################################################################################
"""
def videoDraw(model_body, class_names, anchors):
    '''
    Main function of drawing that controls all actions
    '''
    global video,ProcessingTime,input_image_shape,frames_captured,DrawFrame,ret   
    input_image_shape = K.placeholder(shape=(2, ))    
    boxes,scores,classes=model_body_processing(model_body, class_names, anchors) #called once to avoid adding nodes to graph

    ########## CALIBRATION FOR SESS RUN #######
#    calibrated=cv2.imread('/home/dina/yad2k/Calibration.bmp')   
#    calibrated=cv2.resize(calibrated,(416,416))
#    calibrated = cv2.cvtColor(calibrated, cv2.COLOR_BGR2RGB) 
#    calibrated = np.reshape(calibrated,(-1,416,416,3))
#    draw(boxes,scores,classes,model_body, class_names, calibrated)   
    ######### END OF CALIBRATION ###########
    
    ##### start capturing from web_cam (frame is captured with size 416x416) #### 
    if Live: 
        video = cv2.VideoCapture(0)
#    video.set(3,416);
#    video.set(4,416);
    else:
        print("opening video")
        video = cv2.VideoCapture( VideoPath)
    
    ####multithreading between : 1)Capturing video frames 2)Displaying video after adding bounding boxes 3)processing on images captured ####
    pool=ThreadPool(processes=2)
    pool.apply_async(GetFrames, [])
    pool.apply_async(display_interval,[])
    if Live:
        while (True):
            sTime = time.time();
            if(len(frames_captured) > 0):
                frames_captured_p=np.asarray(frames_captured[:]) #frames captured are passed to another temp variable (frames captured_p)
                del frames_captured[:] # frames_captured are deleted after being passed to other temp variable (frames_captured_p) to avoid memory overflow
                frames_captured_p=process_data(frames_captured_p)#pass frames captured to be preprocessed
                out_classes=draw(boxes,scores,classes,model_body, class_names, frames_captured_p[:]) #pass frames captured to draw function to add bounding boxes 
                classes_to_be_shown=average_classes(out_classes) ### to be passed to AR code
                eTime=time.time()
                ProcessingTime = (eTime-sTime) * 1.1 #Processing time which controls the fps of video capture
    else:
        while (video.isOpened()):
            if ret==False:break
            sTime = time.time();
            if(len(frames_captured) > 0):
                frames_captured_p=np.asarray(frames_captured[:]) #frames captured are passed to another temp variable (frames captured_p)
                del frames_captured[:] # frames_captured are deleted after being passed to other temp variable (frames_captured_p) to avoid memory overflow
                frames_captured_p=process_data(frames_captured_p)#pass frames captured to be preprocessed
                
                out_classes=draw(boxes,scores,classes,model_body, class_names, frames_captured_p[:]) #pass frames captured to draw function to add bounding boxes 
                classes_to_be_shown=average_classes(out_classes) ### to be passed to AR code
                eTime=time.time()
                ProcessingTime = (eTime-sTime) * 1.1 #Processing time which controls the fps of video capture

"""
#####################################################################################
##################################### Draw FUNCTION #################################
#####################################################################################
"""
def draw(boxes,scores,classes,model_body, class_names,image_data):
    '''
    Draw bounding boxes on image datac
    '''
    global input_image_shape ,imageQueue,q
    global buffer
    global next_route,IsCalling,Mob_Name,ans,music_state,direc,arrived,speed
 
    image_data = np.array([np.expand_dims(image, axis=0) for image in image_data])
    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.
    for i in range(len(image_data)):
        out_boxes, out_scores, out_classes = sess.run([boxes, scores, classes],
            feed_dict={
                model_body.input: image_data[i],
                input_image_shape: [416, 416]
                ,K.learning_phase():0 #testing phase
            })

        if AR_Mode:
            classes_to_be_shown=average_classes(out_classes)            
            buffer[6],buffer[7],buffer[8],buffer[9],buffer[10],buffer[11]=WriteBuffer(out_boxes,out_classes,classes_to_be_shown)
            image_with_boxes = DrawDirection(buffer,IsFrame=True, FramePath=None, Frame=image_data[i][0])
            image_with_boxes=image_with_boxes.resize((416,416),Image.ANTIALIAS)
            cv2.imwrite('Video_Images/{}.jpg'.format(q),cv2.cvtColor(np.array(image_with_boxes), cv2.COLOR_BGR2RGB))
        else:
            image_with_boxes = draw_boxes(image_data[i][0], out_boxes, out_classes,class_names, out_scores) 
        imageQueue.put(np.array(image_with_boxes)); #queue image_with_boxes to imageQueue.

    return out_classes
"""
#####################################################################################
############################# display_interval FUNCTION #############################
#####################################################################################
"""
def display_interval():
    '''
    Function to be run on separate thread to display frames after adding bounding boxes continously
    '''
    global  imageQueue,ret
    
    tempImage = [];
    print("in interval")
    while (video.isOpened()):
        if (ret==False):

            break
        if (not imageQueue.empty()):
            tempImage = imageQueue.get();#dequeue frames with boxes to avoid overflow of memory
            tempImage=cv2.cvtColor(tempImage, cv2.COLOR_BGR2RGB)
            print("in if")
        print("out if")
        cv2.imshow('Frame',tempImage)               
        if cv2.waitKey(1) and 0xFF == ord('q'):break;
    cv2.destroyAllWindows()
    
"""
#####################################################################################
################################ average_classes FUNCTION ###########################
#####################################################################################
"""
def average_classes(out_classes):
    global start,count_occurrence,frames_counted,number_of_frames_averaged,min_occurrence   
    show_class=[]
    out_classes=list(map(int,out_classes))
    for z in out_classes:
        start[z]=1
        count_occurrence[z]= count_occurrence[z]+1 
    frames_counted=[frames_counted[f]+1 if t==1 else frames_counted[f] for f,t in enumerate(start)]
    frames_counted=list(map(int,frames_counted))
    show_class=[show_class for show_class,x in enumerate(count_occurrence) if x > min_occurrence]
    show_class=list(map(int,show_class))

    if (any(r==number_of_frames_averaged for r in frames_counted)):        
        indices_frames_counted=[]
        indices_frames_counted_temp=list(np.nonzero(np.array(frames_counted) >= number_of_frames_averaged))
        for i in range (len(indices_frames_counted_temp[0])):
            indices_frames_counted.append(indices_frames_counted_temp[0][i])

        for z in indices_frames_counted:
            frames_counted[z]=0 
            count_occurrence[z]=0
            start[z]=0
            frames_counted=list(map(int,frames_counted))
            count_occurrence=list(map(int,count_occurrence))
            start=list(map(int,start))
    return show_class


                         
if __name__ == '__main__':
   try:
       _main()
   except KeyboardInterrupt: #catch CTRL+C press
       print("                   ###### LIVE VIDEO RELEASED ########      ")
       video.release() #release camera resources after pressing CTRL+C
       get_ipython().magic('%reset -sf') #delete all variables after pressing CTRL+C
