import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
import argparse
import pickle
import sys
import time
import math
from espeak import espeak
from utils import label_map_util
from utils import visualization_utils as vis_util




def objPosition(xmin,xmax,ymin,ymax):
    if xmin >0 and xmin <=80 and xmax <= 640 and ymin > 0 and ymin <= 80 and ymax <= 640:
        espeak.synth("object is close to you")
        print("object is close to you")
    elif xmin >0 and xmin <=320 and xmax <= 320 and ymin >320 and ymax <= 640:
        espeak.synth("object is at down left corner")
        print("object is at down left corner")
    elif xmin >320 and xmax <= 640 and ymin >320 and ymax <= 640:
        espeak.synth("object is at down right corner")
        print("object is at down right corner")
    elif xmin >=160 and xmin <=480 and xmax >= 160 and xmax <= 480 and ymin >=160 and ymin <= 480 and ymax >=160 and ymax <= 480:
        espeak.synth("object is at centre")
        print("object is at centre")
    elif xmin >=80 and xmin <=560 and xmax >= 80 and xmax <= 560 and ymin >=80 and ymin <= 560 and ymax >=80 and ymax <= 560:
        espeak.synth("object is detected")
        print("object is detected")
    elif xmin >0 and xmin <=320 and xmax <= 320 and ymin >0 and ymin<=320 and ymax <= 320:
        espeak.synth("object is at top left corner")
        print("object is at top left corner")
    elif xmin >320 and xmax <= 640 and ymin >0 and ymin <=320 and ymax <= 320:
        espeak.synth("object is at top right corner")
        print("object is at top right corner")
    elif xmin >0 and xmin <= 320 and xmax <= 320 and ymin >0 and ymax <= 640:
        espeak.synth("object is on your left")
        print("object is on your left")
    elif xmin >320 and xmax <= 640 and ymin >0 and ymax <= 640:
        espeak.synth("object is on your right")
        print("object is on your right")
    elif xmin >0 and xmax <=640 and ymin >0 and ymin <=320 and ymax <=320:
        espeak.synth("object is at top")
        print("object is at top")
    elif xmin >0 and xmax <=640 and ymin >320 and ymax <=640:
        espeak.synth("object is at bottom")
        print("object is at bottom")
    else :
        espeak.synth("no object is in front of you")
        print("no object is in front of you")
    #espeak.synth(objectDetectText) 
    print(xmin)
    print(xmax)
    print(ymin)
    print(ymax)
    return



  
def objApproach(xmin,xmax,ymin,ymax):
    length = xmax-xmin
    breadth = ymax-ymin
    f= (length * length) + (breadth * breadth)
    diaglength = int(math.sqrt(f))
    approach_list.append(diaglength)
    global approach_count
    print("approach count")
    print(approach_count)
#    print(approach_list)
    if (approach_count%4)==0:
        for x in range (3):
#            print("x")
#            print(x)
#            print(approach_list[x])
#            print("x+1")
#            print(approach_list[x+1])
            if approach_list[x]<approach_list[x+1]:
                global count
                count+=1
                if (count%4)==0:
                    espeak.synth("object detected is approaching towards you")
                    print("object detected is approaching towards you")
            elif approach_list[x]>approach_list[x+1]:
                global count1
                count1+=1
                if (count1%4)==0:
                    espeak.synth("object detected is moving away from you")
                    print("object detected is moving away from you")
            elif approach_list[x]==approach_list[x+1]:
                global count2
                count2+=1
                if (count2%4)==0:
                    espeak.synth("object detected is at the same place")
                    print("object is detected at the same place")
            else:
                print("xyz")
        del approach_list[0:4]
#    print(diaglength)
    approach_count+=1
    return




def faceApproach(x,y,w,h,name):
    length = w-x
    breadth = h-y
    f= (length * length) + (breadth * breadth)
    diaglength = int(math.sqrt(f))
    approach_list1.append(diaglength)
    global approach_count1
    print("face approach count")
    print(approach_count1)
#    print(approach_list)
    if (approach_count1%4)==0:
        for m in range (3):
#            print("x")
#            print(x)
#            print(approach_list[x])
#            print("x+1")
#            print(approach_list[x+1])
            if approach_list1[m]<approach_list1[m+1]:
                global count3
                count3+=1
                if (count3%4)==0:
                    espeak.synth(name+" detected is approaching towards you")
                    print(name+" detected is approaching towards you")
            elif approach_list1[m]>approach_list1[m+1]:
                global count4
                count4+=1
                if (count4%4)==0:
                    espeak.synth(name+" detected is moving away from you")
                    print(name+" detected is moving away from you")
            elif approach_list1[m]==approach_list1[m+1]:
                global count5
                count5+=1
                if (count5%4)==0:
                    espeak.synth(name+" detected is at the same place")
                    print(name+" detected is at the same place")
            else:
                print("xyz")
        del approach_list1[0:4]
#    print(diaglength)
    approach_count1+=1
    return








    



#importing classifier and other trainning models
espeak.synth("Starting object detection")
face_cascade = cv2.CascadeClassifier('/usr/local/lib/python3.5/dist-packages/cv2/data/lbpcascade_frontalface_improved.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("/home/pi/tensorflow1/models/research/object_detection/trainner.yml")
language = 'en'
labels = {"person_name": 1}
count=1
count1=1
count2=1
count3=1
count4=1
count5=1
approach_count=1
approach_count1=1
global approach_list
approach_list=[]
global approach_list1
approach_list1=[]

with open("labels.pickle", 'rb') as f:
    #f: means files
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}
    
#IM_WIDTH = 1280
#IM_HEIGHT = 720
IM_WIDTH = 640    #Use smaller resolution for
IM_HEIGHT = 640   #slightly faster framerate
#IM_WIDTH = 320
#IM_HEIGHT = 240

camera_type = 'picamera'


sys.path.append('..')


MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'

CWD_PATH = os.getcwd()


PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

NUM_CLASSES = 90


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)



image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

num_detections = detection_graph.get_tensor_by_name('num_detections:0')
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX
#finishing of all the imports and training models




#start of main function
if camera_type == 'picamera':
    camera = PiCamera()
    camera.resolution = (IM_WIDTH,IM_HEIGHT)
    camera.framerate = 10
    rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
    rawCapture.truncate(0)

    for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
        t1 = cv2.getTickCount()
        frame = np.copy(frame1.array)
        frame.setflags(write=1)
        frame_expanded = np.expand_dims(frame, axis=0)
        #objectDetection
        objDetect(frame)
        #faceDetection
        faceDetect(frame)
        print(frame_rate_calc)
        cv2.imshow('Face and Object detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

        rawCapture.truncate(0)

    camera.close()

    camera.release()

cv2.destroyAllWindows()
#finish of main function



