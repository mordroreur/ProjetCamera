#import opencv librairy
import cv2 
import numpy as np 
import time 
import random

#vid = cv2.VideoCapture("./input_cam/video_mathieu.mp4")
#vid = cv2.VideoCapture("./input_cam/video_mathieu_2.mp4")
vid = cv2.VideoCapture("./input_cam/video_des_deux.mp4")
#vid = cv2.VideoCapture("./input_cam/video_enzo.mp4")

# load images
ret, preLast = vid.read()
ret, lastImage = vid.read() 

# ksize 
ksize = (15, 15) 
lsize = (5, 5) 

# do until we want to stop
while(True):
    # Capture frame-by-frame the camera
    ret, image = vid.read()
    if(not ret):
        break

    
    
    # Using cv2.blur() method  
    bluredim = cv2.blur(image, ksize)

    

    # compute difference
    difference = cv2.subtract(preLast, bluredim)
    #difference = cv2.add(lastImage, image)

    # color the mask red
    Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    Conv_hsv_Gray = cv2.blur(Conv_hsv_Gray, lsize)
    ret, mask = cv2.threshold(Conv_hsv_Gray, 255, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
    difference[mask != 255] = [0, 0, 255]

    preLast = lastImage.copy()
    lastImage = bluredim.copy()
    
    # add the red mask to the images to make the differences obvious
    #image[mask != 255] = [0, 0, 255]

    imgray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 40, 255, 0)
    contours,hierarchy=cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)


    boundRect = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly)
    for i in range(len(contours)):
        cv2.rectangle(image, (int(boundRect[i][0]), int(boundRect[i][1])), (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), (255,255,255), 2)
    

    cv2.imshow('frame',image)
    time.sleep(0.1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    