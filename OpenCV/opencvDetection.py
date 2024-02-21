#import opencv librairy
import cv2 
import numpy as np 
import time 
import random

#vid = cv2.VideoCapture("./input_cam/video_mathieu.mp4")
vid = cv2.VideoCapture("./input_cam/video_mathieu_2.mp4")
#vid = cv2.VideoCapture("./input_cam/video_des_deux.mp4")
#vid = cv2.VideoCapture("./input_cam/video_enzo.mp4")


lastbox = [-1, -1, -1, -1]


# do until we want to stop
while(True):
    # Capture frame-by-frame the camera
    ret, frame = vid.read()
    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #img_dilate = cv2.dilate(imgray, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1), (1, 1)))
    #ret, thresh = cv2.threshold(imgray, 40, 255, 0)

    thresh = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,8)

    contours,hierarchy=cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    #im2, contours,hierarchy=cv2.findContours(frame,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

    #frame[:] = (0,0,0)

    """
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    for i in range(len(contours)):
        #if(len(contours[i]) > 20):
            contour = contours[i]
            #print("contour = ", len(contour))
            random.seed(i)
            color = (255*random.random(),255*random.random(),255*random.random())
            cv2.drawContours(frame,[contour], -1, color, 3)
    """

    #cv2.drawContours(frame, contours, -1, (255,0,0), 1)
    
    boundRect = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly)

    max = -1
    idmax = -1
    for i in range(len(boundRect)):
        val = int(boundRect[i][2])*int(boundRect[i][3])
        if(boundRect[i][2] < boundRect[i][3] and val > max and (boundRect[i][0] != 0 or boundRect[i][1] != 0 or boundRect[i][2] != frame.shape[1] or boundRect[i][3] != frame.shape[0])):
            #print(oldval, boundRect[i][3]/boundRect[i][2])
            max = val
            idmax = i
    #print(max)
    cv2.rectangle(frame, (int(boundRect[idmax][0]), int(boundRect[idmax][1])), (int(boundRect[idmax][0]+boundRect[idmax][2]), int(boundRect[idmax][1]+boundRect[idmax][3])), (255,255,255), 3)
    #print("final = ", idmax, max, " val = ", boundRect[idmax][0], boundRect[idmax][1], boundRect[idmax][2],frame.shape[1], boundRect[idmax][3],frame.shape[0])
    
    #for i in range(len(contours)):
        #cv2.rectangle(frame, (int(boundRect[i][0]), int(boundRect[i][1])), (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), (255,255,255), 2)
    


    # Display the resulting frame
    cv2.imshow('frame',frame)
    #time.sleep(0.1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


"""




"""