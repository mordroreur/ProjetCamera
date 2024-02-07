# import the necessary packages
# openc cv package to get and use image
import cv2
# argparsing to use arguments
import argparse
# numpy to use better matrix
import numpy as np

import time

# list the possible arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", help="path where the output video will be created")
ap.add_argument("-v", "--video", help="path to a video file")
#ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
#ap.add_argument("-t", "--tracker", type=str, default="csrt", help="OpenCV object tracker type")
args = vars(ap.parse_args())

# disable the demonstration output video
ENABLE_OUTPUT = args.get("output") is not None

# open the video stream we will work on
if(args.get("video") is None):
    # try to open a reel camera
    vid = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
    if not vid.isOpened():
        vid = cv2.VideoCapture("/dev/video0")
        if not vid.isOpened():
            raise IOError("Cannot open webcam")
else:
    # open a video
    vid = cv2.VideoCapture(args["video"])

# the output will be written to the given file
if(ENABLE_OUTPUT):
    out = cv2.VideoWriter(args["output"], cv2.VideoWriter_fourcc(*'MJPG'), 15.,(640,480))


# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


# initialisation des variables utiles
humanSquare = None











# do until we want to stop
while(True):
    # Capture frame-by-frame the camera
    ret, frame = vid.read()

    # resizing for faster detection
    frame = cv2.resize(frame, (640, 480))

    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


    # detect all potential people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(frame, winStride=(4, 4),
                                            padding=(4, 4),
                                            scale=1.5)
    
    if(humanSquare == None and len(boxes) != 0):
        humanSquare = []
        humanSquare.append(boxes[0][0])
        humanSquare.append(boxes[0][1])
        humanSquare.append(boxes[0][0]+boxes[0][2])
        humanSquare.append(boxes[0][1]+boxes[0][3])
    elif(humanSquare != None and len(boxes)!=0):
        for (xA, yA, xB, yB) in boxes:
            xB = xA+xB
            yB = yA+yB
            if(xA > humanSquare[0] and xA < humanSquare[2] and xA > humanSquare[1] and xA < humanSquare[3]):
                humanSquare = [xA, yA, xB, yB]
            elif(xB > humanSquare[0] and xB < humanSquare[2] and xB > humanSquare[1] and xB < humanSquare[3]):
                humanSquare = [xA, yA, xB, yB]
            elif(xA < humanSquare[0] and xB > humanSquare[2]):
                if((xA > humanSquare[1] and xA < humanSquare[3]) or (xA < humanSquare[1] and xB > humanSquare[3])):
                    humanSquare = [xA, yA, xB, yB]

    if(humanSquare != None):
        cv2.rectangle(frame, (humanSquare[0], humanSquare[1]), (humanSquare[2], humanSquare[3]),
                          (0, 255, 0), 2)
    

    
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    #for (xA, yA, xB, yB) in boxes:
    for i in range(len(boxes)):
        # display the detected boxes in the colour picture
        cv2.rectangle(frame, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]),
                          (255, 0, 0), 1)
        cv2.putText(frame, str(weights[i]), (boxes[i][0], boxes[i][1]), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (255, 0, 0), 1, cv2.LINE_AA) 
    


    # Write the output video
    if(ENABLE_OUTPUT):
        out.write(frame.astype('uint8'))

    # Display the resulting frame
    cv2.imshow('frame',frame)
    # time.sleep(2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release everything
vid.release()
if(ENABLE_OUTPUT):
    out.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)


"""







    

    for b in boxes:
        print(b)
    #boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        # display the detected boxes in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                          (0, 255, 0), 2)
    
    
    


"""