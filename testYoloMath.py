# import the necessary packages
# openc cv package to get and use image
import cv2
# argparsing to use arguments
import argparse
# numpy to use better matrix
import numpy as np
# ultralytics to use models
from ultralytics import YOLO


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

# initializing the model
model = YOLO('yolov8n.pt')
model.info()


# initialisation des variables utiles
humanSquare = None




# do until we want to stop
while(True):
    # Capture frame-by-frame the camera
    ret, frame = vid.read()



    results = model(frame, verbose=False)
    print(results[0].boxes)

    for lb in results[0].boxes.xyxy:
        (x, y, w, h)  = lb
        print(x.item())
        cv2.rectangle(frame, (int(x.item()), int(y.item())), (int(w.item()), int(h.item())), (255, 0, 0), 1)

    frame = results[0].plot()

    # cv2.rectangle(frame, (results[0].boxes.xywh[0], results[0].boxes.xywh[1]), (results[0].boxes.xywh[0][2]+results[0].boxes.xywh[0][0], results[0].boxes.xywh[0][3]+results[0].boxes.xywh[0][1]),
    #                      (255, 0, 0), 1)
    
    print("next\n\n")

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


