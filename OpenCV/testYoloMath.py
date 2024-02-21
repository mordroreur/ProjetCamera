# import the necessary packages
# openc cv package to get and use image
import cv2
# argparsing to use arguments
import argparse
# numpy to use better matrix
import numpy as np
# ultralytics to use models
from ultralytics import YOLO



def distance_euclidienne(point1, point2):
    return (point1[0] - point2[0])**2 + (point1[1] - point2[1])**2

def associer_point_liste(point, listes):
    meilleure_liste = 0
    meilleure_distance = distance_euclidienne(point, listes[0][-1])
    
    for i in range(1, len(listes)):
        if(len(listes[i]) != 0):
            distance_last = distance_euclidienne(point, listes[i][-1])
            if distance_last < meilleure_distance:
                meilleure_distance = distance_last
                meilleure_liste = i
    
    return meilleure_liste


def findNewList(meilleure_qui, humanPosition, humanPath, humanqui):
    for i in range(humanPath):
        if(len(humanPath[i]) == 0):
            humanqui[meilleure_qui] = i
            return
    print("C'est la merde...")

def getCloser(path, positions, usedPosition):
    best_pos = -1
    best_dist = 800000000

    for i in range(len(positions)):
        if(usedPosition[i]):
            pos_dist = distance_euclidienne(positions[i], path[-1])
            # print(best_dist, pos_dist, " position ", i)
            if pos_dist < best_dist:
                best_dist = pos_dist
                best_pos = i
    return best_pos









# list the possible arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", help="path where the output video will be created")
ap.add_argument("-v", "--video", help="path to a video file")
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
humanPath = [] # chemin emprunte par chaque humain detecte
lastSeen = []
lastHumanNumber = 0

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (0, 0, 0), (255, 255, 255)]


# do until we want to stop
while(True):
    # Capture frame-by-frame the camera
    ret, frame = vid.read()

    humanPosition = []
    humanLinking = []
    usedPosition = []

    results = model(frame, verbose=False)
    # print(results[0].boxes)


    # detection du nombre d'humain
    for i in range(len(results[0].boxes.xyxy)):
        if(int(results[0].boxes.cls[i].item()) == 0):
            humanPosition.append((int(results[0].boxes.xywh[i][0].item()), int(results[0].boxes.xywh[i][1].item())))
            usedPosition.append(True)
    
    for i in range(len(lastSeen)-1, -1, -1):
        lastSeen[i] = lastSeen[i]-1
        if(lastSeen[i] == 0):
            humanPath.pop(i)
            lastSeen.pop(i)
            i = i - 1

    for i in range(len(humanPosition)):
        humanLinking.append(-1)

    # detection de la suite de chaque chemin
    for i in range(len(humanPath)):
        cl = getCloser(humanPath[i], humanPosition, usedPosition)
        if(cl != -1):
            usedPosition[cl] = False
            humanLinking[cl] = i
            humanPath[i].append(humanPosition[cl])
            lastSeen[i] = 10
    
    # creation des nouveaux chemins
    for i in range(len(humanPosition)):
        if(humanLinking[i] == -1):
            humanLinking[i] = len(humanPath)
            humanPath.append([])
            lastSeen.append(10)
            humanPath[-1].append(humanPosition[i])
    
            

    for i in range(len(humanPath)):
        for j in range(len(humanPath[i])-1):
            cv2.line(frame, humanPath[i][j], humanPath[i][j+1], colors[i], 1)   


    nbhumVue = 0
    for i in range(len(results[0].boxes.xyxy)):
        if(int(results[0].boxes.cls[i].item()) == 0):
            # print(nbhumVue, "vue linke", humanLinking[nbhumVue])
            nbhumVue = nbhumVue +1
            (x, y, w, h)  = results[0].boxes.xyxy[i]
            #print(humanLinking[i])
            cv2.rectangle(frame, (int(x.item()), int(y.item())), (int(w.item()), int(h.item())), colors[humanLinking[nbhumVue-1]], 1)

            

    # frame = results[0].plot()

    # cv2.rectangle(frame, (results[0].boxes.xywh[0], results[0].boxes.xywh[1]), (results[0].boxes.xywh[0][2]+results[0].boxes.xywh[0][0], results[0].boxes.xywh[0][3]+results[0].boxes.xywh[0][1]),
    #                      (255, 0, 0), 1)
    
    # print("next\n\n")

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


