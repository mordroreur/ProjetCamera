import time
from micropython import const

import uasyncio as asyncio
import aioble
import bluetooth

import struct

import sensor, image, time, os, tf, math, uos, gc

################################
#        Camera Driver         #
################################

debug = True
chariot_allume = False
video = True

# Camera (sensor) settings
sensor.reset()                         # Reset and initialize the sensor.
sensor.set_pixformat(sensor.RGB565)    # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)      # Set frame size to QVGA (320x240)
sensor.set_windowing((240, 240))       # Set 240x240 window.
sensor.skip_frames(time=2000)          # Let the camera adjust.

# Load built in model
net, labels = None, None
try: labels, net = tf.load_builtin_model('trained')
except Exception as e: raise Exception(e)

# Flux vidéo sur la caméra
stream = image.ImageIO("/video_des_deux.bin", "r")

# Read first image
# tmp = img.scale(x_scale=90)
if video:
    img = stream.read(copy_to_fb=True, loop=True, pause=True)
    # img.scale(x_size=100, y_size=100)
    tmp = img.copy()
else:
    img = sensor.snapshot()
    # img.scale(x_size=96, y_size=96)
    tmp = img.copy()

# Global variables
min_confidence = 0.55
seuil_h_min = 15
seuil_area_max = 1500
seuil_prob = 20

squareX = 16
squareY = 16
sizeX = tmp.width()/squareX
sizeY = tmp.height()/squareY

heat = 5

# Grille de probabilités
probability = []
for i in range(sizeX+1):
    probability.append([])
    for j in range(sizeY+1):
        probability[i].append(0)

visited = []
for i in range(sizeX):
    visited.append([])
    for j in range(sizeY):
        visited[i].append(0)

numberOfPeople = 0
pileRectangleEnglobe = []
oldRect = []
oldColor = []

person_max_distance = 15
person_max_area = 1000

# Processing

#clock = time.clock()  # Create a clock object to track the FPS.

def whereToGo():
    global oldColor
    global oldRect
    global sizeX
    global sizeY
    """
    for i in range(sizeX+1):
        for j in range(sizeY+1):
            probability[i][j] = 0"""


    # Get new image
    if video:
        img = stream.read(copy_to_fb=True, loop=True, pause=True)
        # img.scale(x_size=100, y_size=100)
        tmp = img.copy()
    else:
        img = sensor.snapshot()
        # img.scale(x_size=96, y_size=96)
        tmp = img.copy()

    tmp.mean(3, threshold=True, offset=5, invert=True)

    # Initialize box coordinates
    x_min = img.width() + 1
    x_max = 0
    y_min = img.height() + 1
    y_max = 0

    # AI person detection
    # 0 : background
    # 1 : person
    detected = net.detect(tmp, thresholds=[(math.ceil(min_confidence * 255), 255)])
    for i, detection_list in enumerate(detected):

        if (i == 0): continue # background class
        if (len(detection_list) == 0): continue # no detections for this class

        for d in detection_list:
            [x, y, w, h] = d.rect()
            if h < seuil_h_min: continue
            if w*h > seuil_area_max: continue

            # Compute the probability heat map
            for xt in range(int(x/squareX), int(x/squareX + w/squareX)+1):
                for yt in range(int(y/squareY), int(y/squareY + h/squareY)+1):
                    if probability[xt][yt] < seuil_prob:
                        probability[xt][yt] += heat

    # print heat map and compute min and max coordinates

    for i in range(sizeX):
        for j in range(sizeY):
            if (probability[i][j] != 0):
                probability[i][j] = probability[i][j]-2
                if(probability[i][j] < 0):
                    probability[i][j] = 0

                if i < x_min:
                    x_min = i
                if j < y_min:
                    y_min = j

                if i > x_max:
                    x_max = i
                if j > y_max:
                    y_max = j

            """

            img.draw_rectangle(i*squareX, j*squareY,
                               squareX, squareY,
                               color=(int(255*probability[i][j]/10),
                                      int(255-255*probability[i][j]/10),
                                      0),
                               thickness=1)
            """



    # img.draw_rectangle(x_min*squareX, y_min*squareY, (x_max-x_min)*squareX, (y_max-y_min)*squareY, color=(0, 255, 255))

    # Differenciate different persons

    pileRectangleEnglobe = []
    numberOfPeople = 0
    for i in range(sizeX):
        for j in range(sizeY):
            if(probability[i][j] != 0):
                probability[i][j] -=1
                if(i != 0):
                    if(visited[i-1][j] != 0):
                        visited[i][j] = visited[i-1][j]
                        who = visited[i][j]-1
                        pileRectangleEnglobe[who][2] = max(pileRectangleEnglobe[who][2],
                                                           i+1-pileRectangleEnglobe[who][0])
                if(j != 0):
                    if(visited[i][j-1] != 0 and visited[i][j] == 0):
                        visited[i][j] = visited[i][j-1]
                        who = visited[i][j]-1
                        pileRectangleEnglobe[who][3] = max(pileRectangleEnglobe[who][3],
                                                           j+1-pileRectangleEnglobe[who][1])
                if(visited[i][j] == 0):
                    numberOfPeople+=1
                    visited[i][j] = numberOfPeople
                    pileRectangleEnglobe.append([i, j, 1, 1])

    for i in range(sizeX):
        for j in range(sizeY):
            visited[i][j] = 0

    # Compute bounding boxes merge

    for rty in range(2):
        i = 0
        while(i < len(pileRectangleEnglobe)):
            ix = pileRectangleEnglobe[i][0]
            iy = pileRectangleEnglobe[i][1]
            iw = pileRectangleEnglobe[i][0]+pileRectangleEnglobe[i][2]
            ih = pileRectangleEnglobe[i][1]+pileRectangleEnglobe[i][3]
            fusioList = []
            for j in range(i+1, len(pileRectangleEnglobe)):
                jx = pileRectangleEnglobe[j][0]
                jy = pileRectangleEnglobe[j][1]
                jw = pileRectangleEnglobe[j][0]+pileRectangleEnglobe[j][2]
                jh = pileRectangleEnglobe[j][1]+pileRectangleEnglobe[j][3]
                if(ix<= jx and iw >=jx and ((jy <= iy and jh >=ih) or (jy >= iy and jh <=ih) or (jy >= iy and jy <= ih) or (jh >= iy and jh <= ih))):
                    fusioList.append(j)
                elif(ix<= jw and iw >=jw and((jy <= iy and jh >=ih) or (jy >= iy and jh <=ih) or (jy >= iy and jy <= ih) or (jh >= iy and jh <= ih))):
                    fusioList.append(j)
                elif(ix<= jx and iw >=jw and((jy <= iy and jh >=ih) or (jy >= iy and jh <=ih) or (jy >= iy and jy <= ih) or (jh >= iy and jh <= ih))):
                    fusioList.append(j)
                elif(ix>= jx and iw <=jw and((jy <= iy and jh >=ih) or (jy >= iy and jh <=ih) or (jy >= iy and jy <= ih) or (jh >= iy and jh <= ih))):
                    fusioList.append(j)
            for k in fusioList:
                old = i
                old2 = k
                newi = []
                if(pileRectangleEnglobe[old][0] < pileRectangleEnglobe[old2][0]):
                    newi.append(pileRectangleEnglobe[old][0])
                else:
                    newi.append(pileRectangleEnglobe[old2][0])

                if(pileRectangleEnglobe[old][1] < pileRectangleEnglobe[old2][1]):
                    newi.append(pileRectangleEnglobe[old][1])
                else:
                    newi.append(pileRectangleEnglobe[old2][1])

                if(pileRectangleEnglobe[old][0]+pileRectangleEnglobe[old][2] > pileRectangleEnglobe[old2][0]+pileRectangleEnglobe[old2][2]):
                    newi.append(pileRectangleEnglobe[old][0]+pileRectangleEnglobe[old][2] - newi[0])
                else:
                    newi.append(pileRectangleEnglobe[old2][0]+pileRectangleEnglobe[old2][2] - newi[0])

                if(pileRectangleEnglobe[old][1]+pileRectangleEnglobe[old][3] > pileRectangleEnglobe[old2][1]+pileRectangleEnglobe[old2][3]):
                    newi.append(pileRectangleEnglobe[old][1]+pileRectangleEnglobe[old][3] - newi[1])
                else:
                    newi.append(pileRectangleEnglobe[old2][1]+pileRectangleEnglobe[old2][3] - newi[1])

                pileRectangleEnglobe[old] = newi

            for k in range(len(fusioList), 0, -1):
                pileRectangleEnglobe.pop(fusioList[k-1])

            i+=1

    # Difference person algorithm

    if(len(oldRect) != 0):
        if(len(pileRectangleEnglobe) == 0):
            oldRect = []
        elif(len(pileRectangleEnglobe) == 1):
            x = (oldRect[0]+oldRect[2]/2)- (pileRectangleEnglobe[0][0]+pileRectangleEnglobe[0][2]/2)
            y = (oldRect[1]+oldRect[3]/2)- (pileRectangleEnglobe[0][1]+pileRectangleEnglobe[0][3]/2)
            if(x*x+y*y < 20):
                if(pileRectangleEnglobe[0][2] + pileRectangleEnglobe[0][3] - oldRect[2]+oldRect[3] > -2):
                    oldRect = pileRectangleEnglobe[0].copy()
                else:
                    oldRect[0] -= x
                    oldRect[1] -= y
        elif(len(pileRectangleEnglobe) > 1):
            x = (oldRect[0]+oldRect[2]/2)- (pileRectangleEnglobe[0][0]+pileRectangleEnglobe[0][2]/2)
            y = (oldRect[1]+oldRect[3]/2)- (pileRectangleEnglobe[0][1]+pileRectangleEnglobe[0][3]/2)
            whomin = 0
            mini = x*x+y*y
            for i in range(1, len(pileRectangleEnglobe)):
                x = (oldRect[0]+oldRect[2]/2)- (pileRectangleEnglobe[i][0]+pileRectangleEnglobe[i][2]/2)
                y = (oldRect[1]+oldRect[3]/2)- (pileRectangleEnglobe[i][1]+pileRectangleEnglobe[i][3]/2)
                actu = x*x+y*y
                if(abs(actu - mini) < person_max_distance):
                    cc = img.get_statistics(roi=(pileRectangleEnglobe[whomin][0]*squareX,pileRectangleEnglobe[whomin][1]*squareY,pileRectangleEnglobe[whomin][2]*squareX,pileRectangleEnglobe[whomin][3]*squareY))
                    colC1 = [cc[0], cc[8], cc[16]] # img statistics - 0: l_mean, 8: a_mean, 16: b_mean
                    cc = img.get_statistics(roi=(pileRectangleEnglobe[i][0]*squareX,pileRectangleEnglobe[i][1]*squareY,pileRectangleEnglobe[i][2]*squareX,pileRectangleEnglobe[i][3]*squareY))
                    colC2 = [cc[0], cc[8], cc[16]] # img statistics - 0: l_mean, 8: a_mean, 16: b_mean
                    if((oldColor[0]-colC1[0])*(oldColor[0]-colC1[0])+(oldColor[1]-colC1[1])*(oldColor[1]-colC1[1])+(oldColor[2]-colC1[2])*(oldColor[2]-colC1[2]) > (oldColor[0]-colC2[0])*(oldColor[0]-colC2[0])+(oldColor[1]-colC2[1])*(oldColor[1]-colC2[1])+(oldColor[2]-colC2[2])*(oldColor[2]-colC2[2])):
                        whomin = i
                        mini = x*x+y*y
                else:
                    if(actu < mini):
                        whomin = i
                        mini = actu
            x = (oldRect[0]+oldRect[2]/2)- (pileRectangleEnglobe[whomin][0]+pileRectangleEnglobe[whomin][2]/2)
            y = (oldRect[1]+oldRect[3]/2)- (pileRectangleEnglobe[whomin][1]+pileRectangleEnglobe[whomin][3]/2)
            if(x*x+y*y < person_max_area):
                if(pileRectangleEnglobe[whomin][2] + pileRectangleEnglobe[whomin][3] - oldRect[2]+oldRect[3] > -2):
                    oldRect = pileRectangleEnglobe[whomin].copy()
                else:
                    oldRect[0] -= x
                    oldRect[1] -= y
            col = img.get_statistics(roi=(int(oldRect[0]*squareX), int(oldRect[1]*squareY), int(oldRect[2]*squareX), int(oldRect[3]*squareY)))
            if((oldColor[0]-col[0])*(oldColor[0]-col[0])+(oldColor[1]-col[8])*(oldColor[1]-col[8])+(oldColor[2]-col[16])*(oldColor[2]-col[16]) < 500):
                oldColor = [col[0], col[8], col[16]] # img statistics - 0: l_mean, 8: a_mean, 16: b_mean
    else: # If there is no person detected
        if(len(pileRectangleEnglobe) > 1):
            oldRect = [x_min, y_min, x_max-x_min, y_max-y_min]
            col = img.get_statistics(roi=(int(oldRect[0]*squareX), int(oldRect[1]*squareY), int(oldRect[2]*squareX), int(oldRect[3]*squareY)))
            oldColor = [col[0], col[8], col[16]] # img statistics - 0: l_mean, 8: a_mean, 16: b_mean
        elif(len(pileRectangleEnglobe) == 1):
            oldRect = pileRectangleEnglobe[0].copy()
            col = img.get_statistics(roi=(int(oldRect[0]*squareX), int(oldRect[1]*squareY), int(oldRect[2]*squareX), int(oldRect[3]*squareY)))
            oldColor = [col[0], col[8], col[16]] # img statistics - 0: l_mean, 8: a_mean, 16: b_mean

    # Debug: Draw bounding boxes of person detected

    if(debug):
        for rec in pileRectangleEnglobe:
            if(len(rec) == 4):
                img.draw_rectangle(rec[0]*squareX, rec[1]*squareY, rec[2]*squareX, rec[3]*squareY,
                                   color=(255, 255, 0), thickness=2)
        if(len(oldRect) != 0):
            img.draw_rectangle(int(oldRect[0]*squareX), int(oldRect[1]*squareY), int(oldRect[2]*squareX), int(oldRect[3]*squareY),
                                color=(0, 255, 0), thickness=1)

    # Compute movements directions

    if(len(oldRect) != 0):
        yoffset = 0
        if(oldRect[1]+(oldRect[3]/2) < 3):
            yoffset = 0.5
        x_center = oldRect[0]+(oldRect[2]/2)
        where = ((2*x_center)/(sizeX))-1
        if(abs(where)+yoffset > 1):
            yoffset = 1-abs(where)
        if(where < 0):
            return (-where+yoffset, yoffset)
        else:
            return (yoffset, where+yoffset)
        """
        if(oldRect[0]+(oldRect[2]/2) < sizeX/3):
            return (0, 0.5)
        elif(oldRect[0]+(oldRect[2]/2) > (sizeX/3) * 2):
            return (0.5, 0)
        else:
            return (0, 0)"""
    else:
        return (0.2, 0.2)

################################
#       Bluetooth Driver       #
################################

################################
#        Global variables      #
################################

ADDR_PUBLIC = const(0x00)
ADDR_RANDOM = const(0x01)

# Définition du device du chariot
trolly_addr = "eA:31:6a:36:5c:9a"
trolly_addr_type = ADDR_RANDOM # ADDR_RANDOM
device = aioble.device.Device(trolly_addr_type, trolly_addr)

# Définition des uuids du chariot
_TROLLY_MOVING_UUID = bluetooth.UUID(0x1400)
_TROLLY_MOVING_LEFT_UUID = bluetooth.UUID(0x1401)
_TROLLY_MOVING_RIGHT_UUID = bluetooth.UUID(0x1402)
_TROLLY_MOVING_FORWARD_UUID = bluetooth.UUID(0x1403)

movement_seuil = 0.3

################################
#           Functions          #
################################

async def scan():
    while True:
        async with aioble.scan(1000) as scanner:
            async for result in scanner:
                print(result.name(), result.device)

async def connect_device():
    try:
        print("Connecting to", device)
        connection = await device.connect()
    except asyncio.TimeoutError:
        print("Timeout during connection")
        return

    print("Successfully connected to", device)

    async with connection:

        # Get Moving Service and Characteristics
        moving_service = await connection.service(_TROLLY_MOVING_UUID)
        moving_left_char = await moving_service.characteristic(_TROLLY_MOVING_LEFT_UUID)
        moving_right_char = await moving_service.characteristic(_TROLLY_MOVING_RIGHT_UUID)
        moving_forward_char = await moving_service.characteristic(_TROLLY_MOVING_FORWARD_UUID)

        # Connection loop
        while True:
            l, r = whereToGo()

            if l > movement_seuil and r > movement_seuil :
                await moving_forward_char.write(b'\x01', response=0)
            elif l > movement_seuil and r < movement_seuil:
                await moving_left_char.write(b'\x01', response=0)
            elif r > movement_seuil and l < movement_seuil:
                await moving_right_char.write(b'\x01', response=0)
            else:
                await moving_forward_char.write(b'\x00', response=0)

            await asyncio.sleep_ms(10)

################################
#         Main Function        #
################################

print("")
print("-"*50)
print("Ready to track human")
print("-"*50)

if chariot_allume:
    asyncio.run(connect_device())
else :
    while True:
        print(whereToGo())
