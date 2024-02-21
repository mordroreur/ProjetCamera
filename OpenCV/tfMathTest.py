# import sensor, image, time, os, tf, pyb
import cv2
import tensorflow as tf

# redLED = pyb.LED(1) # built-in red LED
# greenLED = pyb.LED(2) # built-in green LED

# sensor.reset()                         # Reset and initialize the sensor.
# sensor.set_pixformat(sensor.RGB565)    # Set pixel format to RGB565 (or GRAYSCALE)
# sensor.set_framesize(sensor.QVGA)      # Set frame size to QVGA (320x240)
# sensor.set_vflip(True)
# sensor.set_hmirror(True)
# sensor.set_windowing((240, 240))       # Set 240x240 window.
# sensor.skip_frames(time=2000)          # Let the camera adjust.

labels, net = tf.load_model('obj_detect1.tflite')

vid = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
if not vid.isOpened():
    vid = cv2.VideoCapture("/dev/video0")
    if not vid.isOpened():
        raise IOError("Cannot open webcam")

# do until we want to stop
while(True):
    # Capture frame-by-frame the camera
    ret, frame = vid.read()
    for obj in tf.classify(net, frame, min_scale=1.0, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5):
        cv2.rectangle(frame, (obj.rect().x, obj.rect().y), (obj.rect().w + obj.rect().x, obj.rect().h + obj.rect().y), (255, 255, 255), 1)

    cv2.imshow('frame',frame)
    # time.sleep(2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


"""
def flashLED(led): # Indicate with LED when target is detected
    found = True
    led.on()
    pyb.delay(3000)
    led.off()
    found = False


clock = time.clock()

while not found:
    clock.tick()
    img = sensor.snapshot()
    for obj in tf.classify(net, img, min_scale=1.0, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5):
        print("**********nPredictions at [x=%d,y=%d,w=%d,h=%d]" % obj.rect())
        img.draw_rectangle(obj.rect())
        predictions_list = list(zip(labels, obj.output()))
        for i in range(len(predictions_list)):
            confidence = predictions_list[i][1]
            label = predictions_list[i][0]
            print("%s = %f" % (label, confidence))
            if confidence > 0.8:
                if label == "rock":
                    print("It's a ROCK-4SE")
                    flashLED(greenLED)
                if label == "rock-5":
                    print("It's a ROCK-5B")
                    flashLED(redLED)

    print(clock.fps(), "fps")
"""