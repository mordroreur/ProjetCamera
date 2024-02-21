#import opencv librairy
import cv2
import numpy as np
  
# define a video capture object 
vid = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

if not vid.isOpened():
    vid = cv2.VideoCapture("/dev/video0")
    if not vid.isOpened():
        raise IOError("Cannot open webcam")
    
print("Pour sortir de l'application appuyez sur 'Q'.")

while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 

    
    # Display the resulting frame 
    cv2.imshow('frame', frame) 
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release()
# Destroy all the windows 
cv2.destroyAllWindows() 