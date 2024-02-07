#import opencv librairy
import cv2
import numpy as np

from transformers import AutoImageProcessor, AutoModel
import torch

class SimpleDatasetLoader:
  def __init__(self, preprocessors=None):
    # store the image preprocessor
    self.preprocessors = preprocessors
    # if the preprocessors are None, initialize them as an
    # empty list
    if self.preprocessors is None:
      self.preprocessors = []

  def load_path(self, imagePaths, verbose=-1):
    # initialize the list of features and labels
    data = []
    labels = []

    # loop over the input images
    for (i, imagePath) in enumerate(imagePaths):
      # load the image and extract the class label assuming
      # that our path has the following format:
      # /path/to/dataset/{class}/{image}.jpg
      image = cv2.imread(imagePath)
      label = imagePath.split(os.path.sep)[-1][0:2]

      # check to see if our preprocessors are not None
      if self.preprocessors is not None:
        # loop over the preprocessors and apply each to
        # the image
        for p in self.preprocessors:
          image = p.preprocess(image)
      # treat our processed image as a "feature vector"
      # by updating the data list followed by the labels
      data.append(image)
      labels.append(label)
      # show an update every `verbose` images
      if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
        print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))

    # return a tuple of the data and labels
    return (np.array(data), np.array(labels))

class CameraDetection():
    def __init__(self, source = 0, path = "/dev/video0"):
        # define a video capture object 
        self.vid = cv2.VideoCapture(source, cv2.CAP_DSHOW) 
        if not self.vid.isOpened():
            self.vid = cv2.VideoCapture(path)
            if not self.vid.isOpened():
                raise IOError("Cannot open webcam")
        print("Pour sortir de l'application appuyez sur 'Q'.")

        self.ret, self.frame = self.vid.read()
        
        # initialize the HOG descriptor/person detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def read(self):
        # Capture the video frame 
        # by frame 
        self.ret, self.frame = self.vid.read()

    def resize(self, width = 640, height = 480):
        # resize the frame:
        self.frame = cv2.resize(self.frame, (width, height))

    def compute_hog(self):
        # detect people in the image
        # returns the bounding boxes for the detected objects
        self.frame = cv2.resize(self.frame, (320, 240))
        frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)
        # _, frame_th = cv2.threshold(frame_gray, 155, 255, cv2.THRESH_BINARY)
        
        boxes, _ = self.hog.detectMultiScale(frame_gray, winStride=(8,8))

        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

        for (xA, yA, xB, yB) in boxes:
            # display the detected boxes in the colour picture
            cv2.rectangle(self.frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

    def dinov2(self):
        image_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        model = AutoModel.from_pretrained('facebook/dinov2-base')

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device: ", device)
        model = model.to(device)

        data = load_dataset("images_test")
        
        image = data[image_id]
        
    def rgb2gray(self):
        # turn to greyscale:
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)

    def threshold(self, level = 80):
        # apply threshold. all pixels with a level larger than 80 are shown in white. the others are shown in black:
        self.ret, self.frame = cv2.threshold(self.frame, level, 255, cv2.THRESH_BINARY)
    
    def show(self, name="frame"):
        # Display the resulting frame 
        cv2.imshow(name, self.frame)

    def __del__(self):
        # After the loop release the cap object 
        self.vid.release()
        # Destroy all the windows 
        cv2.destroyAllWindows()

cam = CameraDetection(path="/dev/video0")

while(True): 

    # Capture the video frame by frame
    cam.read()
    cam.compute_hog()

    # Display the resulting frame 
    cam.show()
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
del cam
