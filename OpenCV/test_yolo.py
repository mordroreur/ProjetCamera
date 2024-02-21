from ultralytics import YOLO

model = YOLO('yolov8s.pt')

model.info()

results = model(source=0, show=True, conf=0.4, save=True)
