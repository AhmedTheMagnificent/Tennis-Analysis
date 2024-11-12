from ultralytics import YOLO

model = YOLO("yolov8x")

model.predict(r"A:\ProgrmmingStuff\Tennis-Analysis\input_videos\image.png", save=True)