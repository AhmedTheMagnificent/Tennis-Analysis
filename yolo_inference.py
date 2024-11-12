from ultralytics import YOLO

model = YOLO("yolov8x.pt")

result = model.track(r"A:\ProgrmmingStuff\Tennis-Analysis\input_videos\input_video.mp4", conf=0.2, save=True)
