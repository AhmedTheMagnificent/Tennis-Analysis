from ultralytics import YOLO

model = YOLO("yolov8x")

result = model.predict(r"A:\ProgrmmingStuff\Tennis-Analysis\input_videos\input_video.mp4", save=True)

print(result)
print("Boxes: ")
for box in result[0].boxes:
    print(box)