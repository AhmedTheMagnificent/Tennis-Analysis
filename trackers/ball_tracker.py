from ultralytics import YOLO
import cv2 as cv
import pickle

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
    def detect_frames(self, frames, read_from_studs=False, stub_path=None):
        ball_detections = []
        
        if read_from_studs and stub_path is not None:
            with open(stub_path, 'rb') as file:
                ball_detections = pickle.load(file)
                return ball_detections
            
        for frame in frames:
            player_dict = self.detect_frame(frame)
            ball_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, "wb") as file:
                pickle.dump(ball_detections, f)
        return ball_detections  # Add this line to return the detections

        
    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.15, persist=True)
        ball_dict = {}

        for bos in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result

        return ball_dict

    
    def draw_bounding_boxes(self, frames, player_detections):
        output_frames = []
        for frame, detections in zip(frames, player_detections):
            for track_id, box in detections.items():
                # Ensure box coordinates are integers
                x1, y1, x2, y2 = map(int, box)
                cv.putText(frame, f"Ball ID: {track_id}", (x1 - 10, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            output_frames.append(frame)
        return output_frames
