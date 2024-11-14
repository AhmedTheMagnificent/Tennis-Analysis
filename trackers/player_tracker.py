from ultralytics import YOLO
import cv2 as cv
import pickle

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
    def detect_frames(self, frames, read_from_stubs=False, stub_path=None):
        player_detections = []
        
        if read_from_stubs and stub_path is not None:
            with open(stub_path, 'rb') as file:
                player_detections = pickle.load(file)
                return player_detections
            
        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, "wb") as file:
                pickle.dump(player_detections, file)
        return player_detections  # Add this line to return the detections

        
    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)
        player_dict = {}

        for result in results:  # results is a list of Detection objects
            id_name_dict = result.names  # Accessing the names from each result
            for box in result.boxes:
                track_id = int(box.id.tolist()[0])
                result = box.xyxy.tolist()[0]
                object_cls_id = box.cls.tolist()[0]
                object_cls_name = id_name_dict[object_cls_id]
                if object_cls_name == "person":
                    player_dict[track_id] = result

        return player_dict

    
    def draw_bounding_boxes(self, frames, player_detections):
        output_frames = []
        for frame, detections in zip(frames, player_detections):
            for track_id, box in detections.items():
                # Ensure box coordinates are integers
                x1, y1, x2, y2 = map(int, box)
                cv.putText(frame, f"Player ID: {track_id}", (x1 - 10, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            output_frames.append(frame)
        return output_frames
