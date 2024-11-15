from ultralytics import YOLO
import cv2 as cv
import pickle
import sys
sys.path.append("../")
from utils import measure_distance, get_center_of_box

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def choose_and_filter_players(self, court_keypoints, player_detections):
        player_detections_first_frame = player_detections[0]
        chosen_players = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: box for track_id, box in player_dict.items() if track_id in chosen_players}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections
        
    def choose_players(self, court_keypoints, player_detection_first_frame):
        distances = []
        for track_id, box in player_detection_first_frame.items():
            player_center = get_center_of_box(box)
            min_distance = float("inf")
            for i in range(0, len(court_keypoints), 2):
                court_point = (court_keypoints[i], court_keypoints[i + 1])
                distance = measure_distance(player_center, court_point)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))
            
        # sort the distances by ascending order
        distances.sort(key=lambda x: x[1])
        # Choose the first 2 tracks
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players
    
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
