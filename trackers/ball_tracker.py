from ultralytics import YOLO
import cv2 as cv
import pickle
import pandas as pd
import numpy

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        df_ball_positons = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])
        # interpolate the missing values
        df_ball_positons = df_ball_positons.interpolate()
        df_ball_positons = df_ball_positons.bfill()
        ball_positions = [{1:x} for x in df_ball_positons.to_numpy().tolist()]
        return ball_positions
    
    def get_ball_shot_frames(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        df_ball_positons = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"]) 
        
        df_ball_positons["mid_y"] = (df_ball_positons["y1"] + df_ball_positons["y2"]) / 2
        df_ball_positons["mid_y_rolling_means"] = df_ball_positons["mid_y"].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positons["delta_y"] = df_ball_positons["mid_y_rolling_means"].diff() 

        minimum_change_frames_for_hit = 25

        # Ensure the `ball_hit` column exists and initialize with zeros
        if "ball_hit" not in df_ball_positons.columns:
            df_ball_positons["ball_hit"] = 0

        for i in range(9, len(df_ball_positons) - int(minimum_change_frames_for_hit * 1.2)):
            negative_position_change = df_ball_positons["delta_y"].iloc[i] > 0 and df_ball_positons["delta_y"].iloc[i + 1] < 0
            positive_position_change = df_ball_positons["delta_y"].iloc[i] < 0 and df_ball_positons["delta_y"].iloc[i + 1] > 0

            if negative_position_change or positive_position_change:
                change_count = 0
                for change_frame in range(i + 1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
                    negative_position_change_following_frame = df_ball_positons["delta_y"].iloc[i] > 0 and df_ball_positons["delta_y"].iloc[change_frame] < 0
                    positive_position_change_following_frame = df_ball_positons["delta_y"].iloc[i] < 0 and df_ball_positons["delta_y"].iloc[change_frame] > 0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count += 1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count += 1

                if change_count > minimum_change_frames_for_hit - 1:
                    df_ball_positons.loc[i, "ball_hit"] = 1
        frame_num_with_ball_hits = df_ball_positons[df_ball_positons["ball_hit"] == 1].index.tolist()
        return frame_num_with_ball_hits

        
    def detect_frames(self, frames, read_from_stubs=False, stub_path=None):
        ball_detections = []
        
        if read_from_stubs and stub_path is not None:
            with open(stub_path, 'rb') as file:
                ball_detections = pickle.load(file)
            return ball_detections
            
        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)
        
        if stub_path is not None:
            with open(stub_path, "wb") as file:
                pickle.dump(ball_detections, file)
                
        return ball_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)
        ball_dict = {}

        for box in results[0].boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result

        return ball_dict

    def draw_bounding_boxes(self, frames, ball_detections):
        output_frames = []
        for frame, detections in zip(frames, ball_detections):
            for track_id, box in detections.items():
                x1, y1, x2, y2 = map(int, box)
                cv.putText(frame, f"Ball ID: {track_id}", (x1 - 10, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            output_frames.append(frame)
        return output_frames
