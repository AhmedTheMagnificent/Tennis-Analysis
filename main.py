from utils import (
    read_video,
    save_video
)
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
import cv2 as cv

def main():
    input_path = r"A:\ProgrmmingStuff\Tennis-Analysis\input_videos\input_video.mp4"
    video_frames = read_video(input_path)
    
    player_tracker = PlayerTracker(model_path=r"A:\ProgrmmingStuff\Tennis-Analysis\yolov8x.pt")
    ball_tracker = BallTracker(model_path=r"A:\ProgrmmingStuff\Tennis-Analysis\models\last.pt")
    
    player_detections = player_tracker.detect_frames(video_frames, read_from_stubs=True, stub_path=r"A:\ProgrmmingStuff\Tennis-Analysis\tracker_stubs\player_detections.pkl")
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stubs=True, stub_path=r"A:\ProgrmmingStuff\Tennis-Analysis\tracker_stubs\ball_detections.pkl")
    
    court_model_path = r"A:\ProgrmmingStuff\Tennis-Analysis\models\keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])
    
    output_frames = player_tracker.draw_bounding_boxes(video_frames, player_detections)
    output_frames = ball_tracker.draw_bounding_boxes(output_frames, ball_detections)
    
    print(court_keypoints)
    output_frames = court_line_detector.draw_keypoints_on_video(output_frames, court_keypoints)
    
    save_video(output_frames, r"A:\ProgrmmingStuff\Tennis-Analysis\output_videos\output_video.avi")
    
if __name__ == "__main__":
    main()