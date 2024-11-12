from utils import (
    read_video,
    save_video
)
from trackers import PlayerTracker


def main():
    input_path = r"A:\ProgrmmingStuff\Tennis-Analysis\input_videos\input_video.mp4"
    video_frames = read_video(input_path)
    
    player_tracker = PlayerTracker(model_path=r"Tennis-Analysis\yolov8x.pt")
    player_detections = player_tracker.detect_frames(video_frames)
    
    output_frames = player_tracker.draw_bounding_boxes(video_frames, player_detections)
    
    save_video(output_frames, r"A:\ProgrmmingStuff\Tennis-Analysis\output_videos\output_video.avi")
    
if __name__ == "__main__":
    main()