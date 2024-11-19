import cv2 as cv
import sys
import numpy as np
sys.path.append("../")
import constants
from utils import (
    convert_meters_to_pixel_distance,
    convert_pixel_distance_to_meters,
    get_foot_position,
    get_closest_keypoint_index,
    get_height_of_box,
    measure_xy_distance,
    get_center_of_box,
    measure_distance
)


class MiniCourt():
    def __init__(self, frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.buffer = 50
        self.padding_court = 20
        self.set_canvas_background_box_position(frame)
        self.set_mini_court_positon()
        self.set_court_drawing_keypoints()
        self.set_court_lines()
        
    
    def set_mini_court_positon(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x
        
    def convert_meters_to_pixels(self, meters):
        return convert_meters_to_pixel_distance(meters, 
                                                constants.DOUBLE_LINE_WIDTH,
                                                self.court_drawing_width)
        
    def set_court_drawing_keypoints(self):
        drawing_key_points = [0] * 28
        
        # point 0
        drawing_key_points[0], drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        # point 1
        drawing_key_points[2], drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
        # point 2
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = self.court_start_y + self.convert_meters_to_pixels(constants.HALF_COURT_LINE_HEIGHT * 2)
        # point 3
        drawing_key_points[6] = drawing_key_points[0] + self.court_drawing_width
        drawing_key_points[7] = drawing_key_points[5] 
        # point 4
        drawing_key_points[8] = drawing_key_points[0] +  self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[9] = drawing_key_points[1] 
        # point 5
        drawing_key_points[10] = drawing_key_points[4] + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[11] = drawing_key_points[5] 
        # point 6
        drawing_key_points[12] = drawing_key_points[2] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[13] = drawing_key_points[3] 
        # point 7
        drawing_key_points[14] = drawing_key_points[6] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[15] = drawing_key_points[7] 
        # point 8
        drawing_key_points[16] = drawing_key_points[8] 
        drawing_key_points[17] = drawing_key_points[9] + self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # point 9
        drawing_key_points[18] = drawing_key_points[16] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[19] = drawing_key_points[17] 
        # point 10
        drawing_key_points[20] = drawing_key_points[10] 
        drawing_key_points[21] = drawing_key_points[11] - self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # point 11
        drawing_key_points[22] = drawing_key_points[20] +  self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[23] = drawing_key_points[21] 
        # point 12
        drawing_key_points[24] = int((drawing_key_points[16] + drawing_key_points[18])/2)
        drawing_key_points[25] = drawing_key_points[17] 
        # point 13
        drawing_key_points[26] = int((drawing_key_points[20] + drawing_key_points[22])/2)
        drawing_key_points[27] = drawing_key_points[21] 

        self.drawing_key_points=drawing_key_points

    def set_court_lines(self):
        self.lines = [
            (0, 2),
            (4, 5),
            (6, 7),
            (1, 3),
            (0, 1),
            (8, 9),
            (10, 11),
            (10, 11),
            (2, 3)
        ]
        
    def set_canvas_background_box_position(self, frame):
        frame = frame.copy()
        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height
        
    def draw_court(self, frame):
        for i in range(0, len(self.drawing_key_points), 2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i + 1])
            cv.circle(frame, (x, y), 5, (0, 100, 255), -1)
            
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0] * 2]), int(self.drawing_key_points[line[0] * 2 + 1]))
            end_point = (int(self.drawing_key_points[line[1] * 2]), int(self.drawing_key_points[line[1] * 2 + 1]))
            cv.line(frame, start_point, end_point, (0, 0, 0), 2)
            
        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2))
        cv.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)
        
        
        return frame
        
    def draw_background_rectange(self, frame):
        shapes = np.zeros_like(frame, np.uint8)
        # Draw the rectangle
        cv.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv.FILLED)
        out = frame.copy()
        alpha = 0.5
        mask = shapes.astype(bool)
        out[mask] = cv.addWeighted(frame, alpha, shapes, 1-alpha, 0)[mask]
        return out
    
    def draw_mini_court(self, frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectange(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
            
        return output_frames
    
    def get_start_point_of_mini_court(self):
        return (self.court_start_x, self.court_start_y)
    
    def get_width_of_mini_court(self):
        return self.court_drawing_width
    
    def get_court_drawing_keypoints(self):
        return self.drawing_key_points
    
    def get_mini_court_coordinates(self, object_position, closest_keypoint, closest_keypoint_index, player_height_in_pixels, player_height_in_meters):
        distance_from_keypoint_x_pixels, distance_from_keypoint_y_pixels = measure_xy_distance(object_position, closest_keypoint)
        
        distance_from_keypoint_x_pixels = convert_pixel_distance_to_meters(distance_from_keypoint_x_pixels,
                                                                            player_height_in_meters,
                                                                            player_height_in_pixels)
    
        distance_from_keypoint_y_pixels = convert_pixel_distance_to_meters(distance_from_keypoint_y_pixels,
                                                                            player_height_in_meters,
                                                                            player_height_in_pixels)
        
        mini_court_x_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_x_pixels)
        mini_court_y_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_y_pixels)
        closest_mini_court_keypoint = (
            self.drawing_key_points[closest_keypoint_index * 2],
            self.drawing_key_points[closest_keypoint_index * 2] + 1
        )
                
        mini_court_player_position = (closest_mini_court_keypoint[0] + mini_court_x_distance_pixels,
                                    closest_mini_court_keypoint[1] + mini_court_y_distance_pixels)
        
        return mini_court_player_position
    
    def convert_bounding_boxes_to_mini_court_coordinates(self, player_boxes, ball_boxes, original_court_keypoints):
        player_heights = {
            1: constants.PLAYER_1_HEIGHT_METERS,
            2: constants.PLAYER_2_HEIGHT_METERS
        }
        
        output_player_boxes = []
        output_ball_boxes = []
        
        for frame_num, player_box in enumerate(player_boxes):
            ball_box = ball_boxes[frame_num]
            ball_position = get_center_of_box(ball_box)
            closest_player_id_to_ball = min()
            
            output_player_boxes_dict = {}
            output_ball_boxes_dict = {}
            for player_id, box in player_box.items():
                foot_position = get_foot_position(box)
                closest_keypoint_index = get_closest_keypoint_index(foot_position, original_court_keypoints, [0, 2, 12, 13])
                closest_keypoint = (original_court_keypoints[closest_keypoint_index * 2],
                                    original_court_keypoints[closest_keypoint_index * 2 + 1])
                
                frame_index_min = max(0, frame_num - 20)
                frame_index_max = min(len(player_boxes), frame_num + 50)
                boxes_heights_in_pixels = [get_height_of_box(player_box[i]) for i in range(frame_index_min, frame_index_max)]
                max_player_height_in_pixels = max(boxes_heights_in_pixels)
                mini_court_player_position = self.get_mini_court_coordinates(
                    foot_position,
                    closest_keypoint,
                    closest_keypoint_index,
                    max_player_height_in_pixels,
                    player_heights[player_id]
                )
                
                output_player_boxes_dict[player_id] = mini_court_player_position
                if closest_player_id_to_ball == player_id:
                    closest_keypoint_index = get_closest_keypoint_index(ball_position, original_court_keypoints, [0, 2, 12, 13])
                    closest_keypoint = (original_court_keypoints[closest_keypoint_index * 2],
                                    original_court_keypoints[closest_keypoint_index * 2 + 1])
                    mini_court_player_position = self.get_mini_court_coordinates(
                                                                                ball_position,
                                                                                closest_keypoint,
                                                                                closest_keypoint_index,
                                                                                max_player_height_in_pixels,
                                                                                player_heights[player_id]
                                                                                )
                    output_ball_boxes.append({1: mini_court_player_position})
            output_player_boxes.append(output_player_boxes_dict)
        return output_player_boxes, output_ball_boxes
                
                