def get_center_of_box(box):
    x1, y1, x2, y2 = box
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)

def measure_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def get_foot_position(box):
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), y2)

def get_closest_keypoint_index(point, keypoints, keypoint_indices):
    closest_distance = float("inf")
    keypoint_ind = keypoint_indices[0]
    for keypoint_idx in keypoint_indices:
        keypoint = keypoints[keypoint_idx * 2], keypoints[keypoint_idx * 2 + 1]
        distance = abs(point[1] - keypoint[1])
        if distance < closest_distance:
            closest_distance = distance
            keypoint_ind = keypoint_idx
            
    return keypoint_ind

def get_height_of_box(box):
    return box[3] - box[1]

def measure_xy_distance(p1, p2):
    return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])

def get_center_of_box(box):
    return (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))