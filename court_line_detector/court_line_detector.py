import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2 as cv
import numpy

class CourtLineDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def predict(self, image):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_tensor = self.transforms(image).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(image_tensor)
        keypoints = outputs.squeeze().cpu().numpy()
        original_height, original_width = image.shape[:2]
        keypoints[::2] *= original_width / 224.0
        keypoints[1::2] *= original_height / 224.0
        
        return keypoints
    
    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            x, y = int(keypoints[i]), int(keypoints[i + 1])
            print(f"Drawing keypoint {i // 2} at ({x}, {y})")  # Debug print
            cv.putText(image, str(i // 2), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv.circle(image, (x, y), 5, (0, 100, 255), -1)
        return image

    
    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_frames.append(frame)
        return output_frames