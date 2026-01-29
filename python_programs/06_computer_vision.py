"""
Module 06: Computer Vision
==========================
Learn how computers see and understand images!
"""

# ============================================
# Computer Vision with OpenCV and PyTorch
# ============================================

code_examples = '''
"""
Computer Vision Examples
========================
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# ============================================
# 1. Basic Image Operations
# ============================================

def load_and_display(image_path):
    """Load and display an image"""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

def resize_image(img, width, height):
    """Resize an image"""
    return cv2.resize(img, (width, height))

def convert_to_grayscale(img):
    """Convert to grayscale"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ============================================
# 2. Edge Detection
# ============================================

def detect_edges(img, low_threshold=50, high_threshold=150):
    """Detect edges using Canny algorithm"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges


# ============================================
# 3. Face Detection
# ============================================

def detect_faces(img):
    """Detect faces using Haar cascades"""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return img, len(faces)


# ============================================
# 4. Image Classification with PyTorch
# ============================================

def classify_image(image_path):
    """Classify image using pre-trained ResNet"""
    # Load pre-trained model
    model = models.resnet50(pretrained=True)
    model.eval()
    
    # Preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load and preprocess image
    img = Image.open(image_path)
    img_tensor = preprocess(img).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = outputs.max(1)
    
    return predicted.item()


# ============================================
# 5. Object Detection with YOLO
# ============================================

def detect_objects_yolo(image_path):
    """Object detection using YOLOv5"""
    import torch
    
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    
    # Inference
    results = model(image_path)
    
    # Results
    results.print()  # Print results
    results.show()   # Display results
    
    return results.pandas().xyxy[0]  # Return as DataFrame


# ============================================
# 6. Image Segmentation
# ============================================

def segment_image(image_path):
    """Semantic segmentation using DeepLabV3"""
    from torchvision.models.segmentation import deeplabv3_resnet50
    
    model = deeplabv3_resnet50(pretrained=True)
    model.eval()
    
    # Load and preprocess
    img = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    input_tensor = preprocess(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    
    return output.argmax(0)


# ============================================
# 7. Data Augmentation
# ============================================

augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
])


# ============================================
# 8. Custom CNN for Image Classification
# ============================================

import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

print("Computer Vision examples loaded!")
print("Install: pip install opencv-python torch torchvision")
'''

def main():
    print("=" * 60)
    print("Computer Vision - Teaching Computers to See")
    print("=" * 60)
    
    print("""
Computer Vision enables machines to interpret visual information!

Key Tasks:
1. Image Classification - What is in this image?
2. Object Detection - Where are objects in the image?
3. Segmentation - Pixel-level understanding
4. Face Recognition - Identify people
5. Pose Estimation - Detect body positions

Popular Libraries:
- OpenCV: Image processing
- PyTorch/TensorFlow: Deep learning
- YOLO: Real-time object detection
- MediaPipe: Face/hand/pose detection
    """)
    
    print("\nPython Code Examples:")
    print(code_examples)

if __name__ == "__main__":
    main()
