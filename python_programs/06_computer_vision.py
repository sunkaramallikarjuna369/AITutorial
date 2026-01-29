"""
=============================================================================
MODULE 06: COMPUTER VISION - COMPREHENSIVE 360 DEGREE COVERAGE
=============================================================================

This module provides COMPLETE coverage of Computer Vision including:
- 4W+H Explanations (What, Why, When, Where, How)
- Image Processing Fundamentals
- CNN Architectures
- Object Detection
- Image Segmentation
- GenAI Model Integrations
- Interview Questions
- Common Pitfalls

SETUP:
------
pip install opencv-python torch torchvision pillow numpy

For GenAI features:
pip install openai anthropic google-generativeai

=============================================================================
"""

import os
import math
from typing import List, Tuple, Dict, Optional

# Try to import optional libraries
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from torchvision import models
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =============================================================================
# GENAI INTEGRATION FOR COMPUTER VISION
# =============================================================================

class VisionAssistant:
    """
    Use GenAI models for vision tasks and explanations
    
    Supports: OpenAI (GPT-4V), Anthropic Claude, Google Gemini
    """
    
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.google_key = os.getenv("GOOGLE_API_KEY")
    
    def explain_with_openai(self, concept: str) -> str:
        """Get AI explanation using OpenAI GPT"""
        try:
            from openai import OpenAI
            if not self.openai_key:
                return f"[Set OPENAI_API_KEY for AI explanations]"
            
            client = OpenAI(api_key=self.openai_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "user",
                    "content": f"Explain {concept} in computer vision with a simple example and code snippet."
                }],
                max_tokens=500
            )
            return response.choices[0].message.content
        except ImportError:
            return "[Install openai: pip install openai]"
        except Exception as e:
            return f"[OpenAI Error: {e}]"
    
    def analyze_image_with_openai(self, image_path: str, question: str = "Describe this image") -> str:
        """Analyze image using GPT-4 Vision"""
        try:
            from openai import OpenAI
            import base64
            
            if not self.openai_key:
                return "[Set OPENAI_API_KEY for image analysis]"
            
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode()
            
            client = OpenAI(api_key=self.openai_key)
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                    ]
                }],
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[Vision Error: {e}]"
    
    def explain_with_ollama(self, concept: str, model: str = "llama2") -> str:
        """Get AI explanation using Ollama (FREE, local)"""
        try:
            import requests
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": f"Explain {concept} in computer vision with an example.",
                    "stream": False
                },
                timeout=60
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            return f"[Ollama error: {response.status_code}]"
        except Exception as e:
            return f"[Start Ollama: ollama serve. Error: {e}]"


# =============================================================================
# SECTION 1: COMPREHENSIVE CV EXPLANATION (4W+H)
# =============================================================================

def explain_cv_comprehensive():
    """Comprehensive 360-degree explanation of Computer Vision"""
    print("=" * 70)
    print("COMPUTER VISION - COMPREHENSIVE 360 DEGREE COVERAGE")
    print("=" * 70)
    
    print("""
WHAT IS COMPUTER VISION?
========================

Computer Vision is a field of AI that enables computers to interpret and
understand visual information from the world (images, videos).

KEY TASKS:
    1. Image Classification - What is in this image?
    2. Object Detection - Where are objects? (bounding boxes)
    3. Semantic Segmentation - Pixel-level classification
    4. Instance Segmentation - Separate each object instance
    5. Pose Estimation - Detect body/hand positions
    6. Face Recognition - Identify people
    7. OCR - Read text from images

WHY IS COMPUTER VISION IMPORTANT?
=================================

1. AUTOMATION: Automate visual inspection, quality control
2. SAFETY: Self-driving cars, surveillance, medical diagnosis
3. ACCESSIBILITY: Help visually impaired people
4. EFFICIENCY: Process millions of images automatically

WHEN TO USE COMPUTER VISION?
============================

USE CV FOR:
    - Quality control in manufacturing
    - Medical image analysis (X-rays, MRIs)
    - Autonomous vehicles
    - Security and surveillance
    - Augmented reality
    - Document processing (OCR)

WHERE IS COMPUTER VISION USED?
==============================

INDUSTRIES:
    - Healthcare: Tumor detection, retinal scans
    - Automotive: Tesla Autopilot, Waymo
    - Retail: Amazon Go, inventory management
    - Agriculture: Crop monitoring, disease detection
    - Security: Face recognition, anomaly detection

PRODUCTS:
    - Google Photos (face clustering)
    - iPhone Face ID
    - Tesla Autopilot
    - Snapchat/Instagram filters

HOW DOES COMPUTER VISION WORK?
==============================

TRADITIONAL APPROACH:
    1. Preprocessing: Resize, normalize, denoise
    2. Feature Extraction: SIFT, HOG, edge detection
    3. Classification: SVM, Random Forest

DEEP LEARNING APPROACH:
    1. Input: Raw pixels (H x W x C)
    2. Convolutional Layers: Extract features automatically
    3. Pooling: Reduce spatial dimensions
    4. Fully Connected: Classification
    5. Output: Class probabilities

CNN ARCHITECTURE:
    Input -> [Conv -> ReLU -> Pool] x N -> Flatten -> Dense -> Output
    """)


# =============================================================================
# SECTION 2: HISTORY AND EVOLUTION
# =============================================================================

def cv_history():
    """History and evolution of Computer Vision"""
    print("\n" + "=" * 70)
    print("HISTORY AND EVOLUTION OF COMPUTER VISION")
    print("=" * 70)
    
    timeline = [
        ("1960s", "Early Research", "Edge detection, pattern recognition"),
        ("1980s", "Feature-Based", "SIFT, HOG, Haar features"),
        ("1998", "LeNet-5", "First successful CNN for digits"),
        ("2009", "ImageNet", "Large-scale visual recognition dataset"),
        ("2012", "AlexNet", "Deep learning revolution, 15% error reduction"),
        ("2014", "VGGNet", "Deeper networks (16-19 layers)"),
        ("2014", "GoogLeNet", "Inception modules, efficient architecture"),
        ("2015", "ResNet", "Skip connections, 152 layers"),
        ("2015", "YOLO", "Real-time object detection"),
        ("2017", "Mask R-CNN", "Instance segmentation"),
        ("2020", "Vision Transformer", "Transformers for images"),
        ("2021", "CLIP", "Vision-language models"),
        ("2023", "SAM", "Segment Anything Model"),
    ]
    
    print("\nTIMELINE:")
    print("-" * 70)
    for year, event, description in timeline:
        print(f"{year}: {event} - {description}")


# =============================================================================
# SECTION 3: CNN ARCHITECTURES
# =============================================================================

def cnn_architectures():
    """Overview of CNN architectures"""
    print("\n" + "=" * 70)
    print("CNN ARCHITECTURES")
    print("=" * 70)
    
    architectures = [
        {
            "name": "LeNet-5 (1998)",
            "layers": "7 layers",
            "innovation": "First successful CNN",
            "use": "Digit recognition"
        },
        {
            "name": "AlexNet (2012)",
            "layers": "8 layers, 60M params",
            "innovation": "ReLU, Dropout, GPU training",
            "use": "ImageNet classification"
        },
        {
            "name": "VGGNet (2014)",
            "layers": "16-19 layers, 138M params",
            "innovation": "Small 3x3 filters throughout",
            "use": "Feature extraction"
        },
        {
            "name": "GoogLeNet/Inception (2014)",
            "layers": "22 layers, 5M params",
            "innovation": "Inception modules (parallel convolutions)",
            "use": "Efficient classification"
        },
        {
            "name": "ResNet (2015)",
            "layers": "50-152 layers",
            "innovation": "Skip connections (residual learning)",
            "use": "Very deep networks"
        },
        {
            "name": "EfficientNet (2019)",
            "layers": "Scalable",
            "innovation": "Compound scaling (depth, width, resolution)",
            "use": "State-of-the-art efficiency"
        },
        {
            "name": "Vision Transformer (2020)",
            "layers": "Transformer-based",
            "innovation": "Self-attention for images",
            "use": "Large-scale training"
        },
    ]
    
    for arch in architectures:
        print(f"\n{arch['name']}")
        print(f"  Layers: {arch['layers']}")
        print(f"  Innovation: {arch['innovation']}")
        print(f"  Use: {arch['use']}")


# =============================================================================
# SECTION 4: IMAGE PROCESSING FUNDAMENTALS
# =============================================================================

def image_processing_demo():
    """Demo basic image processing"""
    print("\n" + "=" * 70)
    print("IMAGE PROCESSING FUNDAMENTALS")
    print("=" * 70)
    
    if not HAS_NUMPY:
        print("\nInstall numpy: pip install numpy")
    
    print("""
IMAGE REPRESENTATION:
    - Grayscale: 2D array (H x W), values 0-255
    - Color (RGB): 3D array (H x W x 3)
    - Normalized: Values 0-1 (divide by 255)

COMMON OPERATIONS:

1. RESIZING:
   cv2.resize(img, (width, height))
   
2. GRAYSCALE CONVERSION:
   cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
3. EDGE DETECTION (Canny):
   edges = cv2.Canny(gray, low_threshold, high_threshold)
   
4. BLURRING (Gaussian):
   blurred = cv2.GaussianBlur(img, (5, 5), 0)
   
5. THRESHOLDING:
   _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

6. MORPHOLOGICAL OPERATIONS:
   kernel = np.ones((5,5), np.uint8)
   dilated = cv2.dilate(img, kernel, iterations=1)
   eroded = cv2.erode(img, kernel, iterations=1)
    """)
    
    if HAS_NUMPY:
        print("\nDEMO: Creating a simple image")
        # Create a simple 5x5 grayscale image
        img = np.array([
            [0, 0, 255, 0, 0],
            [0, 255, 255, 255, 0],
            [255, 255, 255, 255, 255],
            [0, 255, 255, 255, 0],
            [0, 0, 255, 0, 0]
        ], dtype=np.uint8)
        print("5x5 Diamond pattern:")
        print(img)
        print(f"Shape: {img.shape}, dtype: {img.dtype}")


# =============================================================================
# SECTION 5: PYTORCH CNN IMPLEMENTATION
# =============================================================================

def pytorch_cnn_demo():
    """Demo CNN implementation with PyTorch"""
    print("\n" + "=" * 70)
    print("PYTORCH CNN IMPLEMENTATION")
    print("=" * 70)
    
    if not HAS_TORCH:
        print("\nInstall PyTorch: pip install torch torchvision")
        print("""
Example CNN code:

import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
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
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
        """)
        return
    
    # Define CNN
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 8 * 8, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x
    
    model = SimpleCNN(num_classes=10)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test with random input
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")


# =============================================================================
# SECTION 6: OBJECT DETECTION
# =============================================================================

def object_detection_overview():
    """Overview of object detection methods"""
    print("\n" + "=" * 70)
    print("OBJECT DETECTION")
    print("=" * 70)
    
    print("""
OBJECT DETECTION = Classification + Localization

OUTPUT: Bounding boxes (x, y, width, height) + class labels + confidence

TWO-STAGE DETECTORS (Accurate but slower):
    1. R-CNN (2014): Region proposals + CNN classification
    2. Fast R-CNN (2015): Shared CNN features
    3. Faster R-CNN (2015): Region Proposal Network (RPN)
    4. Mask R-CNN (2017): + Instance segmentation

ONE-STAGE DETECTORS (Fast, real-time):
    1. YOLO (2016): You Only Look Once - single pass
    2. SSD (2016): Single Shot Detector
    3. RetinaNet (2017): Focal loss for class imbalance
    4. YOLOv5/v8 (2020+): State-of-the-art speed/accuracy

METRICS:
    - IoU (Intersection over Union): Overlap between predicted and ground truth
    - mAP (mean Average Precision): Average precision across classes
    - FPS (Frames Per Second): Speed metric

YOLO EXAMPLE:
    import torch
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    results = model('image.jpg')
    results.show()  # Display results
    """)


# =============================================================================
# SECTION 7: IMAGE SEGMENTATION
# =============================================================================

def segmentation_overview():
    """Overview of image segmentation"""
    print("\n" + "=" * 70)
    print("IMAGE SEGMENTATION")
    print("=" * 70)
    
    print("""
TYPES OF SEGMENTATION:

1. SEMANTIC SEGMENTATION:
   - Classify each pixel into a category
   - All cats are "cat", all dogs are "dog"
   - Models: FCN, U-Net, DeepLab

2. INSTANCE SEGMENTATION:
   - Separate each object instance
   - Cat 1, Cat 2, Dog 1, etc.
   - Models: Mask R-CNN, YOLACT

3. PANOPTIC SEGMENTATION:
   - Combines semantic + instance
   - Both stuff (sky, road) and things (cars, people)

KEY ARCHITECTURES:

U-Net (2015):
    - Encoder-decoder with skip connections
    - Great for medical imaging
    - Preserves spatial information

DeepLab (2017):
    - Atrous (dilated) convolutions
    - Atrous Spatial Pyramid Pooling (ASPP)
    - Multi-scale feature extraction

Segment Anything (SAM, 2023):
    - Foundation model for segmentation
    - Zero-shot segmentation
    - Prompt-based (points, boxes, text)
    """)


# =============================================================================
# SECTION 8: INTERVIEW QUESTIONS
# =============================================================================

def interview_questions():
    """Common CV interview questions"""
    print("\n" + "=" * 70)
    print("COMPUTER VISION INTERVIEW QUESTIONS")
    print("=" * 70)
    
    questions = [
        ("What is a convolution operation?",
         "Sliding a filter/kernel over an image, computing element-wise multiplication and sum. Extracts features like edges, textures."),
        ("What is pooling and why is it used?",
         "Downsampling operation (max/avg pooling) that reduces spatial dimensions, provides translation invariance, and reduces computation."),
        ("What is the vanishing gradient problem in deep CNNs?",
         "Gradients become very small in early layers. Solutions: ReLU activation, skip connections (ResNet), batch normalization."),
        ("Explain transfer learning in CV.",
         "Use pre-trained model (e.g., ImageNet) as starting point. Freeze early layers (generic features), fine-tune later layers for specific task."),
        ("What is data augmentation?",
         "Creating variations of training images (flip, rotate, crop, color jitter) to increase dataset size and improve generalization."),
        ("What is IoU and how is it calculated?",
         "Intersection over Union = Area of Overlap / Area of Union. Measures how well predicted box matches ground truth. IoU > 0.5 is typically 'correct'."),
    ]
    
    for i, (q, a) in enumerate(questions, 1):
        print(f"\nQ{i}: {q}")
        print(f"A: {a}")


# =============================================================================
# SECTION 9: COMMON PITFALLS
# =============================================================================

def common_pitfalls():
    """Common mistakes in CV"""
    print("\n" + "=" * 70)
    print("COMMON CV PITFALLS")
    print("=" * 70)
    
    pitfalls = [
        ("Not normalizing images", "Normalize to [0,1] or use ImageNet mean/std"),
        ("Wrong input dimensions", "Check if model expects (B,C,H,W) or (B,H,W,C)"),
        ("Data leakage in augmentation", "Apply augmentation only to training data"),
        ("Ignoring class imbalance", "Use weighted loss, oversampling, or focal loss"),
        ("Not using pretrained models", "Transfer learning almost always helps"),
        ("Training on small images", "Higher resolution often improves accuracy"),
    ]
    
    for mistake, solution in pitfalls:
        print(f"\nMistake: {mistake}")
        print(f"Solution: {solution}")


# =============================================================================
# MAIN PROGRAM
# =============================================================================

def main():
    print("=" * 70)
    print("   MODULE 06: COMPUTER VISION - COMPREHENSIVE 360 DEGREE COVERAGE")
    print("=" * 70)
    
    assistant = VisionAssistant()
    
    while True:
        print("\n" + "-" * 70)
        print("Choose a topic:")
        print()
        print("FUNDAMENTALS:")
        print("  1. What is Computer Vision? (4W+H)")
        print("  2. History and Evolution")
        print("  3. CNN Architectures")
        print()
        print("TECHNIQUES:")
        print("  4. Image Processing Fundamentals")
        print("  5. PyTorch CNN Implementation")
        print("  6. Object Detection Overview")
        print("  7. Image Segmentation Overview")
        print()
        print("DEEP DIVE:")
        print("  8. Interview Questions")
        print("  9. Common Pitfalls")
        print()
        print("GENAI FEATURES:")
        print("  10. AI Explanation (OpenAI)")
        print("  11. AI Explanation (Ollama - FREE)")
        print()
        print("  12. Run ALL Topics")
        print("  0. Exit")
        print("-" * 70)
        
        choice = input("\nEnter choice (0-12): ").strip()
        
        if choice == "0":
            print("\nHappy learning!")
            break
        elif choice == "1":
            explain_cv_comprehensive()
        elif choice == "2":
            cv_history()
        elif choice == "3":
            cnn_architectures()
        elif choice == "4":
            image_processing_demo()
        elif choice == "5":
            pytorch_cnn_demo()
        elif choice == "6":
            object_detection_overview()
        elif choice == "7":
            segmentation_overview()
        elif choice == "8":
            interview_questions()
        elif choice == "9":
            common_pitfalls()
        elif choice == "10":
            concept = input("Enter concept to explain: ").strip()
            print(assistant.explain_with_openai(concept))
        elif choice == "11":
            concept = input("Enter concept to explain: ").strip()
            print(assistant.explain_with_ollama(concept))
        elif choice == "12":
            explain_cv_comprehensive()
            cv_history()
            cnn_architectures()
            image_processing_demo()
            pytorch_cnn_demo()
            object_detection_overview()
            segmentation_overview()
            interview_questions()
            common_pitfalls()
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
