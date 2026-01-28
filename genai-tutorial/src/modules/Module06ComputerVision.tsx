import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Eye, Camera, Scan, Code, Image, Box } from 'lucide-react'

const Module06ComputerVision = () => {
  const [selectedFilter, setSelectedFilter] = useState<string | null>(null)
  const [detectedObjects, setDetectedObjects] = useState<string[]>([])

  const filters = [
    { name: 'Edge Detection', kernel: [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], color: 'blue' },
    { name: 'Blur', kernel: [[1,1,1],[1,1,1],[1,1,1]], color: 'green' },
    { name: 'Sharpen', kernel: [[0,-1,0],[-1,5,-1],[0,-1,0]], color: 'purple' },
  ]

  const simulateDetection = () => {
    const objects = ['Cat', 'Dog', 'Car', 'Person', 'Tree', 'Building']
    const detected = objects.filter(() => Math.random() > 0.5)
    setDetectedObjects(detected.length > 0 ? detected : ['No objects detected'])
  }

  const pythonCode = `# Computer Vision - Teaching Computers to See!
# From basic image processing to object detection

import numpy as np
import cv2  # OpenCV library

# ============================================
# Basic Image Operations
# ============================================

# Load an image
image = cv2.imread('photo.jpg')

# Get image properties
height, width, channels = image.shape
print(f"Image size: {width}x{height}, Channels: {channels}")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resize image
resized = cv2.resize(image, (224, 224))

# Save image
cv2.imwrite('output.jpg', gray)

# ============================================
# Image Filtering (Convolution)
# ============================================

def apply_filter(image, kernel):
    """
    Apply a filter to an image using convolution
    Like looking through different glasses!
    """
    return cv2.filter2D(image, -1, kernel)

# Edge detection kernel
edge_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

# Blur kernel (averaging)
blur_kernel = np.ones((5, 5)) / 25

# Sharpen kernel
sharpen_kernel = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
])

# Apply filters
edges = apply_filter(gray, edge_kernel)
blurred = apply_filter(image, blur_kernel)
sharpened = apply_filter(image, sharpen_kernel)

# ============================================
# Edge Detection with Canny
# ============================================

# Canny edge detection - finds edges automatically!
edges_canny = cv2.Canny(gray, 100, 200)

# ============================================
# Face Detection
# ============================================

# Load pre-trained face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def detect_faces(image):
    """
    Find faces in an image
    Like playing 'Where's Waldo' but for faces!
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return image, len(faces)

# ============================================
# Object Detection with YOLO
# ============================================

# Using YOLOv5 for object detection
import torch

# Load pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_objects(image_path):
    """
    Detect multiple objects in an image
    Like having super-powered eyes!
    """
    # Run inference
    results = model(image_path)
    
    # Get detections
    detections = results.pandas().xyxy[0]
    
    for _, row in detections.iterrows():
        print(f"Found: {row['name']} (confidence: {row['confidence']:.2f})")
    
    # Save result with bounding boxes
    results.save()
    
    return detections

# Example usage
# detections = detect_objects('street_scene.jpg')

# ============================================
# Image Classification with PyTorch
# ============================================

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Load pre-trained ResNet
model = models.resnet50(pretrained=True)
model.eval()

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def classify_image(image_path):
    """
    Classify what's in an image
    Like asking 'What is this?'
    """
    # Load and preprocess image
    image = Image.open(image_path)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_batch)
    
    # Get top 5 predictions
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_idx = torch.topk(probabilities, 5)
    
    # Load ImageNet labels
    with open('imagenet_classes.txt') as f:
        labels = [line.strip() for line in f.readlines()]
    
    for i in range(5):
        print(f"{labels[top5_idx[i]]}: {top5_prob[i].item()*100:.2f}%")

# ============================================
# Image Segmentation
# ============================================

from torchvision.models.segmentation import deeplabv3_resnet50

# Load segmentation model
seg_model = deeplabv3_resnet50(pretrained=True)
seg_model.eval()

def segment_image(image_path):
    """
    Divide image into regions
    Like coloring different parts of a picture!
    """
    image = Image.open(image_path)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = seg_model(input_batch)['out'][0]
    
    # Get class for each pixel
    segmentation = output.argmax(0)
    
    return segmentation

# ============================================
# Real-time Video Processing
# ============================================

def process_video():
    """
    Process video from webcam in real-time
    Like giving your computer eyes!
    """
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply processing (e.g., face detection)
        processed, num_faces = detect_faces(frame.copy())
        
        # Display result
        cv2.imshow('Computer Vision', processed)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# ============================================
# Using Hugging Face for Vision
# ============================================

from transformers import pipeline

# Image classification
classifier = pipeline("image-classification")
result = classifier("cat.jpg")
print(f"Classification: {result}")

# Object detection
detector = pipeline("object-detection")
result = detector("street.jpg")
print(f"Objects: {result}")

# Image captioning
captioner = pipeline("image-to-text")
result = captioner("beach.jpg")
print(f"Caption: {result}")`

  return (
    <div className="space-y-8">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-white mb-4">Computer Vision</h1>
        <p className="text-xl text-purple-200">Teaching computers to see and understand images!</p>
      </div>

      <Tabs defaultValue="learn" className="w-full">
        <TabsList className="grid w-full grid-cols-3 bg-slate-800">
          <TabsTrigger value="learn">Learn</TabsTrigger>
          <TabsTrigger value="visualize">Visualize</TabsTrigger>
          <TabsTrigger value="code">Python Code</TabsTrigger>
        </TabsList>

        <TabsContent value="learn" className="space-y-6">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Eye className="w-6 h-6 text-orange-400" />
                What is Computer Vision?
              </CardTitle>
            </CardHeader>
            <CardContent className="text-slate-300 space-y-4">
              <p className="text-lg">
                <strong className="text-orange-400">Computer Vision</strong> is the field of AI that teaches 
                computers to interpret and understand visual information from the world - just like how your 
                eyes and brain work together to see!
              </p>
              <div className="bg-slate-900/50 p-4 rounded-lg border border-orange-500/30">
                <h4 className="text-orange-400 font-semibold mb-2">Think of it like this:</h4>
                <p>When you look at a photo, you instantly know if it's a cat or a dog, where the objects are, 
                and what's happening. Computer vision gives machines this same ability - to "see" and understand 
                images and videos!</p>
              </div>
            </CardContent>
          </Card>

          <div className="grid md:grid-cols-3 gap-4">
            {[
              { icon: Image, title: 'Image Classification', desc: 'What is in this image?', example: 'Is this a cat or dog?', color: 'bg-blue-500' },
              { icon: Box, title: 'Object Detection', desc: 'Where are the objects?', example: 'Find all cars in the image', color: 'bg-green-500' },
              { icon: Scan, title: 'Segmentation', desc: 'Outline each object', example: 'Color each object differently', color: 'bg-purple-500' },
            ].map((item, i) => (
              <Card key={i} className="bg-slate-800/50 border-slate-700">
                <CardHeader>
                  <div className={`w-12 h-12 rounded-lg ${item.color} flex items-center justify-center mb-2`}>
                    <item.icon className="w-6 h-6 text-white" />
                  </div>
                  <CardTitle className="text-white text-lg">{item.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-slate-400 text-sm mb-2">{item.desc}</p>
                  <Badge variant="outline" className="text-slate-500">{item.example}</Badge>
                </CardContent>
              </Card>
            ))}
          </div>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">How Computers "See" Images</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid md:grid-cols-3 gap-4">
                <div className="bg-slate-900 p-4 rounded-lg text-center">
                  <h4 className="text-blue-400 font-semibold mb-2">1. Pixels</h4>
                  <div className="grid grid-cols-4 gap-1 mx-auto w-fit mb-2">
                    {Array(16).fill(0).map((_, i) => (
                      <div key={i} className={`w-4 h-4 rounded ${i % 3 === 0 ? 'bg-red-500' : i % 3 === 1 ? 'bg-green-500' : 'bg-blue-500'}`} />
                    ))}
                  </div>
                  <p className="text-slate-400 text-xs">Images are grids of colored dots</p>
                </div>
                <div className="bg-slate-900 p-4 rounded-lg text-center">
                  <h4 className="text-green-400 font-semibold mb-2">2. Numbers</h4>
                  <div className="grid grid-cols-3 gap-1 mx-auto w-fit mb-2 text-xs">
                    {[255, 128, 64, 200, 100, 50, 180, 90, 45].map((n, i) => (
                      <div key={i} className="w-8 h-8 bg-slate-700 rounded flex items-center justify-center text-slate-300">{n}</div>
                    ))}
                  </div>
                  <p className="text-slate-400 text-xs">Each pixel is RGB values (0-255)</p>
                </div>
                <div className="bg-slate-900 p-4 rounded-lg text-center">
                  <h4 className="text-purple-400 font-semibold mb-2">3. Patterns</h4>
                  <div className="text-4xl mb-2">🐱</div>
                  <p className="text-slate-400 text-xs">AI finds patterns in the numbers</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Real-World Applications</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-3">
                {[
                  { emoji: '🚗', title: 'Self-Driving Cars', desc: 'Detecting roads, signs, pedestrians' },
                  { emoji: '📱', title: 'Face Unlock', desc: 'Recognizing your face to unlock phone' },
                  { emoji: '🏥', title: 'Medical Imaging', desc: 'Finding tumors in X-rays and MRIs' },
                  { emoji: '🛒', title: 'Retail', desc: 'Cashier-less stores, inventory tracking' },
                  { emoji: '🔒', title: 'Security', desc: 'Surveillance and threat detection' },
                  { emoji: '📸', title: 'Photo Apps', desc: 'Filters, background removal, enhancement' },
                ].map((app, i) => (
                  <div key={i} className="flex items-center gap-3 p-3 bg-slate-900 rounded-lg">
                    <span className="text-2xl">{app.emoji}</span>
                    <div>
                      <h4 className="text-white font-medium">{app.title}</h4>
                      <p className="text-slate-400 text-sm">{app.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="visualize" className="space-y-6">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Camera className="w-5 h-5 text-orange-400" />
                Image Filters Visualization
              </CardTitle>
              <CardDescription className="text-slate-400">
                See how different filters transform images!
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid md:grid-cols-3 gap-4">
                {filters.map((filter, i) => (
                  <Button
                    key={i}
                    variant={selectedFilter === filter.name ? 'default' : 'outline'}
                    className={`h-auto py-4 flex-col ${selectedFilter === filter.name ? `bg-${filter.color}-600` : ''}`}
                    onClick={() => setSelectedFilter(filter.name)}
                  >
                    <span className="font-semibold">{filter.name}</span>
                    <div className="grid grid-cols-3 gap-0.5 mt-2">
                      {filter.kernel.flat().map((v, j) => (
                        <div key={j} className={`w-5 h-5 text-xs flex items-center justify-center rounded ${v > 0 ? 'bg-green-500/50' : v < 0 ? 'bg-red-500/50' : 'bg-slate-600'}`}>
                          {v}
                        </div>
                      ))}
                    </div>
                  </Button>
                ))}
              </div>

              {selectedFilter && (
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-slate-900 p-4 rounded-lg">
                    <h4 className="text-slate-400 text-sm mb-2">Original Image</h4>
                    <div className="aspect-video bg-gradient-to-br from-blue-500 to-purple-500 rounded-lg flex items-center justify-center">
                      <span className="text-6xl">🏔️</span>
                    </div>
                  </div>
                  <div className="bg-slate-900 p-4 rounded-lg">
                    <h4 className="text-slate-400 text-sm mb-2">After {selectedFilter}</h4>
                    <div className={`aspect-video rounded-lg flex items-center justify-center ${
                      selectedFilter === 'Edge Detection' ? 'bg-slate-800 border-2 border-white' :
                      selectedFilter === 'Blur' ? 'bg-gradient-to-br from-blue-500/50 to-purple-500/50 blur-sm' :
                      'bg-gradient-to-br from-blue-600 to-purple-600'
                    }`}>
                      <span className={`text-6xl ${selectedFilter === 'Blur' ? 'blur-sm' : ''}`}>🏔️</span>
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Box className="w-5 h-5 text-green-400" />
                Object Detection Simulator
              </CardTitle>
              <CardDescription className="text-slate-400">
                See how AI detects objects in images!
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="relative bg-slate-900 rounded-lg p-4 aspect-video">
                <div className="absolute inset-4 grid grid-cols-3 gap-4">
                  {['🐱', '🚗', '🌳', '🏠', '🐕', '👤'].map((emoji, i) => (
                    <div 
                      key={i} 
                      className={`flex items-center justify-center text-4xl rounded-lg transition-all ${
                        detectedObjects.includes(['Cat', 'Car', 'Tree', 'Building', 'Dog', 'Person'][i]) 
                          ? 'ring-2 ring-green-500 bg-green-500/20' 
                          : 'bg-slate-800'
                      }`}
                    >
                      {emoji}
                    </div>
                  ))}
                </div>
              </div>
              
              <Button onClick={simulateDetection} className="w-full bg-orange-600 hover:bg-orange-700">
                Run Object Detection
              </Button>

              {detectedObjects.length > 0 && (
                <div className="p-4 bg-green-500/10 rounded-lg border border-green-500/30">
                  <h4 className="text-green-400 font-semibold mb-2">Detected Objects:</h4>
                  <div className="flex flex-wrap gap-2">
                    {detectedObjects.map((obj, i) => (
                      <Badge key={i} className="bg-green-500">{obj}</Badge>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">CNN Feature Hierarchy</CardTitle>
              <CardDescription className="text-slate-400">
                How CNNs learn to see - from edges to objects
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col md:flex-row items-center justify-between gap-4">
                {[
                  { layer: 'Layer 1', features: 'Edges, Lines', visual: '╱ ╲ │ ─' },
                  { layer: 'Layer 2', features: 'Corners, Curves', visual: '◢ ◣ ◠ ◡' },
                  { layer: 'Layer 3', features: 'Textures, Patterns', visual: '▦ ▧ ▨ ▩' },
                  { layer: 'Layer 4', features: 'Parts', visual: '👁️ 👃 👂' },
                  { layer: 'Layer 5', features: 'Objects', visual: '🐱 🐕 🚗' },
                ].map((item, i) => (
                  <div key={i} className="flex flex-col items-center">
                    <div className="bg-slate-900 p-4 rounded-lg text-center w-32">
                      <p className="text-orange-400 font-semibold text-sm">{item.layer}</p>
                      <p className="text-2xl my-2">{item.visual}</p>
                      <p className="text-slate-400 text-xs">{item.features}</p>
                    </div>
                    {i < 4 && <span className="text-slate-500 text-2xl md:hidden">↓</span>}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="code" className="space-y-6">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Code className="w-5 h-5 text-orange-400" />
                Python Code: Computer Vision
              </CardTitle>
              <CardDescription className="text-slate-400">
                From basic image processing to deep learning vision!
              </CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="bg-slate-900 p-4 rounded-lg overflow-x-auto text-sm max-h-96 overflow-y-auto">
                <code className="text-green-400">{pythonCode}</code>
              </pre>
              <div className="mt-4 p-4 bg-orange-500/10 rounded-lg border border-orange-500/30">
                <h4 className="text-orange-400 font-semibold mb-2">Required Libraries:</h4>
                <code className="text-slate-300 text-sm bg-slate-800 px-2 py-1 rounded">
                  pip install opencv-python torch torchvision transformers pillow
                </code>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default Module06ComputerVision
