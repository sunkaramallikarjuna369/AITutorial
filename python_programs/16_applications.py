"""
Module 16: AI Applications
==========================
Real-world AI use cases across industries!
"""

def explain_applications():
    print("=" * 60)
    print("AI Applications Across Industries")
    print("=" * 60)
    print("""
AI is transforming every industry!

1. HEALTHCARE
   - Medical image analysis (X-rays, MRIs)
   - Drug discovery
   - Patient diagnosis assistance
   - Personalized treatment plans

2. FINANCE
   - Fraud detection
   - Algorithmic trading
   - Credit scoring
   - Risk assessment

3. AUTONOMOUS VEHICLES
   - Self-driving cars
   - Drones
   - Robotics

4. CREATIVE ARTS
   - AI art generation
   - Music composition
   - Video editing
   - Game design

5. EDUCATION
   - Personalized tutoring
   - Automated grading
   - Learning analytics

6. MANUFACTURING
   - Quality control
   - Predictive maintenance
   - Supply chain optimization
    """)

code_examples = '''
"""
AI Application Examples
=======================
"""

# ============================================
# 1. Medical Image Classification
# ============================================

import torch
import torch.nn as nn
from torchvision import models, transforms

class MedicalImageClassifier:
    def __init__(self, num_classes=2):
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict(self, image):
        self.model.eval()
        img_tensor = self.transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            prob = torch.softmax(output, dim=1)
        
        return {
            'prediction': 'positive' if prob[0][1] > 0.5 else 'negative',
            'confidence': prob[0].max().item()
        }


# ============================================
# 2. Fraud Detection
# ============================================

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class FraudDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = IsolationForest(contamination=0.01, random_state=42)
    
    def fit(self, transactions):
        scaled = self.scaler.fit_transform(transactions)
        self.model.fit(scaled)
    
    def predict(self, transaction):
        scaled = self.scaler.transform([transaction])
        prediction = self.model.predict(scaled)
        score = self.model.score_samples(scaled)
        
        return {
            'is_fraud': prediction[0] == -1,
            'anomaly_score': -score[0]
        }


# ============================================
# 3. Recommendation System
# ============================================

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RecommendationEngine:
    def __init__(self):
        self.user_item_matrix = None
        self.item_similarity = None
    
    def fit(self, ratings_matrix):
        self.user_item_matrix = ratings_matrix
        self.item_similarity = cosine_similarity(ratings_matrix.T)
    
    def recommend(self, user_id, n_recommendations=5):
        user_ratings = self.user_item_matrix[user_id]
        
        scores = np.zeros(len(user_ratings))
        for i, rating in enumerate(user_ratings):
            if rating > 0:
                scores += rating * self.item_similarity[i]
        
        # Exclude already rated items
        scores[user_ratings > 0] = -np.inf
        
        top_items = np.argsort(scores)[-n_recommendations:][::-1]
        return top_items


# ============================================
# 4. Chatbot / Customer Service
# ============================================

from openai import OpenAI

class CustomerServiceBot:
    def __init__(self):
        self.client = OpenAI()
        self.conversation_history = []
        
        self.system_prompt = """You are a helpful customer service agent.
        Be polite, professional, and helpful.
        If you cannot help, offer to connect with a human agent."""
    
    def chat(self, user_message):
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": self.system_prompt},
                *self.conversation_history
            ]
        )
        
        assistant_message = response.choices[0].message.content
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message


# ============================================
# 5. AI Tutoring System
# ============================================

class AITutor:
    def __init__(self, subject):
        self.client = OpenAI()
        self.subject = subject
        self.student_level = "beginner"
    
    def explain_concept(self, concept):
        prompt = f"""Explain {concept} in {self.subject} 
        for a {self.student_level} student.
        Use simple language and examples."""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
    
    def generate_quiz(self, topic, num_questions=5):
        prompt = f"""Create {num_questions} multiple choice questions 
        about {topic} in {self.subject} for a {self.student_level} student.
        Format: Question, A), B), C), D), Answer"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content


# ============================================
# 6. Object Detection for Autonomous Vehicles
# ============================================

def detect_objects_yolo(image_path):
    import torch
    
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    results = model(image_path)
    
    detections = results.pandas().xyxy[0]
    
    # Filter for relevant objects
    relevant = ['car', 'person', 'bicycle', 'truck', 'traffic light', 'stop sign']
    filtered = detections[detections['name'].isin(relevant)]
    
    return filtered


print("AI Application examples loaded!")
'''

def main():
    explain_applications()
    print("\nPython Code Examples:")
    print(code_examples)

if __name__ == "__main__":
    main()
