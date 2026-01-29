"""
Module 15: AI Ethics & Safety
=============================
Building responsible and fair AI systems!
"""

def explain_ethics():
    print("=" * 60)
    print("AI Ethics & Safety")
    print("=" * 60)
    print("""
AI Ethics = Making sure AI is fair, safe, and beneficial!

Key Concerns:

1. BIAS & FAIRNESS
   - AI can learn biases from data
   - Example: Hiring AI that discriminates
   - Solution: Audit for bias, diverse training data

2. PRIVACY
   - AI needs data, but data is personal
   - Example: Facial recognition surveillance
   - Solution: Differential privacy, federated learning

3. TRANSPARENCY
   - "Black box" AI is hard to trust
   - Example: Why was my loan denied?
   - Solution: Explainable AI (XAI)

4. SAFETY
   - AI should not cause harm
   - Example: Self-driving car accidents
   - Solution: Rigorous testing, human oversight

5. MISINFORMATION
   - AI can generate fake content
   - Example: Deepfakes, fake news
   - Solution: Detection tools, watermarking

Responsible AI Principles:
- Fairness: Treat all groups equitably
- Accountability: Clear responsibility
- Transparency: Explainable decisions
- Privacy: Protect personal data
- Safety: Prevent harm
    """)

code_examples = '''
"""
AI Ethics & Safety Examples
===========================
"""

import numpy as np
from sklearn.metrics import confusion_matrix

# ============================================
# 1. Bias Detection
# ============================================

class BiasAuditor:
    """Detect bias in model predictions"""
    
    def __init__(self, model, protected_attribute):
        self.model = model
        self.protected_attr = protected_attribute
    
    def demographic_parity(self, X, sensitive_features):
        """Check if positive rates are equal across groups"""
        predictions = self.model.predict(X)
        
        groups = {}
        for i, group in enumerate(sensitive_features):
            if group not in groups:
                groups[group] = []
            groups[group].append(predictions[i])
        
        rates = {}
        for group, preds in groups.items():
            rates[group] = sum(preds) / len(preds)
        
        return rates
    
    def equalized_odds(self, X, y_true, sensitive_features):
        """Check if TPR and FPR are equal across groups"""
        predictions = self.model.predict(X)
        
        results = {}
        for group in set(sensitive_features):
            mask = [s == group for s in sensitive_features]
            y_group = [y for y, m in zip(y_true, mask) if m]
            pred_group = [p for p, m in zip(predictions, mask) if m]
            
            tn, fp, fn, tp = confusion_matrix(y_group, pred_group).ravel()
            
            results[group] = {
                'tpr': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0
            }
        
        return results


# ============================================
# 2. Fairness Constraints
# ============================================

from sklearn.linear_model import LogisticRegression

class FairClassifier:
    """Classifier with fairness constraints"""
    
    def __init__(self, fairness_weight=0.5):
        self.model = LogisticRegression()
        self.fairness_weight = fairness_weight
    
    def fit(self, X, y, sensitive_features):
        # Train with reweighting for fairness
        weights = self._compute_weights(y, sensitive_features)
        self.model.fit(X, y, sample_weight=weights)
    
    def _compute_weights(self, y, sensitive_features):
        """Compute sample weights to balance groups"""
        weights = np.ones(len(y))
        
        for group in set(sensitive_features):
            mask = [s == group for s in sensitive_features]
            group_size = sum(mask)
            
            for i, m in enumerate(mask):
                if m:
                    weights[i] = len(y) / (2 * group_size)
        
        return weights


# ============================================
# 3. Differential Privacy
# ============================================

class DifferentialPrivacy:
    """Add noise for privacy protection"""
    
    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon
    
    def add_laplace_noise(self, value, sensitivity=1.0):
        """Add Laplace noise to protect privacy"""
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return value + noise
    
    def private_mean(self, data, sensitivity=1.0):
        """Compute mean with differential privacy"""
        true_mean = np.mean(data)
        return self.add_laplace_noise(true_mean, sensitivity / len(data))


# ============================================
# 4. Explainable AI (XAI)
# ============================================

import shap

def explain_prediction(model, X, instance_idx):
    """Explain a single prediction using SHAP"""
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X[instance_idx:instance_idx+1])
    
    # Plot explanation
    shap.plots.waterfall(shap_values[0])
    
    return shap_values


# ============================================
# 5. Model Cards
# ============================================

def create_model_card(model_name, description, intended_use, 
                      limitations, ethical_considerations):
    """Create a model card for documentation"""
    
    card = f"""
# Model Card: {model_name}

## Description
{description}

## Intended Use
{intended_use}

## Limitations
{limitations}

## Ethical Considerations
{ethical_considerations}

## Bias Evaluation
[Include bias audit results here]

## Training Data
[Describe training data and any known biases]
"""
    return card


# ============================================
# 6. Content Moderation
# ============================================

class ContentModerator:
    """Detect harmful content"""
    
    def __init__(self, toxicity_model):
        self.model = toxicity_model
    
    def check_content(self, text):
        """Check if content is harmful"""
        score = self.model.predict(text)
        
        return {
            'text': text,
            'toxicity_score': score,
            'is_toxic': score > 0.7,
            'action': 'block' if score > 0.9 else 'review' if score > 0.7 else 'allow'
        }


print("AI Ethics examples loaded!")
print("Install: pip install shap fairlearn")
'''

def main():
    explain_ethics()
    print("\nPython Code Examples:")
    print(code_examples)

if __name__ == "__main__":
    main()
