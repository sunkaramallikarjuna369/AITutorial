import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Shield, AlertTriangle, Scale, Code, Eye, Lock, Users, Heart } from 'lucide-react'

const Module15Ethics = () => {
  const [selectedScenario, setSelectedScenario] = useState<number | null>(null)

  const scenarios = [
    {
      title: 'Biased Hiring AI',
      description: 'An AI system trained on historical hiring data recommends fewer women for technical roles.',
      issue: 'Training data reflects past discrimination',
      solution: 'Audit for bias, use diverse training data, implement fairness constraints',
      category: 'Bias'
    },
    {
      title: 'Deepfake Videos',
      description: 'AI-generated videos of politicians saying things they never said spread on social media.',
      issue: 'Misinformation and manipulation',
      solution: 'Watermarking, detection tools, media literacy education',
      category: 'Misuse'
    },
    {
      title: 'Medical AI Error',
      description: 'An AI diagnosis system misses a cancer case because the patient\'s symptoms were underrepresented in training.',
      issue: 'Lack of diverse training data, overreliance on AI',
      solution: 'Human oversight, diverse datasets, clear limitations disclosure',
      category: 'Safety'
    },
    {
      title: 'Surveillance AI',
      description: 'Facial recognition used to track citizens without consent in public spaces.',
      issue: 'Privacy violation, potential for abuse',
      solution: 'Strict regulations, consent requirements, transparency',
      category: 'Privacy'
    },
  ]

  const pythonCode = `# AI Ethics & Safety
# Building responsible AI systems

import numpy as np
from sklearn.metrics import confusion_matrix
import torch

# ============================================
# Bias Detection and Mitigation
# ============================================

class BiasAuditor:
    """
    Audit AI models for bias across different groups
    
    Key metrics:
    - Demographic Parity: Equal positive rates across groups
    - Equalized Odds: Equal TPR and FPR across groups
    - Calibration: Predictions mean the same thing for all groups
    """
    
    def __init__(self, predictions, labels, sensitive_attribute):
        self.predictions = np.array(predictions)
        self.labels = np.array(labels)
        self.sensitive = np.array(sensitive_attribute)
        self.groups = np.unique(self.sensitive)
    
    def demographic_parity(self):
        """
        Check if positive prediction rates are equal across groups
        
        Ideal: P(Y_pred=1 | A=0) = P(Y_pred=1 | A=1)
        """
        rates = {}
        for group in self.groups:
            mask = self.sensitive == group
            rates[group] = self.predictions[mask].mean()
        
        disparity = max(rates.values()) - min(rates.values())
        return rates, disparity
    
    def equalized_odds(self):
        """
        Check if TPR and FPR are equal across groups
        """
        metrics = {}
        for group in self.groups:
            mask = self.sensitive == group
            y_true = self.labels[mask]
            y_pred = self.predictions[mask]
            
            # True Positive Rate
            tpr = y_pred[y_true == 1].mean() if (y_true == 1).sum() > 0 else 0
            # False Positive Rate
            fpr = y_pred[y_true == 0].mean() if (y_true == 0).sum() > 0 else 0
            
            metrics[group] = {'TPR': tpr, 'FPR': fpr}
        
        return metrics
    
    def report(self):
        """Generate full bias report"""
        print("=== Bias Audit Report ===\\n")
        
        rates, disparity = self.demographic_parity()
        print("Demographic Parity:")
        for group, rate in rates.items():
            print(f"  Group {group}: {rate:.3f}")
        print(f"  Disparity: {disparity:.3f}")
        print(f"  Fair: {'Yes' if disparity < 0.1 else 'No'}\\n")
        
        odds = self.equalized_odds()
        print("Equalized Odds:")
        for group, metrics in odds.items():
            print(f"  Group {group}: TPR={metrics['TPR']:.3f}, FPR={metrics['FPR']:.3f}")

# Example usage
predictions = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
labels =      [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
gender =      [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0=male, 1=female

auditor = BiasAuditor(predictions, labels, gender)
auditor.report()

# ============================================
# Fairness Constraints in Training
# ============================================

def fair_loss(predictions, labels, sensitive, lambda_fair=1.0):
    """
    Loss function that penalizes unfair predictions
    
    Combines accuracy loss with fairness penalty
    """
    # Standard cross-entropy loss
    ce_loss = torch.nn.functional.binary_cross_entropy(
        predictions, labels.float()
    )
    
    # Fairness penalty: difference in positive rates
    group_0_mask = sensitive == 0
    group_1_mask = sensitive == 1
    
    rate_0 = predictions[group_0_mask].mean()
    rate_1 = predictions[group_1_mask].mean()
    
    fairness_penalty = torch.abs(rate_0 - rate_1)
    
    # Combined loss
    total_loss = ce_loss + lambda_fair * fairness_penalty
    
    return total_loss

# ============================================
# Model Interpretability
# ============================================

def explain_prediction(model, input_data, feature_names):
    """
    Simple feature importance explanation
    
    For more sophisticated explanations, use:
    - SHAP (SHapley Additive exPlanations)
    - LIME (Local Interpretable Model-agnostic Explanations)
    """
    import shap
    
    # Create SHAP explainer
    explainer = shap.Explainer(model)
    shap_values = explainer(input_data)
    
    # Get feature importances
    importances = np.abs(shap_values.values).mean(0)
    
    # Sort by importance
    sorted_idx = np.argsort(importances)[::-1]
    
    print("Feature Importances:")
    for idx in sorted_idx[:10]:
        print(f"  {feature_names[idx]}: {importances[idx]:.4f}")
    
    return shap_values

# Using LIME for local explanations
from lime.lime_tabular import LimeTabularExplainer

def lime_explain(model, X_train, X_test, feature_names, idx=0):
    """Explain a single prediction with LIME"""
    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=['Negative', 'Positive'],
        mode='classification'
    )
    
    exp = explainer.explain_instance(
        X_test[idx],
        model.predict_proba,
        num_features=10
    )
    
    return exp

# ============================================
# Privacy-Preserving AI
# ============================================

class DifferentialPrivacy:
    """
    Add noise to protect individual privacy
    
    Differential Privacy: Adding noise so that the presence
    or absence of any individual doesn't significantly
    change the output
    """
    
    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon  # Privacy budget
    
    def add_laplace_noise(self, value, sensitivity=1.0):
        """Add Laplace noise for differential privacy"""
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return value + noise
    
    def private_mean(self, data, bounds=(0, 1)):
        """Compute differentially private mean"""
        clipped = np.clip(data, bounds[0], bounds[1])
        true_mean = np.mean(clipped)
        sensitivity = (bounds[1] - bounds[0]) / len(data)
        return self.add_laplace_noise(true_mean, sensitivity)

# Federated Learning (train on distributed data without sharing)
"""
Federated Learning keeps data on user devices:
1. Server sends model to devices
2. Each device trains on local data
3. Devices send only model updates (not data)
4. Server aggregates updates

Benefits:
- Data never leaves the device
- Privacy preserved
- Can use sensitive data safely
"""

# ============================================
# AI Safety Measures
# ============================================

class SafeAIWrapper:
    """
    Wrapper to add safety checks to AI models
    """
    
    def __init__(self, model, confidence_threshold=0.7):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.human_review_queue = []
    
    def predict(self, input_data):
        """Make prediction with safety checks"""
        # Get prediction and confidence
        prediction = self.model.predict(input_data)
        confidence = self.model.predict_proba(input_data).max()
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            self.human_review_queue.append({
                'input': input_data,
                'prediction': prediction,
                'confidence': confidence
            })
            return {
                'prediction': prediction,
                'confidence': confidence,
                'status': 'NEEDS_REVIEW',
                'message': 'Low confidence - flagged for human review'
            }
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'status': 'OK'
        }
    
    def get_review_queue(self):
        """Get items needing human review"""
        return self.human_review_queue

# ============================================
# Content Moderation
# ============================================

class ContentModerator:
    """
    Filter harmful or inappropriate content
    """
    
    def __init__(self):
        self.harmful_patterns = [
            'violence', 'hate', 'harassment',
            'illegal', 'dangerous'
        ]
    
    def check_content(self, text):
        """Check if content is safe"""
        text_lower = text.lower()
        
        flags = []
        for pattern in self.harmful_patterns:
            if pattern in text_lower:
                flags.append(pattern)
        
        return {
            'safe': len(flags) == 0,
            'flags': flags,
            'confidence': 0.9 if flags else 0.95
        }
    
    def filter_output(self, ai_response):
        """Filter AI output before showing to user"""
        result = self.check_content(ai_response)
        
        if not result['safe']:
            return {
                'response': "[Content filtered for safety]",
                'original_flags': result['flags']
            }
        
        return {'response': ai_response}

# ============================================
# Responsible AI Checklist
# ============================================

responsible_ai_checklist = """
Before deploying an AI system, verify:

1. FAIRNESS
   [ ] Tested for bias across demographic groups
   [ ] Fairness metrics meet acceptable thresholds
   [ ] Diverse training data used

2. TRANSPARENCY
   [ ] Model decisions can be explained
   [ ] Users know they're interacting with AI
   [ ] Limitations are clearly documented

3. PRIVACY
   [ ] Data collection is minimized
   [ ] User consent obtained
   [ ] Data is properly secured

4. SAFETY
   [ ] Failure modes identified and handled
   [ ] Human oversight mechanisms in place
   [ ] Harmful outputs are filtered

5. ACCOUNTABILITY
   [ ] Clear ownership of the system
   [ ] Incident response plan exists
   [ ] Regular audits scheduled

6. ROBUSTNESS
   [ ] Tested against adversarial inputs
   [ ] Handles edge cases gracefully
   [ ] Monitoring in place for drift
"""`

  return (
    <div className="space-y-8">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-white mb-4">AI Ethics & Safety</h1>
        <p className="text-xl text-purple-200">Building responsible AI that benefits everyone!</p>
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
                <Shield className="w-6 h-6 text-rose-400" />
                Why AI Ethics Matters
              </CardTitle>
            </CardHeader>
            <CardContent className="text-slate-300 space-y-4">
              <p className="text-lg">
                <strong className="text-rose-400">AI Ethics</strong> ensures that artificial intelligence 
                systems are fair, transparent, and beneficial to society. As AI becomes more powerful, 
                we must ensure it's used responsibly!
              </p>
              <div className="bg-slate-900/50 p-4 rounded-lg border border-rose-500/30">
                <h4 className="text-rose-400 font-semibold mb-2">Think of it like this:</h4>
                <p>With great power comes great responsibility! AI can help millions of people, but if 
                built carelessly, it can also cause harm. Ethics helps us build AI that helps everyone 
                fairly and safely.</p>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Key Ethical Principles</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-4">
                {[
                  { icon: Scale, title: 'Fairness', desc: 'AI should treat all people equally, without discrimination', color: 'text-blue-400' },
                  { icon: Eye, title: 'Transparency', desc: 'People should understand how AI makes decisions', color: 'text-green-400' },
                  { icon: Lock, title: 'Privacy', desc: 'Personal data should be protected and respected', color: 'text-purple-400' },
                  { icon: Shield, title: 'Safety', desc: 'AI should not cause harm to people or society', color: 'text-red-400' },
                  { icon: Users, title: 'Accountability', desc: 'Someone should be responsible for AI decisions', color: 'text-orange-400' },
                  { icon: Heart, title: 'Beneficence', desc: 'AI should be designed to benefit humanity', color: 'text-pink-400' },
                ].map((principle, i) => (
                  <div key={i} className="bg-slate-900 p-4 rounded-lg">
                    <div className="flex items-center gap-3 mb-2">
                      <principle.icon className={`w-6 h-6 ${principle.color}`} />
                      <h4 className="text-white font-semibold">{principle.title}</h4>
                    </div>
                    <p className="text-slate-400 text-sm">{principle.desc}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-yellow-400" />
                Common AI Risks
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {[
                { risk: 'Bias & Discrimination', desc: 'AI learns biases from training data', example: 'Hiring AI favoring certain groups' },
                { risk: 'Privacy Violations', desc: 'Collecting or exposing personal data', example: 'Facial recognition without consent' },
                { risk: 'Misinformation', desc: 'AI generating false content', example: 'Deepfakes, fake news' },
                { risk: 'Job Displacement', desc: 'Automation replacing human workers', example: 'Customer service, manufacturing' },
                { risk: 'Security Threats', desc: 'AI used for malicious purposes', example: 'Automated hacking, surveillance' },
              ].map((item, i) => (
                <div key={i} className="flex items-start gap-3 p-3 bg-yellow-500/10 rounded-lg border border-yellow-500/30">
                  <AlertTriangle className="w-5 h-5 text-yellow-400 mt-0.5" />
                  <div>
                    <h4 className="text-white font-medium">{item.risk}</h4>
                    <p className="text-slate-400 text-sm">{item.desc}</p>
                    <p className="text-slate-500 text-xs mt-1">Example: {item.example}</p>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="visualize" className="space-y-6">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Scale className="w-5 h-5 text-rose-400" />
                Ethical Dilemmas in AI
              </CardTitle>
              <CardDescription className="text-slate-400">
                Click on a scenario to explore the ethical issues!
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid md:grid-cols-2 gap-3">
                {scenarios.map((scenario, i) => (
                  <Button
                    key={i}
                    variant={selectedScenario === i ? 'default' : 'outline'}
                    className={`h-auto py-3 justify-start text-left ${selectedScenario === i ? 'bg-rose-500' : ''}`}
                    onClick={() => setSelectedScenario(selectedScenario === i ? null : i)}
                  >
                    <div>
                      <p className="font-semibold">{scenario.title}</p>
                      <Badge variant="outline" className="mt-1 text-xs">{scenario.category}</Badge>
                    </div>
                  </Button>
                ))}
              </div>

              {selectedScenario !== null && (
                <div className="space-y-3 animate-in fade-in">
                  <div className="bg-slate-900 p-4 rounded-lg">
                    <h4 className="text-white font-semibold mb-2">{scenarios[selectedScenario].title}</h4>
                    <p className="text-slate-300 text-sm mb-3">{scenarios[selectedScenario].description}</p>
                    
                    <div className="grid md:grid-cols-2 gap-3">
                      <div className="bg-red-500/10 p-3 rounded border border-red-500/30">
                        <p className="text-red-400 font-medium text-sm">The Problem:</p>
                        <p className="text-slate-300 text-sm">{scenarios[selectedScenario].issue}</p>
                      </div>
                      <div className="bg-green-500/10 p-3 rounded border border-green-500/30">
                        <p className="text-green-400 font-medium text-sm">The Solution:</p>
                        <p className="text-slate-300 text-sm">{scenarios[selectedScenario].solution}</p>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Responsible AI Checklist</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-4">
                {[
                  { category: 'Fairness', items: ['Test for bias', 'Diverse training data', 'Equal treatment'] },
                  { category: 'Transparency', items: ['Explainable decisions', 'Clear documentation', 'User awareness'] },
                  { category: 'Privacy', items: ['Data minimization', 'User consent', 'Secure storage'] },
                  { category: 'Safety', items: ['Human oversight', 'Failure handling', 'Content filtering'] },
                ].map((section, i) => (
                  <div key={i} className="bg-slate-900 p-4 rounded-lg">
                    <h4 className="text-rose-400 font-semibold mb-2">{section.category}</h4>
                    <ul className="space-y-1">
                      {section.items.map((item, j) => (
                        <li key={j} className="flex items-center gap-2 text-slate-300 text-sm">
                          <div className="w-4 h-4 border border-slate-500 rounded" />
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">AI Governance Framework</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col items-center space-y-4">
                {[
                  { level: 'Principles', desc: 'High-level ethical guidelines', color: 'bg-blue-500' },
                  { level: 'Policies', desc: 'Organizational rules and standards', color: 'bg-green-500' },
                  { level: 'Processes', desc: 'Review and approval workflows', color: 'bg-purple-500' },
                  { level: 'Tools', desc: 'Technical implementations', color: 'bg-orange-500' },
                ].map((item, i) => (
                  <div key={i} className={`${item.color} text-white px-8 py-3 rounded-lg text-center w-full max-w-md`}>
                    <p className="font-semibold">{item.level}</p>
                    <p className="text-xs opacity-80">{item.desc}</p>
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
                <Code className="w-5 h-5 text-rose-400" />
                Python Code: AI Ethics & Safety
              </CardTitle>
              <CardDescription className="text-slate-400">
                Tools for bias detection, privacy, and safety!
              </CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="bg-slate-900 p-4 rounded-lg overflow-x-auto text-sm max-h-96 overflow-y-auto">
                <code className="text-green-400">{pythonCode}</code>
              </pre>
              <div className="mt-4 p-4 bg-rose-500/10 rounded-lg border border-rose-500/30">
                <h4 className="text-rose-400 font-semibold mb-2">Required Libraries:</h4>
                <code className="text-slate-300 text-sm bg-slate-800 px-2 py-1 rounded">
                  pip install numpy scikit-learn shap lime torch
                </code>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default Module15Ethics
