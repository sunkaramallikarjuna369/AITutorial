"""
=============================================================================
MODULE 2: MACHINE LEARNING BASICS - 360 DEGREE COVERAGE
=============================================================================

This comprehensive module covers:
- What is Machine Learning and types (Supervised, Unsupervised, Reinforcement)
- Linear Regression from scratch
- K-Nearest Neighbors classification
- Decision Trees
- Train/Test Split and Cross-Validation
- Overfitting vs Underfitting
- Gradient Descent optimization
- Using GenAI models to explain ML concepts
- Scikit-learn examples

SETUP INSTRUCTIONS:
-------------------
1. Install required packages:
   pip install scikit-learn numpy pandas matplotlib
   pip install openai anthropic google-generativeai requests

2. Set up API keys for GenAI features:
   export OPENAI_API_KEY="your-key"
   export ANTHROPIC_API_KEY="your-key"

3. For Ollama (FREE local models):
   Install from https://ollama.ai
   Run: ollama pull llama2

=============================================================================
"""

import random
import math
import os
from typing import Optional, List, Tuple


# =============================================================================
# GENAI HELPER - Use AI to explain ML concepts
# =============================================================================

class MLExplainer:
    """Use GenAI models to explain ML concepts in simple terms"""
    
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
    
    def explain_with_openai(self, concept: str) -> str:
        """Get AI explanation of an ML concept"""
        try:
            from openai import OpenAI
            if not self.openai_key:
                return f"[Set OPENAI_API_KEY to get AI explanations of {concept}]"
            
            client = OpenAI(api_key=self.openai_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "user", 
                    "content": f"Explain {concept} in machine learning to a beginner in 3 sentences."
                }],
                max_tokens=200
            )
            return response.choices[0].message.content
        except:
            return f"[Install openai package for AI explanations]"
    
    def explain_with_ollama(self, concept: str) -> str:
        """Get local AI explanation using Ollama"""
        try:
            import requests
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama2",
                    "prompt": f"Explain {concept} in machine learning simply in 3 sentences.",
                    "stream": False
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            return "[Ollama not responding]"
        except:
            return "[Start Ollama with: ollama serve]"

# ============================================
# What is Machine Learning?
# ============================================

def explain_machine_learning():
    """
    Explain what machine learning is in simple terms
    """
    print("=" * 60)
    print("What is Machine Learning?")
    print("=" * 60)
    
    print("""
Machine Learning is teaching computers to learn from examples!

Traditional Programming:
    Input + Rules → Output
    Example: "If temperature > 30, say 'hot'"

Machine Learning:
    Input + Output → Rules (learned automatically!)
    Example: Show many temperatures labeled 'hot' or 'cold',
             computer learns the pattern itself!

Types of Machine Learning:
1. Supervised Learning   - Learn from labeled examples
2. Unsupervised Learning - Find patterns without labels
3. Reinforcement Learning - Learn from rewards/penalties
    """)


# ============================================
# Linear Regression - Predicting Numbers
# ============================================

class SimpleLinearRegression:
    """
    The simplest ML algorithm - fitting a line to data!
    
    y = mx + b
    where m is slope and b is intercept
    """
    
    def __init__(self):
        self.slope = 0
        self.intercept = 0
    
    def fit(self, X, y):
        """
        Learn the best line from training data
        
        Uses the formula for least squares regression
        """
        n = len(X)
        
        # Calculate means
        mean_x = sum(X) / n
        mean_y = sum(y) / n
        
        # Calculate slope
        numerator = sum((X[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denominator = sum((X[i] - mean_x) ** 2 for i in range(n))
        
        self.slope = numerator / denominator if denominator != 0 else 0
        self.intercept = mean_y - self.slope * mean_x
        
        print(f"Learned equation: y = {self.slope:.2f}x + {self.intercept:.2f}")
    
    def predict(self, x):
        """Make a prediction for new data"""
        return self.slope * x + self.intercept
    
    def score(self, X, y):
        """Calculate R² score (how good is our model)"""
        predictions = [self.predict(x) for x in X]
        mean_y = sum(y) / len(y)
        
        ss_tot = sum((yi - mean_y) ** 2 for yi in y)
        ss_res = sum((y[i] - predictions[i]) ** 2 for i in range(len(y)))
        
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        return r2


def linear_regression_demo():
    """
    Demo: Predict house prices based on size
    """
    print("\n" + "=" * 60)
    print("Linear Regression Demo: Predicting House Prices")
    print("=" * 60)
    
    # Training data: house size (sq ft) and price ($1000s)
    sizes = [1000, 1500, 2000, 2500, 3000, 3500, 4000]
    prices = [150, 200, 250, 300, 350, 400, 450]
    
    print("\nTraining Data:")
    print("-" * 40)
    for size, price in zip(sizes, prices):
        print(f"  {size} sq ft → ${price}k")
    
    # Train the model
    print("\nTraining the model...")
    model = SimpleLinearRegression()
    model.fit(sizes, prices)
    
    # Make predictions
    print("\nPredictions for new houses:")
    print("-" * 40)
    test_sizes = [1200, 2200, 3200]
    for size in test_sizes:
        prediction = model.predict(size)
        print(f"  {size} sq ft → ${prediction:.0f}k (predicted)")
    
    # Show accuracy
    r2 = model.score(sizes, prices)
    print(f"\nModel accuracy (R²): {r2:.2%}")


# ============================================
# K-Nearest Neighbors - Classification
# ============================================

def euclidean_distance(point1, point2):
    """Calculate distance between two points"""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))


class SimpleKNN:
    """
    K-Nearest Neighbors - classify by looking at neighbors!
    
    "You are the average of the 5 people you spend the most time with"
    """
    
    def __init__(self, k=3):
        self.k = k
        self.X_train = []
        self.y_train = []
    
    def fit(self, X, y):
        """Store training data"""
        self.X_train = X
        self.y_train = y
        print(f"Stored {len(X)} training examples")
    
    def predict(self, x):
        """Find k nearest neighbors and vote"""
        # Calculate distances to all training points
        distances = []
        for i, train_point in enumerate(self.X_train):
            dist = euclidean_distance(x, train_point)
            distances.append((dist, self.y_train[i]))
        
        # Sort by distance and get k nearest
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.k]
        
        # Vote for the most common label
        labels = [label for _, label in k_nearest]
        return max(set(labels), key=labels.count)


def knn_demo():
    """
    Demo: Classify fruits by weight and color
    """
    print("\n" + "=" * 60)
    print("K-Nearest Neighbors Demo: Classifying Fruits")
    print("=" * 60)
    
    # Training data: [weight (g), sweetness (1-10)] → fruit type
    X_train = [
        [150, 8],  # Apple
        [160, 7],  # Apple
        [140, 9],  # Apple
        [180, 3],  # Orange
        [190, 4],  # Orange
        [170, 2],  # Orange
        [120, 10], # Grape (small, sweet)
        [110, 9],  # Grape
    ]
    y_train = ["Apple", "Apple", "Apple", "Orange", "Orange", "Orange", "Grape", "Grape"]
    
    print("\nTraining Data:")
    print("-" * 40)
    print("Weight(g) | Sweetness | Fruit")
    for features, label in zip(X_train, y_train):
        print(f"   {features[0]}    |     {features[1]}     | {label}")
    
    # Train the model
    model = SimpleKNN(k=3)
    model.fit(X_train, y_train)
    
    # Make predictions
    print("\nPredictions for unknown fruits:")
    print("-" * 40)
    test_fruits = [[155, 8], [175, 3], [115, 9]]
    for fruit in test_fruits:
        prediction = model.predict(fruit)
        print(f"  Weight={fruit[0]}g, Sweetness={fruit[1]} → {prediction}")


# ============================================
# Decision Tree - Simple Rules
# ============================================

def decision_tree_demo():
    """
    Demo: Simple decision tree for weather prediction
    """
    print("\n" + "=" * 60)
    print("Decision Tree Demo: Should I Play Tennis?")
    print("=" * 60)
    
    def should_play_tennis(outlook, temperature, humidity, wind):
        """
        A simple decision tree learned from data
        
        This is what a decision tree algorithm would learn!
        """
        if outlook == "sunny":
            if humidity == "high":
                return "No"
            else:
                return "Yes"
        elif outlook == "overcast":
            return "Yes"
        else:  # rainy
            if wind == "strong":
                return "No"
            else:
                return "Yes"
    
    print("""
Decision Tree Structure:
                    [Outlook?]
                   /    |    \\
              Sunny  Overcast  Rainy
               /        |        \\
        [Humidity?]    Yes    [Wind?]
          /    \\              /    \\
       High   Normal     Strong  Weak
        |       |          |       |
       No      Yes        No      Yes
    """)
    
    # Test cases
    test_cases = [
        ("sunny", "hot", "high", "weak"),
        ("sunny", "mild", "normal", "weak"),
        ("overcast", "hot", "high", "weak"),
        ("rainy", "mild", "high", "strong"),
        ("rainy", "cool", "normal", "weak"),
    ]
    
    print("Testing the decision tree:")
    print("-" * 60)
    for outlook, temp, humidity, wind in test_cases:
        result = should_play_tennis(outlook, temp, humidity, wind)
        print(f"  {outlook}, {temp}, {humidity} humidity, {wind} wind → {result}")


# ============================================
# Training and Testing Split
# ============================================

def train_test_split_demo():
    """
    Explain why we split data into training and testing
    """
    print("\n" + "=" * 60)
    print("Why Split Data into Training and Testing?")
    print("=" * 60)
    
    print("""
Imagine studying for a test:

❌ BAD: Study the exact test questions, then take the same test
   → You'll get 100%, but did you really learn?

✓ GOOD: Study from a textbook, then take a different test
   → This shows if you truly understood the material!

In Machine Learning:
- Training Data (70-80%): The "textbook" - model learns from this
- Testing Data (20-30%): The "test" - check if model really learned

Example with 100 data points:
    """)
    
    data = list(range(100))
    random.shuffle(data)
    
    train_size = 80
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    print(f"  Total data points: {len(data)}")
    print(f"  Training set: {len(train_data)} points ({len(train_data)}%)")
    print(f"  Testing set: {len(test_data)} points ({len(test_data)}%)")
    print(f"\n  Training indices (first 10): {sorted(train_data[:10])}")
    print(f"  Testing indices: {sorted(test_data)}")


# ============================================
# Overfitting vs Underfitting
# ============================================

def overfitting_demo():
    """
    Explain overfitting and underfitting
    """
    print("\n" + "=" * 60)
    print("Overfitting vs Underfitting")
    print("=" * 60)
    
    print("""
🎯 The Goal: Find the right balance!

📉 UNDERFITTING (Too Simple)
   - Model is too basic
   - Doesn't capture patterns
   - Like memorizing "all animals have 4 legs"
   - Bad on training AND testing data
   
📈 OVERFITTING (Too Complex)  
   - Model memorizes training data
   - Doesn't generalize to new data
   - Like memorizing every animal you've seen
   - Great on training, BAD on testing
   
✅ JUST RIGHT (Good Fit)
   - Model captures real patterns
   - Works well on new data
   - Like learning "mammals have 4 legs, birds have 2"
   - Good on both training AND testing

How to Detect:
   Training Accuracy | Testing Accuracy | Problem
   ------------------|------------------|----------
        Low          |       Low        | Underfitting
        High         |       Low        | Overfitting
        High         |       High       | Good fit! ✓
    """)


# ============================================
# Gradient Descent - How Models Learn
# ============================================

def gradient_descent_demo():
    """
    Visualize how gradient descent works
    """
    print("\n" + "=" * 60)
    print("Gradient Descent: How Models Learn")
    print("=" * 60)
    
    print("""
Imagine you're blindfolded on a hill and want to reach the bottom:

1. Feel which direction goes down (calculate gradient)
2. Take a step in that direction (update weights)
3. Repeat until you reach the bottom (minimum error)

The "Learning Rate" controls step size:
- Too big: You might overshoot and miss the bottom
- Too small: Takes forever to reach the bottom
- Just right: Efficiently reaches the minimum
    """)
    
    # Simple gradient descent example
    # Find x that minimizes f(x) = (x - 5)^2
    
    print("\nDemo: Finding x that minimizes (x - 5)²")
    print("-" * 40)
    
    x = 0  # Starting point
    learning_rate = 0.1
    
    print(f"Starting at x = {x}")
    print(f"Learning rate = {learning_rate}")
    print()
    
    for i in range(10):
        # Calculate gradient: d/dx (x-5)² = 2(x-5)
        gradient = 2 * (x - 5)
        
        # Update x
        x = x - learning_rate * gradient
        
        # Calculate error
        error = (x - 5) ** 2
        
        print(f"Step {i+1}: x = {x:.4f}, error = {error:.4f}")
    
    print(f"\nFinal answer: x ≈ {x:.2f} (optimal is 5)")


# ============================================
# Main Program
# ============================================

# =============================================================================
# SCIKIT-LEARN EXAMPLES - Real ML Library
# =============================================================================

def sklearn_demo():
    """Demonstrate real ML with scikit-learn"""
    print("\n" + "=" * 60)
    print("Scikit-Learn Demo: Real Machine Learning Library")
    print("=" * 60)
    
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        import numpy as np
        
        print("\n1. Linear Regression with sklearn:")
        print("-" * 40)
        X = np.array([[1000], [1500], [2000], [2500], [3000]])
        y = np.array([150, 200, 250, 300, 350])
        
        model = LinearRegression()
        model.fit(X, y)
        
        prediction = model.predict([[2200]])
        print(f"   House 2200 sq ft predicted price: ${prediction[0]:.0f}k")
        print(f"   Model coefficient: {model.coef_[0]:.4f}")
        
        print("\n2. KNN Classification with sklearn:")
        print("-" * 40)
        X = np.array([[150, 8], [160, 7], [180, 3], [190, 4], [120, 10]])
        y = np.array(["Apple", "Apple", "Orange", "Orange", "Grape"])
        
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X, y)
        
        test = np.array([[155, 7]])
        print(f"   Fruit [155g, sweetness 7] predicted: {knn.predict(test)[0]}")
        
        print("\n3. Decision Tree with sklearn:")
        print("-" * 40)
        X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])  # [sunny, humid]
        y = np.array(["No", "Yes", "Yes", "Yes"])  # play tennis
        
        tree = DecisionTreeClassifier()
        tree.fit(X, y)
        
        print(f"   Sunny + Humid: {tree.predict([[1, 1]])[0]}")
        print(f"   Sunny + Not Humid: {tree.predict([[1, 0]])[0]}")
        
    except ImportError:
        print("\nInstall scikit-learn: pip install scikit-learn numpy")
        print("Then run this demo again!")


def main():
    """
    Main function to run all demos
    """
    print("\n" + "=" * 60)
    print("   MODULE 2: MACHINE LEARNING BASICS")
    print("   360-Degree Coverage with GenAI Integration")
    print("=" * 60)
    
    while True:
        print("\n" + "-" * 60)
        print("Choose a demo:")
        print("1. What is Machine Learning?")
        print("2. Linear Regression (Predicting Numbers)")
        print("3. K-Nearest Neighbors (Classification)")
        print("4. Decision Trees (Rules)")
        print("5. Train/Test Split")
        print("6. Overfitting vs Underfitting")
        print("7. Gradient Descent (How Models Learn)")
        print("8. Scikit-Learn Demo (Real ML Library)")
        print("9. Run All Demos")
        print("0. Exit")
        print("-" * 60)
        
        choice = input("\nEnter your choice (0-9): ").strip()
        
        if choice == "0":
            print("\nThanks for learning about ML! Goodbye!")
            break
        elif choice == "1":
            explain_machine_learning()
        elif choice == "2":
            linear_regression_demo()
        elif choice == "3":
            knn_demo()
        elif choice == "4":
            decision_tree_demo()
        elif choice == "5":
            train_test_split_demo()
        elif choice == "6":
            overfitting_demo()
        elif choice == "7":
            gradient_descent_demo()
        elif choice == "8":
            sklearn_demo()
        elif choice == "9":
            explain_machine_learning()
            linear_regression_demo()
            knn_demo()
            decision_tree_demo()
            sklearn_demo()
        else:
            print("Invalid choice. Please enter 0-9.")


if __name__ == "__main__":
    main()
