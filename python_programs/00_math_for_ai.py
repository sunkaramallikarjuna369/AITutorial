"""
=============================================================================
MODULE 0: MATHEMATICS FOR AI - 360 DEGREE COVERAGE
=============================================================================

This comprehensive module covers ALL essential math concepts for AI/ML with:
- 4W+H Explanations (What, Why, When, Where, How)
- Visual demonstrations
- Python implementations
- Real-world AI/ML applications

TOPICS COVERED:
---------------
1. LINEAR ALGEBRA: Vectors, Matrices, Dot Product, Matrix Multiplication, Eigenvalues
2. CALCULUS: Derivatives, Gradients, Chain Rule, Partial Derivatives
3. PROBABILITY: Bayes Theorem, Distributions, Conditional Probability
4. STATISTICS: Mean, Variance, Standard Deviation, Correlation
5. OPTIMIZATION: Gradient Descent, Loss Functions, Convex Optimization

SETUP:
------
pip install numpy matplotlib scipy

=============================================================================
"""

import math
from typing import List, Tuple

# Try to import numpy for advanced operations
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Note: Install numpy for advanced features: pip install numpy")


# =============================================================================
# SECTION 1: LINEAR ALGEBRA
# =============================================================================

class LinearAlgebra:
    """
    Linear Algebra - The Language of AI
    ====================================
    
    WHAT: Branch of mathematics dealing with vectors, matrices, and linear transformations
    WHY: Neural networks are essentially matrix operations! Every layer is matrix multiplication
    WHEN: Used in EVERY AI model - from simple regression to transformers
    WHERE: Neural networks, embeddings, PCA, SVD, attention mechanisms
    HOW: Through matrix operations (addition, multiplication, transpose, inverse)
    """
    
    @staticmethod
    def vectors_explained():
        """
        VECTORS - The Building Blocks
        ==============================
        
        WHAT: An ordered list of numbers representing a point or direction in space
        WHY: Data is represented as vectors! Images, text, audio - all become vectors
        WHEN: Always - every piece of data in ML is a vector
        WHERE: Word embeddings, image pixels, feature vectors
        HOW: As arrays/lists of numbers
        """
        print("=" * 70)
        print("VECTORS - The Building Blocks of AI")
        print("=" * 70)
        
        print("""
WHAT IS A VECTOR?
-----------------
A vector is an ordered list of numbers. Think of it as coordinates!

    2D Vector: [3, 4]     -> A point at x=3, y=4
    3D Vector: [1, 2, 3]  -> A point in 3D space
    
In AI, vectors can have thousands of dimensions!
    Word "king" might be: [0.2, -0.5, 0.8, 0.1, ...]  (300 dimensions)

WHY VECTORS IN AI?
------------------
Everything becomes a vector:
    - Image (28x28 pixels) -> Vector of 784 numbers
    - Word "hello"         -> Vector of 300 numbers (word embedding)
    - Audio clip           -> Vector of frequency values
    
WHEN ARE VECTORS USED?
----------------------
    - Input data representation
    - Word embeddings (Word2Vec, GloVe)
    - Image features
    - Hidden states in neural networks

WHERE IN AI?
------------
    - Every neural network layer
    - Attention mechanisms (Query, Key, Value vectors)
    - Recommendation systems (user/item vectors)
    - Similarity search (cosine similarity)
        """)
        
        # Demo: Vector operations
        print("\nVECTOR OPERATIONS DEMO:")
        print("-" * 40)
        
        v1 = [1, 2, 3]
        v2 = [4, 5, 6]
        
        # Addition
        v_add = [v1[i] + v2[i] for i in range(len(v1))]
        print(f"Vector Addition: {v1} + {v2} = {v_add}")
        
        # Scalar multiplication
        scalar = 2
        v_scaled = [scalar * x for x in v1]
        print(f"Scalar Multiplication: {scalar} * {v1} = {v_scaled}")
        
        # Dot product
        dot = sum(v1[i] * v2[i] for i in range(len(v1)))
        print(f"Dot Product: {v1} . {v2} = {dot}")
        
        # Magnitude (length)
        magnitude = math.sqrt(sum(x**2 for x in v1))
        print(f"Magnitude of {v1} = {magnitude:.4f}")
        
        print("""
HOW TO USE IN AI:
-----------------
# Word similarity using cosine similarity
def cosine_similarity(v1, v2):
    dot = sum(a*b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a**2 for a in v1))
    mag2 = math.sqrt(sum(b**2 for b in v2))
    return dot / (mag1 * mag2)

# Similar words have similar vectors!
# cosine_similarity(king, queen) ≈ 0.8
# cosine_similarity(king, banana) ≈ 0.1
        """)
    
    @staticmethod
    def matrices_explained():
        """
        MATRICES - The Heart of Neural Networks
        ========================================
        """
        print("\n" + "=" * 70)
        print("MATRICES - The Heart of Neural Networks")
        print("=" * 70)
        
        print("""
WHAT IS A MATRIX?
-----------------
A matrix is a 2D array of numbers arranged in rows and columns.

    A = | 1  2  3 |
        | 4  5  6 |
        
    This is a 2x3 matrix (2 rows, 3 columns)

WHY MATRICES IN AI?
-------------------
Neural network layers ARE matrix operations!

    Input (vector)  x  Weights (matrix)  =  Output (vector)
    [1, 2, 3]       x  | w11 w12 |       =  [y1, y2]
                       | w21 w22 |
                       | w31 w32 |

WHEN ARE MATRICES USED?
-----------------------
    - Weight matrices in neural networks
    - Attention scores in transformers
    - Convolution kernels in CNNs
    - Covariance matrices in statistics

WHERE IN AI?
------------
    - Every layer: output = activation(input @ weights + bias)
    - Transformers: Attention = softmax(Q @ K.T / sqrt(d)) @ V
    - CNNs: Feature maps from convolution operations
        """)
        
        print("\nMATRIX OPERATIONS DEMO:")
        print("-" * 40)
        
        # Simple matrix operations without numpy
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        
        # Matrix addition
        C = [[A[i][j] + B[i][j] for j in range(2)] for i in range(2)]
        print(f"Matrix A: {A}")
        print(f"Matrix B: {B}")
        print(f"A + B = {C}")
        
        # Matrix multiplication
        def matrix_multiply(A, B):
            rows_A, cols_A = len(A), len(A[0])
            rows_B, cols_B = len(B), len(B[0])
            result = [[0] * cols_B for _ in range(rows_A)]
            for i in range(rows_A):
                for j in range(cols_B):
                    for k in range(cols_A):
                        result[i][j] += A[i][k] * B[k][j]
            return result
        
        C = matrix_multiply(A, B)
        print(f"A @ B = {C}")
        
        print("""
HOW NEURAL NETWORKS USE MATRICES:
---------------------------------
# A simple neural network layer
def neural_layer(input_vector, weight_matrix, bias):
    # Matrix multiplication + bias
    output = []
    for i in range(len(weight_matrix[0])):
        sum_val = bias[i]
        for j in range(len(input_vector)):
            sum_val += input_vector[j] * weight_matrix[j][i]
        output.append(sum_val)
    return output

# With numpy (much faster!):
# output = np.dot(input_vector, weight_matrix) + bias
        """)
    
    @staticmethod
    def dot_product_explained():
        """
        DOT PRODUCT - Measuring Similarity
        ===================================
        """
        print("\n" + "=" * 70)
        print("DOT PRODUCT - Measuring Similarity")
        print("=" * 70)
        
        print("""
WHAT IS DOT PRODUCT?
--------------------
The dot product multiplies corresponding elements and sums them up.

    a = [1, 2, 3]
    b = [4, 5, 6]
    a . b = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32

WHY DOT PRODUCT IN AI?
----------------------
1. SIMILARITY: Measures how similar two vectors are
   - High dot product = similar direction
   - Zero dot product = perpendicular (unrelated)
   - Negative dot product = opposite directions

2. NEURAL NETWORKS: Every neuron computes a dot product!
   output = dot(inputs, weights) + bias

WHEN IS DOT PRODUCT USED?
-------------------------
    - Attention mechanisms (Query . Key)
    - Similarity search
    - Every neuron in every layer
    - Recommendation systems

WHERE IN AI?
------------
    - Transformers: attention_score = Q . K
    - Word2Vec: word similarity
    - Neural networks: weighted sum of inputs
        """)
        
        # Demo
        print("\nDOT PRODUCT DEMO:")
        print("-" * 40)
        
        # Similarity example
        king = [0.5, 0.8, 0.2]
        queen = [0.4, 0.9, 0.3]
        banana = [-0.2, 0.1, 0.9]
        
        def dot_product(a, b):
            return sum(x*y for x, y in zip(a, b))
        
        def cosine_sim(a, b):
            dot = dot_product(a, b)
            mag_a = math.sqrt(sum(x**2 for x in a))
            mag_b = math.sqrt(sum(x**2 for x in b))
            return dot / (mag_a * mag_b)
        
        print(f"king vector:   {king}")
        print(f"queen vector:  {queen}")
        print(f"banana vector: {banana}")
        print()
        print(f"Similarity(king, queen):  {cosine_sim(king, queen):.4f} (HIGH - related!)")
        print(f"Similarity(king, banana): {cosine_sim(king, banana):.4f} (LOW - unrelated)")


# =============================================================================
# SECTION 2: CALCULUS
# =============================================================================

class Calculus:
    """
    Calculus - How AI Learns
    =========================
    
    WHAT: Branch of mathematics studying rates of change (derivatives) and accumulation (integrals)
    WHY: Derivatives tell us how to adjust weights to reduce error - this IS learning!
    WHEN: During training - backpropagation uses chain rule to compute gradients
    WHERE: Every training step of every neural network
    HOW: Through automatic differentiation (PyTorch, TensorFlow do this for you)
    """
    
    @staticmethod
    def derivatives_explained():
        """
        DERIVATIVES - The Direction of Improvement
        ===========================================
        """
        print("\n" + "=" * 70)
        print("DERIVATIVES - The Direction of Improvement")
        print("=" * 70)
        
        print("""
WHAT IS A DERIVATIVE?
---------------------
The derivative measures how much a function changes when its input changes.

    f(x) = x^2
    f'(x) = 2x  (derivative)
    
    At x=3: f'(3) = 6
    This means: if x increases by 1, f(x) increases by about 6

WHY DERIVATIVES IN AI?
----------------------
Derivatives tell us HOW TO IMPROVE!

    Loss = 10 (bad prediction)
    d(Loss)/d(weight) = -2
    
    This means: if we INCREASE the weight, the loss will DECREASE!
    So we should increase the weight to make better predictions.

WHEN ARE DERIVATIVES USED?
--------------------------
    - Every training step (backpropagation)
    - Computing gradients
    - Optimization (finding minimum loss)

WHERE IN AI?
------------
    - Gradient descent: weight = weight - learning_rate * gradient
    - Backpropagation: computing gradients layer by layer
    - All neural network training
        """)
        
        print("\nDERIVATIVE DEMO:")
        print("-" * 40)
        
        # Numerical derivative
        def f(x):
            return x ** 2
        
        def derivative(f, x, h=0.0001):
            return (f(x + h) - f(x)) / h
        
        x = 3
        print(f"f(x) = x^2")
        print(f"At x = {x}:")
        print(f"  f({x}) = {f(x)}")
        print(f"  f'({x}) = {derivative(f, x):.4f} (should be 2*{x} = {2*x})")
        
        print("""
HOW DERIVATIVES ENABLE LEARNING:
--------------------------------
# Simple gradient descent
def train_step(weight, learning_rate=0.01):
    # 1. Make prediction
    prediction = weight * input_data
    
    # 2. Calculate loss
    loss = (prediction - target) ** 2
    
    # 3. Calculate gradient (derivative of loss w.r.t. weight)
    gradient = 2 * (prediction - target) * input_data
    
    # 4. Update weight in opposite direction of gradient
    weight = weight - learning_rate * gradient
    
    return weight  # Now makes better predictions!
        """)
    
    @staticmethod
    def gradients_explained():
        """
        GRADIENTS - Multi-dimensional Derivatives
        ==========================================
        """
        print("\n" + "=" * 70)
        print("GRADIENTS - Multi-dimensional Derivatives")
        print("=" * 70)
        
        print("""
WHAT IS A GRADIENT?
-------------------
A gradient is a vector of partial derivatives - one for each input variable.

    f(x, y) = x^2 + y^2
    gradient = [df/dx, df/dy] = [2x, 2y]
    
    At (3, 4): gradient = [6, 8]
    This points in the direction of STEEPEST INCREASE

WHY GRADIENTS IN AI?
--------------------
Neural networks have MILLIONS of weights!
The gradient tells us how to adjust ALL of them at once.

    weights = [w1, w2, w3, ..., w1000000]
    gradient = [dL/dw1, dL/dw2, ..., dL/dw1000000]
    
    Each gradient component tells us how to adjust that specific weight.

WHEN ARE GRADIENTS USED?
------------------------
    - Every training iteration
    - Backpropagation computes gradients for all weights
    - Optimizer uses gradients to update weights

WHERE IN AI?
------------
    - loss.backward() in PyTorch computes all gradients
    - optimizer.step() uses gradients to update weights
    - Every deep learning framework
        """)
        
        print("\nGRADIENT DESCENT VISUALIZATION:")
        print("-" * 40)
        
        # Simple 2D gradient descent
        def f(x, y):
            return x**2 + y**2  # Bowl-shaped function, minimum at (0, 0)
        
        def gradient(x, y):
            return [2*x, 2*y]
        
        # Start at a random point
        x, y = 5.0, 5.0
        learning_rate = 0.1
        
        print(f"Finding minimum of f(x,y) = x^2 + y^2")
        print(f"Starting at ({x}, {y}), f = {f(x, y)}")
        print()
        
        for i in range(10):
            grad = gradient(x, y)
            x = x - learning_rate * grad[0]
            y = y - learning_rate * grad[1]
            print(f"Step {i+1}: ({x:.4f}, {y:.4f}), f = {f(x, y):.4f}")
        
        print(f"\nConverged to minimum at approximately (0, 0)!")
    
    @staticmethod
    def chain_rule_explained():
        """
        CHAIN RULE - The Secret of Backpropagation
        ===========================================
        """
        print("\n" + "=" * 70)
        print("CHAIN RULE - The Secret of Backpropagation")
        print("=" * 70)
        
        print("""
WHAT IS THE CHAIN RULE?
-----------------------
When functions are composed, multiply their derivatives!

    y = f(g(x))
    dy/dx = f'(g(x)) * g'(x)

Example:
    y = (2x + 1)^2
    Let u = 2x + 1, so y = u^2
    
    dy/du = 2u
    du/dx = 2
    dy/dx = dy/du * du/dx = 2u * 2 = 4(2x + 1)

WHY CHAIN RULE IN AI?
---------------------
Neural networks are COMPOSED functions!

    output = activation(layer3(layer2(layer1(input))))
    
To find how input affects output, we multiply derivatives through each layer.
This IS backpropagation!

WHEN IS CHAIN RULE USED?
------------------------
    - Every backpropagation step
    - Computing gradients through multiple layers
    - Any composed function

WHERE IN AI?
------------
    - Deep networks: gradient flows backward through all layers
    - RNNs: gradient flows through time steps
    - Transformers: gradient flows through attention layers
        """)
        
        print("\nCHAIN RULE IN NEURAL NETWORKS:")
        print("-" * 40)
        print("""
Forward pass:
    x -> [Layer 1] -> h1 -> [Layer 2] -> h2 -> [Layer 3] -> output -> [Loss]
    
Backward pass (chain rule):
    dL/dW3 = dL/doutput * doutput/dW3
    dL/dW2 = dL/doutput * doutput/dh2 * dh2/dW2
    dL/dW1 = dL/doutput * doutput/dh2 * dh2/dh1 * dh1/dW1
    
Each layer multiplies by its local gradient!
        """)


# =============================================================================
# SECTION 3: PROBABILITY
# =============================================================================

class Probability:
    """
    Probability - Handling Uncertainty
    ===================================
    
    WHAT: Mathematics of uncertainty and likelihood
    WHY: AI deals with uncertain predictions - probability quantifies confidence
    WHEN: Classification (probability of each class), language models (probability of next word)
    WHERE: Softmax outputs, Bayesian methods, generative models
    HOW: Through probability distributions and Bayes' theorem
    """
    
    @staticmethod
    def basics_explained():
        """
        PROBABILITY BASICS
        ==================
        """
        print("\n" + "=" * 70)
        print("PROBABILITY - Handling Uncertainty in AI")
        print("=" * 70)
        
        print("""
WHAT IS PROBABILITY?
--------------------
Probability measures how likely something is to happen.
    
    P(event) = Number of favorable outcomes / Total outcomes
    
    P(heads) = 1/2 = 0.5 = 50%
    P(rolling 6) = 1/6 ≈ 0.167 = 16.7%

WHY PROBABILITY IN AI?
----------------------
AI models output PROBABILITIES, not certainties!

    Image classifier output:
        P(cat) = 0.85  (85% confident it's a cat)
        P(dog) = 0.10
        P(bird) = 0.05
    
    Language model:
        P(next word = "the") = 0.3
        P(next word = "a") = 0.2
        ...

WHEN IS PROBABILITY USED?
-------------------------
    - Classification: softmax gives class probabilities
    - Language models: probability of next token
    - Generative models: sampling from distributions
    - Uncertainty estimation

WHERE IN AI?
------------
    - Softmax layer: converts scores to probabilities
    - Cross-entropy loss: compares predicted vs true probabilities
    - Bayesian neural networks: uncertainty quantification
    - Diffusion models: probabilistic denoising
        """)
        
        print("\nSOFTMAX DEMO (Converting scores to probabilities):")
        print("-" * 40)
        
        def softmax(scores):
            exp_scores = [math.exp(s) for s in scores]
            total = sum(exp_scores)
            return [e / total for e in exp_scores]
        
        # Raw neural network outputs (logits)
        logits = [2.0, 1.0, 0.1]
        classes = ["cat", "dog", "bird"]
        
        probs = softmax(logits)
        
        print(f"Raw scores (logits): {logits}")
        print(f"After softmax (probabilities):")
        for cls, prob in zip(classes, probs):
            print(f"  P({cls}) = {prob:.4f} ({prob*100:.1f}%)")
        print(f"Sum of probabilities: {sum(probs):.4f} (always = 1.0)")
    
    @staticmethod
    def bayes_theorem_explained():
        """
        BAYES' THEOREM - Updating Beliefs with Evidence
        ================================================
        """
        print("\n" + "=" * 70)
        print("BAYES' THEOREM - Updating Beliefs with Evidence")
        print("=" * 70)
        
        print("""
WHAT IS BAYES' THEOREM?
-----------------------
Bayes' theorem tells us how to update our beliefs when we get new evidence.

    P(A|B) = P(B|A) * P(A) / P(B)
    
    P(A|B) = Probability of A given B (posterior)
    P(B|A) = Probability of B given A (likelihood)
    P(A) = Prior probability of A
    P(B) = Probability of B (evidence)

WHY BAYES IN AI?
----------------
1. SPAM FILTERING: P(spam | contains "free money")
2. MEDICAL DIAGNOSIS: P(disease | symptoms)
3. RECOMMENDATION: P(user likes movie | past behavior)

WHEN IS BAYES USED?
-------------------
    - Naive Bayes classifiers
    - Bayesian optimization (hyperparameter tuning)
    - Probabilistic programming
    - Uncertainty estimation

WHERE IN AI?
------------
    - Spam filters
    - Medical diagnosis systems
    - A/B testing analysis
    - Bayesian neural networks
        """)
        
        print("\nBAYES' THEOREM EXAMPLE - Medical Test:")
        print("-" * 40)
        
        # Disease testing example
        p_disease = 0.01  # 1% of population has disease
        p_positive_given_disease = 0.99  # Test is 99% accurate for sick people
        p_positive_given_healthy = 0.05  # 5% false positive rate
        
        # P(positive) = P(pos|disease)*P(disease) + P(pos|healthy)*P(healthy)
        p_positive = (p_positive_given_disease * p_disease + 
                      p_positive_given_healthy * (1 - p_disease))
        
        # Bayes' theorem: P(disease|positive)
        p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive
        
        print(f"Prior probability of disease: {p_disease*100:.1f}%")
        print(f"Test accuracy (true positive): {p_positive_given_disease*100:.1f}%")
        print(f"False positive rate: {p_positive_given_healthy*100:.1f}%")
        print()
        print(f"If test is POSITIVE:")
        print(f"  P(actually have disease) = {p_disease_given_positive*100:.1f}%")
        print()
        print("Surprising! Even with a 99% accurate test, a positive result")
        print("only means ~17% chance of disease (because disease is rare).")


# =============================================================================
# SECTION 4: STATISTICS
# =============================================================================

class Statistics:
    """
    Statistics - Understanding Data
    ================================
    
    WHAT: Mathematics of collecting, analyzing, and interpreting data
    WHY: AI learns from data - we need to understand data properties
    WHEN: Data preprocessing, model evaluation, feature engineering
    WHERE: Normalization, batch statistics, evaluation metrics
    HOW: Through measures of central tendency, spread, and relationships
    """
    
    @staticmethod
    def basics_explained():
        """
        STATISTICAL BASICS
        ==================
        """
        print("\n" + "=" * 70)
        print("STATISTICS - Understanding Your Data")
        print("=" * 70)
        
        print("""
KEY STATISTICAL MEASURES:
-------------------------

1. MEAN (Average)
   WHAT: Sum of values divided by count
   WHY: Center of your data
   WHERE: Batch normalization, loss averaging
   
2. VARIANCE
   WHAT: Average squared distance from mean
   WHY: Measures spread/variability
   WHERE: Normalization, detecting outliers
   
3. STANDARD DEVIATION
   WHAT: Square root of variance
   WHY: Spread in original units
   WHERE: Z-score normalization, confidence intervals
   
4. CORRELATION
   WHAT: Measure of linear relationship (-1 to 1)
   WHY: Find related features
   WHERE: Feature selection, multicollinearity detection
        """)
        
        print("\nSTATISTICS DEMO:")
        print("-" * 40)
        
        data = [2, 4, 4, 4, 5, 5, 7, 9]
        
        # Mean
        mean = sum(data) / len(data)
        
        # Variance
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        
        # Standard deviation
        std_dev = math.sqrt(variance)
        
        # Min, Max
        min_val, max_val = min(data), max(data)
        
        print(f"Data: {data}")
        print(f"Mean: {mean}")
        print(f"Variance: {variance}")
        print(f"Standard Deviation: {std_dev:.4f}")
        print(f"Min: {min_val}, Max: {max_val}")
        
        print("""
WHY STATISTICS MATTER IN AI:
----------------------------
1. NORMALIZATION: Scale features to mean=0, std=1
   normalized = (x - mean) / std
   
2. BATCH NORMALIZATION: Normalize activations in neural networks
   Helps training converge faster!
   
3. EVALUATION: Mean accuracy, standard deviation of results
   "Model achieves 95% +/- 2% accuracy"
        """)
    
    @staticmethod
    def normalization_explained():
        """
        NORMALIZATION - Preparing Data for AI
        ======================================
        """
        print("\n" + "=" * 70)
        print("NORMALIZATION - Preparing Data for AI")
        print("=" * 70)
        
        print("""
WHAT IS NORMALIZATION?
----------------------
Scaling data to a standard range or distribution.

Types:
1. MIN-MAX NORMALIZATION: Scale to [0, 1]
   x_norm = (x - min) / (max - min)
   
2. Z-SCORE NORMALIZATION: Scale to mean=0, std=1
   x_norm = (x - mean) / std
   
3. BATCH NORMALIZATION: Normalize within mini-batches

WHY NORMALIZE?
--------------
1. Features on same scale -> faster training
2. Prevents large values from dominating
3. Helps gradient descent converge
4. Required for many algorithms (SVM, KNN)

WHEN TO NORMALIZE?
------------------
    - Before training neural networks
    - When features have different scales
    - For distance-based algorithms
        """)
        
        print("\nNORMALIZATION DEMO:")
        print("-" * 40)
        
        # Original data with different scales
        ages = [25, 30, 35, 40, 45]  # Range: 25-45
        salaries = [30000, 50000, 70000, 90000, 110000]  # Range: 30k-110k
        
        print("Original data:")
        print(f"  Ages: {ages}")
        print(f"  Salaries: {salaries}")
        
        # Min-max normalization
        def min_max_normalize(data):
            min_val, max_val = min(data), max(data)
            return [(x - min_val) / (max_val - min_val) for x in data]
        
        # Z-score normalization
        def z_score_normalize(data):
            mean = sum(data) / len(data)
            std = math.sqrt(sum((x - mean) ** 2 for x in data) / len(data))
            return [(x - mean) / std for x in data]
        
        print("\nMin-Max Normalized (0 to 1):")
        print(f"  Ages: {[f'{x:.2f}' for x in min_max_normalize(ages)]}")
        print(f"  Salaries: {[f'{x:.2f}' for x in min_max_normalize(salaries)]}")
        
        print("\nZ-Score Normalized (mean=0, std=1):")
        print(f"  Ages: {[f'{x:.2f}' for x in z_score_normalize(ages)]}")
        print(f"  Salaries: {[f'{x:.2f}' for x in z_score_normalize(salaries)]}")


# =============================================================================
# SECTION 5: OPTIMIZATION
# =============================================================================

class Optimization:
    """
    Optimization - Finding the Best Solution
    =========================================
    
    WHAT: Finding the best parameters that minimize (or maximize) a function
    WHY: Training = finding weights that minimize loss
    WHEN: Every training step
    WHERE: All machine learning algorithms
    HOW: Gradient descent and its variants
    """
    
    @staticmethod
    def gradient_descent_explained():
        """
        GRADIENT DESCENT - The Learning Algorithm
        ==========================================
        """
        print("\n" + "=" * 70)
        print("GRADIENT DESCENT - How AI Learns")
        print("=" * 70)
        
        print("""
WHAT IS GRADIENT DESCENT?
-------------------------
An algorithm to find the minimum of a function by following the gradient downhill.

    1. Start at random point
    2. Calculate gradient (direction of steepest increase)
    3. Move in OPPOSITE direction (to decrease)
    4. Repeat until converged

    new_weight = old_weight - learning_rate * gradient

WHY GRADIENT DESCENT?
---------------------
It's how neural networks LEARN!
    - Loss function measures how wrong predictions are
    - Gradient tells us how to adjust weights
    - We adjust weights to reduce loss
    - Repeat millions of times = trained model!

VARIANTS:
---------
1. BATCH GD: Use all data each step (slow but stable)
2. STOCHASTIC GD: Use one sample (fast but noisy)
3. MINI-BATCH GD: Use small batches (best of both)
4. ADAM: Adaptive learning rates (most popular)
        """)
        
        print("\nGRADIENT DESCENT DEMO:")
        print("-" * 40)
        
        # Find minimum of f(x) = (x - 3)^2
        # Minimum is at x = 3
        
        def f(x):
            return (x - 3) ** 2
        
        def gradient(x):
            return 2 * (x - 3)
        
        x = 10.0  # Start far from minimum
        learning_rate = 0.1
        
        print(f"Finding minimum of f(x) = (x - 3)^2")
        print(f"True minimum at x = 3")
        print(f"Starting at x = {x}")
        print()
        
        for i in range(15):
            grad = gradient(x)
            x = x - learning_rate * grad
            print(f"Step {i+1:2d}: x = {x:.6f}, f(x) = {f(x):.6f}, gradient = {grad:.6f}")
            
            if abs(grad) < 0.0001:
                print(f"\nConverged! Found minimum at x ≈ {x:.4f}")
                break
    
    @staticmethod
    def loss_functions_explained():
        """
        LOSS FUNCTIONS - Measuring Errors
        ==================================
        """
        print("\n" + "=" * 70)
        print("LOSS FUNCTIONS - Measuring How Wrong We Are")
        print("=" * 70)
        
        print("""
WHAT IS A LOSS FUNCTION?
------------------------
A function that measures how wrong our predictions are.
Lower loss = better predictions!

COMMON LOSS FUNCTIONS:
----------------------

1. MEAN SQUARED ERROR (MSE) - For regression
   WHAT: Average of squared differences
   WHY: Penalizes large errors more
   WHERE: Predicting continuous values (prices, temperatures)
   
   MSE = (1/n) * sum((predicted - actual)^2)

2. CROSS-ENTROPY LOSS - For classification
   WHAT: Measures difference between probability distributions
   WHY: Works well with softmax outputs
   WHERE: Classification tasks (cat vs dog, spam detection)
   
   CE = -sum(actual * log(predicted))

3. BINARY CROSS-ENTROPY - For binary classification
   WHAT: Cross-entropy for two classes
   WHERE: Yes/No predictions (spam/not spam)
        """)
        
        print("\nLOSS FUNCTION DEMO:")
        print("-" * 40)
        
        # MSE example
        actual = [3, 5, 2, 7]
        predicted = [2.5, 5.2, 2.1, 6.8]
        
        mse = sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual)
        
        print("Mean Squared Error (Regression):")
        print(f"  Actual:    {actual}")
        print(f"  Predicted: {predicted}")
        print(f"  MSE = {mse:.4f}")
        
        # Cross-entropy example
        print("\nCross-Entropy Loss (Classification):")
        true_class = [1, 0, 0]  # True label: class 0
        predicted_probs = [0.7, 0.2, 0.1]  # Model's prediction
        
        ce_loss = -sum(t * math.log(p + 1e-10) for t, p in zip(true_class, predicted_probs))
        
        print(f"  True class: {true_class}")
        print(f"  Predicted:  {predicted_probs}")
        print(f"  Cross-Entropy Loss = {ce_loss:.4f}")


# =============================================================================
# MAIN PROGRAM
# =============================================================================

def main():
    print("=" * 70)
    print("   MODULE 0: MATHEMATICS FOR AI - 360 DEGREE COVERAGE")
    print("   4W+H Explanations (What, Why, When, Where, How)")
    print("=" * 70)
    
    while True:
        print("\n" + "-" * 70)
        print("Choose a topic:")
        print()
        print("LINEAR ALGEBRA:")
        print("  1. Vectors (Building blocks of AI)")
        print("  2. Matrices (Heart of neural networks)")
        print("  3. Dot Product (Measuring similarity)")
        print()
        print("CALCULUS:")
        print("  4. Derivatives (Direction of improvement)")
        print("  5. Gradients (Multi-dimensional derivatives)")
        print("  6. Chain Rule (Secret of backpropagation)")
        print()
        print("PROBABILITY:")
        print("  7. Probability Basics (Handling uncertainty)")
        print("  8. Bayes' Theorem (Updating beliefs)")
        print()
        print("STATISTICS:")
        print("  9. Statistical Basics (Understanding data)")
        print("  10. Normalization (Preparing data)")
        print()
        print("OPTIMIZATION:")
        print("  11. Gradient Descent (How AI learns)")
        print("  12. Loss Functions (Measuring errors)")
        print()
        print("  13. Run ALL Topics")
        print("  0. Exit")
        print("-" * 70)
        
        choice = input("\nEnter choice (0-13): ").strip()
        
        if choice == "0":
            print("\nHappy learning!")
            break
        elif choice == "1":
            LinearAlgebra.vectors_explained()
        elif choice == "2":
            LinearAlgebra.matrices_explained()
        elif choice == "3":
            LinearAlgebra.dot_product_explained()
        elif choice == "4":
            Calculus.derivatives_explained()
        elif choice == "5":
            Calculus.gradients_explained()
        elif choice == "6":
            Calculus.chain_rule_explained()
        elif choice == "7":
            Probability.basics_explained()
        elif choice == "8":
            Probability.bayes_theorem_explained()
        elif choice == "9":
            Statistics.basics_explained()
        elif choice == "10":
            Statistics.normalization_explained()
        elif choice == "11":
            Optimization.gradient_descent_explained()
        elif choice == "12":
            Optimization.loss_functions_explained()
        elif choice == "13":
            # Run all
            LinearAlgebra.vectors_explained()
            LinearAlgebra.matrices_explained()
            LinearAlgebra.dot_product_explained()
            Calculus.derivatives_explained()
            Calculus.gradients_explained()
            Calculus.chain_rule_explained()
            Probability.basics_explained()
            Probability.bayes_theorem_explained()
            Statistics.basics_explained()
            Statistics.normalization_explained()
            Optimization.gradient_descent_explained()
            Optimization.loss_functions_explained()
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
