"""
Module 03: Neural Networks
==========================
Learn how artificial neurons work - inspired by the brain!

This program demonstrates neural network concepts with interactive examples.
"""

import math
import random

# ============================================
# What is a Neural Network?
# ============================================

def explain_neural_networks():
    """
    Explain neural networks in simple terms
    """
    print("=" * 60)
    print("What is a Neural Network?")
    print("=" * 60)
    
    print("""
A Neural Network is like a mini brain made of connected "neurons"!

Your Brain:
    - Has ~86 billion neurons
    - Neurons connect to each other
    - They send electrical signals
    - Learning = strengthening connections

Artificial Neural Network:
    - Has artificial neurons (math functions)
    - Connected by "weights" (numbers)
    - They pass numbers around
    - Learning = adjusting weights

Structure:
    Input Layer → Hidden Layers → Output Layer
    (receives)    (processes)     (answers)

Example: Recognizing a cat photo
    Input: Pixel values (784 numbers for 28x28 image)
    Hidden: Detect edges, shapes, features
    Output: "Cat" or "Not Cat"
    """)


# ============================================
# Single Neuron (Perceptron)
# ============================================

class Neuron:
    """
    A single artificial neuron - the building block of neural networks!
    
    It does 3 things:
    1. Multiply inputs by weights
    2. Add them up (plus bias)
    3. Apply activation function
    """
    
    def __init__(self, num_inputs):
        # Initialize random weights between -1 and 1
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)
    
    def forward(self, inputs):
        """
        Calculate the neuron's output
        
        output = activation(sum(inputs * weights) + bias)
        """
        # Step 1: Weighted sum
        total = sum(x * w for x, w in zip(inputs, self.weights))
        total += self.bias
        
        # Step 2: Activation function (sigmoid)
        output = self.sigmoid(total)
        return output
    
    def sigmoid(self, x):
        """
        Sigmoid activation: squashes any number to 0-1
        
        This is like the neuron "firing" (1) or not (0)
        """
        return 1 / (1 + math.exp(-max(-500, min(500, x))))
    
    def __str__(self):
        return f"Neuron(weights={[round(w, 3) for w in self.weights]}, bias={round(self.bias, 3)})"


def single_neuron_demo():
    """
    Demo: A single neuron learning AND gate
    """
    print("\n" + "=" * 60)
    print("Single Neuron Demo: Learning AND Gate")
    print("=" * 60)
    
    # AND gate truth table
    training_data = [
        ([0, 0], 0),
        ([0, 1], 0),
        ([1, 0], 0),
        ([1, 1], 1),
    ]
    
    print("\nAND Gate Truth Table:")
    print("Input A | Input B | Output")
    for inputs, output in training_data:
        print(f"   {inputs[0]}   |    {inputs[1]}    |   {output}")
    
    # Create and train neuron
    neuron = Neuron(2)
    learning_rate = 0.5
    
    print(f"\nInitial neuron: {neuron}")
    print("\nTraining...")
    
    for epoch in range(1000):
        total_error = 0
        for inputs, expected in training_data:
            # Forward pass
            output = neuron.forward(inputs)
            
            # Calculate error
            error = expected - output
            total_error += abs(error)
            
            # Update weights (gradient descent)
            for i in range(len(neuron.weights)):
                neuron.weights[i] += learning_rate * error * inputs[i]
            neuron.bias += learning_rate * error
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}: Error = {total_error:.4f}")
    
    print(f"\nTrained neuron: {neuron}")
    print("\nTesting:")
    for inputs, expected in training_data:
        output = neuron.forward(inputs)
        print(f"  {inputs} → {output:.4f} (expected: {expected})")


# ============================================
# Multi-Layer Neural Network
# ============================================

class NeuralNetwork:
    """
    A simple neural network with one hidden layer
    
    Structure: Input → Hidden → Output
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights randomly
        self.weights_ih = [[random.uniform(-1, 1) for _ in range(input_size)] 
                          for _ in range(hidden_size)]
        self.bias_h = [random.uniform(-1, 1) for _ in range(hidden_size)]
        
        self.weights_ho = [[random.uniform(-1, 1) for _ in range(hidden_size)] 
                          for _ in range(output_size)]
        self.bias_o = [random.uniform(-1, 1) for _ in range(output_size)]
    
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-max(-500, min(500, x))))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, inputs):
        """Forward pass through the network"""
        # Input to Hidden
        self.hidden = []
        for i in range(self.hidden_size):
            total = sum(inputs[j] * self.weights_ih[i][j] for j in range(self.input_size))
            total += self.bias_h[i]
            self.hidden.append(self.sigmoid(total))
        
        # Hidden to Output
        self.output = []
        for i in range(self.output_size):
            total = sum(self.hidden[j] * self.weights_ho[i][j] for j in range(self.hidden_size))
            total += self.bias_o[i]
            self.output.append(self.sigmoid(total))
        
        return self.output
    
    def train(self, inputs, targets, learning_rate=0.5):
        """Train using backpropagation"""
        # Forward pass
        self.forward(inputs)
        
        # Calculate output errors
        output_errors = [targets[i] - self.output[i] for i in range(self.output_size)]
        
        # Calculate output gradients
        output_gradients = [output_errors[i] * self.sigmoid_derivative(self.output[i]) 
                          for i in range(self.output_size)]
        
        # Update hidden-output weights
        for i in range(self.output_size):
            for j in range(self.hidden_size):
                self.weights_ho[i][j] += learning_rate * output_gradients[i] * self.hidden[j]
            self.bias_o[i] += learning_rate * output_gradients[i]
        
        # Calculate hidden errors
        hidden_errors = [sum(output_errors[i] * self.weights_ho[i][j] 
                            for i in range(self.output_size)) 
                        for j in range(self.hidden_size)]
        
        # Calculate hidden gradients
        hidden_gradients = [hidden_errors[i] * self.sigmoid_derivative(self.hidden[i]) 
                          for i in range(self.hidden_size)]
        
        # Update input-hidden weights
        for i in range(self.hidden_size):
            for j in range(self.input_size):
                self.weights_ih[i][j] += learning_rate * hidden_gradients[i] * inputs[j]
            self.bias_h[i] += learning_rate * hidden_gradients[i]
        
        return sum(abs(e) for e in output_errors)


def xor_demo():
    """
    Demo: Neural network learning XOR (requires hidden layer!)
    """
    print("\n" + "=" * 60)
    print("Neural Network Demo: Learning XOR")
    print("=" * 60)
    
    print("""
XOR is special - a single neuron CAN'T learn it!
We need a hidden layer to solve this problem.
    """)
    
    # XOR truth table
    training_data = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0]),
    ]
    
    print("XOR Truth Table:")
    print("Input A | Input B | Output")
    for inputs, output in training_data:
        print(f"   {inputs[0]}   |    {inputs[1]}    |   {output[0]}")
    
    # Create network: 2 inputs, 4 hidden, 1 output
    nn = NeuralNetwork(2, 4, 1)
    
    print("\nTraining neural network...")
    
    for epoch in range(5000):
        total_error = 0
        for inputs, targets in training_data:
            error = nn.train(inputs, targets)
            total_error += error
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Error = {total_error:.4f}")
    
    print("\nTesting:")
    for inputs, expected in training_data:
        output = nn.forward(inputs)
        print(f"  {inputs} → {output[0]:.4f} (expected: {expected[0]})")


# ============================================
# Activation Functions
# ============================================

def activation_functions_demo():
    """
    Explain different activation functions
    """
    print("\n" + "=" * 60)
    print("Activation Functions")
    print("=" * 60)
    
    print("""
Activation functions decide if a neuron should "fire" or not.
They add non-linearity, allowing networks to learn complex patterns.
    """)
    
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
    
    def tanh(x):
        return math.tanh(x)
    
    def relu(x):
        return max(0, x)
    
    def leaky_relu(x, alpha=0.01):
        return x if x > 0 else alpha * x
    
    test_values = [-2, -1, -0.5, 0, 0.5, 1, 2]
    
    print("\nComparison of activation functions:")
    print("-" * 60)
    print(f"{'Input':>8} | {'Sigmoid':>8} | {'Tanh':>8} | {'ReLU':>8} | {'LeakyReLU':>10}")
    print("-" * 60)
    
    for x in test_values:
        print(f"{x:>8.1f} | {sigmoid(x):>8.4f} | {tanh(x):>8.4f} | {relu(x):>8.4f} | {leaky_relu(x):>10.4f}")
    
    print("""
When to use each:
- Sigmoid: Output layer for binary classification (0-1)
- Tanh: Hidden layers, outputs -1 to 1
- ReLU: Most popular for hidden layers, fast to compute
- LeakyReLU: Fixes "dying ReLU" problem
    """)


# ============================================
# Backpropagation Explained
# ============================================

def backpropagation_demo():
    """
    Explain backpropagation in simple terms
    """
    print("\n" + "=" * 60)
    print("Backpropagation: How Neural Networks Learn")
    print("=" * 60)
    
    print("""
Backpropagation is like learning from mistakes!

Imagine you're playing darts blindfolded:
1. Throw a dart (forward pass)
2. Friend tells you "too far left" (calculate error)
3. Adjust your throw (update weights)
4. Repeat until you hit the target!

The Algorithm:
1. FORWARD PASS: Send input through network, get output
2. CALCULATE ERROR: Compare output to expected answer
3. BACKWARD PASS: Send error backwards through network
4. UPDATE WEIGHTS: Adjust each weight based on its contribution to error

Key Insight: Chain Rule from Calculus
    - Error at output depends on hidden layer
    - Hidden layer depends on input layer
    - We can calculate how much each weight contributed to the error!

Visual:
    Input → [w1] → Hidden → [w2] → Output → Error
                                      ↓
    Input ← [Δw1] ← Hidden ← [Δw2] ← Gradient
    """)


# ============================================
# Network Architectures
# ============================================

def architectures_demo():
    """
    Explain different neural network architectures
    """
    print("\n" + "=" * 60)
    print("Neural Network Architectures")
    print("=" * 60)
    
    architectures = [
        {
            "name": "Feedforward (MLP)",
            "description": "Data flows one direction: input → output",
            "use_cases": ["Classification", "Regression", "Simple predictions"],
            "structure": "Input → Hidden → Hidden → Output"
        },
        {
            "name": "Convolutional (CNN)",
            "description": "Specialized for images, uses filters",
            "use_cases": ["Image recognition", "Object detection", "Face recognition"],
            "structure": "Input → Conv → Pool → Conv → Pool → Dense → Output"
        },
        {
            "name": "Recurrent (RNN)",
            "description": "Has memory, processes sequences",
            "use_cases": ["Text generation", "Speech recognition", "Time series"],
            "structure": "Input → RNN (loops back) → Output"
        },
        {
            "name": "Transformer",
            "description": "Uses attention, processes in parallel",
            "use_cases": ["ChatGPT", "Translation", "BERT"],
            "structure": "Input → Attention → FFN → Attention → Output"
        }
    ]
    
    for arch in architectures:
        print(f"\n📌 {arch['name']}")
        print(f"   {arch['description']}")
        print(f"   Structure: {arch['structure']}")
        print(f"   Use cases: {', '.join(arch['use_cases'])}")


# ============================================
# Main Program
# ============================================

def main():
    """
    Main function to run all demos
    """
    print("\n" + "🧠" * 25)
    print("\n   NEURAL NETWORKS")
    print("\n" + "🧠" * 25)
    
    while True:
        print("\n" + "-" * 60)
        print("Choose a demo:")
        print("1. What is a Neural Network?")
        print("2. Single Neuron (Perceptron)")
        print("3. Multi-Layer Network (XOR)")
        print("4. Activation Functions")
        print("5. Backpropagation Explained")
        print("6. Network Architectures")
        print("7. Run All Demos")
        print("0. Exit")
        print("-" * 60)
        
        choice = input("\nEnter your choice (0-7): ").strip()
        
        if choice == "0":
            print("\nThanks for learning about Neural Networks! Goodbye! 👋")
            break
        elif choice == "1":
            explain_neural_networks()
        elif choice == "2":
            single_neuron_demo()
        elif choice == "3":
            xor_demo()
        elif choice == "4":
            activation_functions_demo()
        elif choice == "5":
            backpropagation_demo()
        elif choice == "6":
            architectures_demo()
        elif choice == "7":
            explain_neural_networks()
            single_neuron_demo()
            xor_demo()
            activation_functions_demo()
            backpropagation_demo()
            architectures_demo()
        else:
            print("Invalid choice. Please enter 0-7.")


if __name__ == "__main__":
    main()
