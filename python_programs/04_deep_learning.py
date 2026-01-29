"""
Module 04: Deep Learning
========================
Learn about deep neural networks with many layers!

This program demonstrates deep learning concepts with PyTorch and TensorFlow examples.
"""

# ============================================
# What is Deep Learning?
# ============================================

def explain_deep_learning():
    """
    Explain deep learning in simple terms
    """
    print("=" * 60)
    print("What is Deep Learning?")
    print("=" * 60)
    
    print("""
Deep Learning = Neural Networks with MANY layers!

Why "Deep"?
    - Shallow network: 1-2 hidden layers
    - Deep network: 10, 100, or even 1000+ layers!

Why does depth matter?
    Layer 1: Detects simple patterns (edges, colors)
    Layer 2: Combines into shapes (circles, lines)
    Layer 3: Recognizes parts (eyes, wheels)
    Layer 4: Identifies objects (faces, cars)
    ...and so on!

Each layer learns more complex features!

Example: Recognizing a face
    Layer 1: Edges and gradients
    Layer 2: Eyes, nose, mouth shapes
    Layer 3: Face structure
    Layer 4: Individual person identity

Key Breakthroughs:
    1. GPUs: Fast parallel computing
    2. Big Data: Millions of training examples
    3. Better algorithms: ReLU, Dropout, BatchNorm
    4. Pre-trained models: Transfer learning
    """)


# ============================================
# PyTorch Deep Learning Examples
# ============================================

pytorch_code = '''
"""
Deep Learning with PyTorch
==========================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ============================================
# 1. Simple Deep Neural Network
# ============================================

class DeepNetwork(nn.Module):
    """
    A deep neural network with multiple layers
    """
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DeepNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Create hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Create a deep network
model = DeepNetwork(
    input_size=784,      # 28x28 image flattened
    hidden_sizes=[512, 256, 128, 64],  # 4 hidden layers
    output_size=10       # 10 digit classes
)

print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")


# ============================================
# 2. Convolutional Neural Network (CNN)
# ============================================

class CNN(nn.Module):
    """
    CNN for image classification
    """
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# ============================================
# 3. Training Loop
# ============================================

def train_model(model, train_loader, epochs=10, lr=0.001):
    """
    Complete training loop for deep learning
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        scheduler.step()
        
        accuracy = 100. * correct / total
        print(f'Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={accuracy:.2f}%')


# ============================================
# 4. Regularization Techniques
# ============================================

class RegularizedNetwork(nn.Module):
    """
    Network with various regularization techniques
    """
    def __init__(self):
        super(RegularizedNetwork, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),      # Batch Normalization
            nn.Dropout(0.3),           # Dropout
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

# L2 Regularization (Weight Decay)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)


# ============================================
# 5. Transfer Learning
# ============================================

import torchvision.models as models

def create_transfer_model(num_classes):
    """
    Use a pre-trained model and fine-tune it
    """
    # Load pre-trained ResNet
    model = models.resnet18(pretrained=True)
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace final layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    
    return model


# ============================================
# 6. Data Augmentation
# ============================================

from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("PyTorch deep learning examples loaded!")
'''


# ============================================
# TensorFlow Deep Learning Examples
# ============================================

tensorflow_code = '''
"""
Deep Learning with TensorFlow/Keras
===================================
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks

# ============================================
# 1. Simple Deep Neural Network
# ============================================

def create_deep_network(input_shape, num_classes):
    """
    Create a deep neural network with Keras
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # Hidden layers
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ============================================
# 2. Convolutional Neural Network
# ============================================

def create_cnn(input_shape, num_classes):
    """
    Create a CNN for image classification
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # First conv block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Second conv block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Third conv block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ============================================
# 3. Training with Callbacks
# ============================================

def train_with_callbacks(model, x_train, y_train, x_val, y_val):
    """
    Train with useful callbacks
    """
    my_callbacks = [
        # Stop if no improvement
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        
        # Save best model
        callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_accuracy',
            save_best_only=True
        ),
        
        # Reduce learning rate
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3
        ),
        
        # TensorBoard logging
        callbacks.TensorBoard(log_dir='./logs')
    ]
    
    history = model.fit(
        x_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(x_val, y_val),
        callbacks=my_callbacks
    )
    
    return history


# ============================================
# 4. Transfer Learning with Keras
# ============================================

def create_transfer_model(num_classes):
    """
    Transfer learning with pre-trained model
    """
    # Load pre-trained model (without top layers)
    base_model = keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Add custom layers
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ============================================
# 5. Data Augmentation
# ============================================

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# Use in model
augmented_model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    data_augmentation,
    # ... rest of model
])


# ============================================
# 6. Custom Training Loop
# ============================================

@tf.function
def train_step(model, optimizer, loss_fn, x, y):
    """
    Custom training step for more control
    """
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss

print("TensorFlow deep learning examples loaded!")
'''


def show_pytorch_examples():
    """Show PyTorch examples"""
    print("\n" + "=" * 60)
    print("PyTorch Deep Learning Examples")
    print("=" * 60)
    print(pytorch_code)


def show_tensorflow_examples():
    """Show TensorFlow examples"""
    print("\n" + "=" * 60)
    print("TensorFlow/Keras Deep Learning Examples")
    print("=" * 60)
    print(tensorflow_code)


# ============================================
# Key Concepts
# ============================================

def key_concepts():
    """
    Explain key deep learning concepts
    """
    print("\n" + "=" * 60)
    print("Key Deep Learning Concepts")
    print("=" * 60)
    
    concepts = [
        {
            "name": "Batch Normalization",
            "what": "Normalizes layer inputs to have mean=0, std=1",
            "why": "Faster training, allows higher learning rates",
            "when": "After linear/conv layers, before activation"
        },
        {
            "name": "Dropout",
            "what": "Randomly sets neurons to 0 during training",
            "why": "Prevents overfitting, acts like ensemble",
            "when": "After activation, typically 0.2-0.5"
        },
        {
            "name": "Learning Rate Scheduling",
            "what": "Changes learning rate during training",
            "why": "Start fast, then fine-tune with smaller steps",
            "when": "After initial training or on plateau"
        },
        {
            "name": "Data Augmentation",
            "what": "Creates variations of training data",
            "why": "More data = better generalization",
            "when": "During training, especially with small datasets"
        },
        {
            "name": "Transfer Learning",
            "what": "Use pre-trained model as starting point",
            "why": "Leverage knowledge from large datasets",
            "when": "Limited data or similar task to pre-trained model"
        }
    ]
    
    for concept in concepts:
        print(f"\n📌 {concept['name']}")
        print(f"   What: {concept['what']}")
        print(f"   Why: {concept['why']}")
        print(f"   When: {concept['when']}")


# ============================================
# Main Program
# ============================================

def main():
    """
    Main function to run all demos
    """
    print("\n" + "🔥" * 25)
    print("\n   DEEP LEARNING")
    print("\n" + "🔥" * 25)
    
    while True:
        print("\n" + "-" * 60)
        print("Choose a demo:")
        print("1. What is Deep Learning?")
        print("2. PyTorch Examples")
        print("3. TensorFlow/Keras Examples")
        print("4. Key Concepts")
        print("5. Show All")
        print("0. Exit")
        print("-" * 60)
        
        choice = input("\nEnter your choice (0-5): ").strip()
        
        if choice == "0":
            print("\nThanks for learning about Deep Learning! Goodbye! 👋")
            break
        elif choice == "1":
            explain_deep_learning()
        elif choice == "2":
            show_pytorch_examples()
        elif choice == "3":
            show_tensorflow_examples()
        elif choice == "4":
            key_concepts()
        elif choice == "5":
            explain_deep_learning()
            key_concepts()
            show_pytorch_examples()
            show_tensorflow_examples()
        else:
            print("Invalid choice. Please enter 0-5.")


if __name__ == "__main__":
    main()
