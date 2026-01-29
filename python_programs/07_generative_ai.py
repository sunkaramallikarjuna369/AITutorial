"""
=============================================================================
MODULE 07: GENERATIVE AI - COMPREHENSIVE 360 DEGREE COVERAGE
=============================================================================

This module provides COMPLETE coverage of Generative AI including:
- 4W+H Explanations (What, Why, When, Where, How)
- GANs (Generative Adversarial Networks)
- VAEs (Variational Autoencoders)
- Diffusion Models
- GenAI Model Integrations
- Interview Questions
- Common Pitfalls

SETUP:
------
pip install torch torchvision numpy

For GenAI features:
pip install openai anthropic google-generativeai

=============================================================================
"""

import os
import math
import random
from typing import List, Tuple, Dict, Optional

# Try to import optional libraries
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =============================================================================
# GENAI INTEGRATION
# =============================================================================

class GenerativeAIAssistant:
    """
    Use GenAI models for explanations and generation
    
    Supports: OpenAI (DALL-E), Anthropic Claude, Google Gemini, Ollama
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
                    "content": f"Explain {concept} in generative AI with a simple analogy and example."
                }],
                max_tokens=500
            )
            return response.choices[0].message.content
        except ImportError:
            return "[Install openai: pip install openai]"
        except Exception as e:
            return f"[OpenAI Error: {e}]"
    
    def generate_image_with_dalle(self, prompt: str) -> str:
        """Generate image using DALL-E"""
        try:
            from openai import OpenAI
            if not self.openai_key:
                return "[Set OPENAI_API_KEY for DALL-E]"
            
            client = OpenAI(api_key=self.openai_key)
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1
            )
            return f"Image URL: {response.data[0].url}"
        except Exception as e:
            return f"[DALL-E Error: {e}]"
    
    def explain_with_ollama(self, concept: str, model: str = "llama2") -> str:
        """Get AI explanation using Ollama (FREE, local)"""
        try:
            import requests
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": f"Explain {concept} in generative AI with an example.",
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
# SECTION 1: COMPREHENSIVE GENERATIVE AI EXPLANATION (4W+H)
# =============================================================================

def explain_generative_ai_comprehensive():
    """Comprehensive 360-degree explanation of Generative AI"""
    print("=" * 70)
    print("GENERATIVE AI - COMPREHENSIVE 360 DEGREE COVERAGE")
    print("=" * 70)
    
    print("""
WHAT IS GENERATIVE AI?
======================

Generative AI creates NEW content (images, text, audio, video) that
didn't exist before, learning patterns from training data.

DISCRIMINATIVE vs GENERATIVE:
    Discriminative: P(y|x) - Given input, predict label
    Generative: P(x) or P(x|z) - Model the data distribution, generate samples

KEY TYPES:
    1. GANs - Two networks compete (Generator vs Discriminator)
    2. VAEs - Encode to latent space, decode to generate
    3. Diffusion - Add noise, learn to denoise
    4. Autoregressive - Generate one token at a time (GPT)
    5. Flow-based - Invertible transformations

WHY IS GENERATIVE AI IMPORTANT?
===============================

1. CREATIVITY: Generate art, music, writing
2. DATA AUGMENTATION: Create synthetic training data
3. CONTENT CREATION: Marketing, design, entertainment
4. SIMULATION: Generate scenarios for training
5. PERSONALIZATION: Custom content for users

WHEN TO USE GENERATIVE AI?
==========================

USE FOR:
    - Image generation and editing
    - Text generation (chatbots, content)
    - Music and audio synthesis
    - Video generation
    - Data augmentation
    - Drug discovery (molecule generation)

WHERE IS GENERATIVE AI USED?
============================

PRODUCTS:
    - ChatGPT, Claude, Gemini (text)
    - DALL-E, Midjourney, Stable Diffusion (images)
    - Suno, Udio (music)
    - Runway, Pika (video)
    - GitHub Copilot (code)

INDUSTRIES:
    - Entertainment: Game assets, movie effects
    - Marketing: Ad copy, product images
    - Healthcare: Drug discovery, medical imaging
    - Fashion: Design generation
    - Architecture: Building designs

HOW DOES GENERATIVE AI WORK?
============================

GAN (Generative Adversarial Network):
    Generator: Noise -> Fake Image
    Discriminator: Image -> Real/Fake
    Training: Generator tries to fool Discriminator
    
VAE (Variational Autoencoder):
    Encoder: Image -> Latent Distribution (mu, sigma)
    Sampling: z ~ N(mu, sigma)
    Decoder: z -> Reconstructed Image
    Loss: Reconstruction + KL Divergence

DIFFUSION:
    Forward: Add noise gradually (T steps)
    Reverse: Learn to denoise (predict noise)
    Generation: Start from noise, denoise step by step
    """)


# =============================================================================
# SECTION 2: HISTORY AND EVOLUTION
# =============================================================================

def generative_ai_history():
    """History and evolution of Generative AI"""
    print("\n" + "=" * 70)
    print("HISTORY AND EVOLUTION OF GENERATIVE AI")
    print("=" * 70)
    
    timeline = [
        ("1980s", "Boltzmann Machines", "Early generative models"),
        ("2013", "VAE", "Variational Autoencoders introduced"),
        ("2014", "GAN", "Generative Adversarial Networks (Goodfellow)"),
        ("2015", "DCGAN", "Deep Convolutional GANs for images"),
        ("2017", "Transformer", "Foundation for modern text generation"),
        ("2018", "GPT-1", "Generative Pre-trained Transformer"),
        ("2019", "StyleGAN", "High-quality face generation"),
        ("2020", "GPT-3", "175B parameters, few-shot learning"),
        ("2020", "DDPM", "Denoising Diffusion Probabilistic Models"),
        ("2021", "DALL-E", "Text-to-image generation"),
        ("2022", "Stable Diffusion", "Open-source diffusion model"),
        ("2022", "ChatGPT", "Conversational AI mainstream"),
        ("2023", "GPT-4", "Multimodal, reasoning"),
        ("2023", "Midjourney v5", "Photorealistic image generation"),
        ("2024", "Sora", "Text-to-video generation"),
    ]
    
    print("\nTIMELINE:")
    print("-" * 70)
    for year, event, description in timeline:
        print(f"{year}: {event} - {description}")


# =============================================================================
# SECTION 3: GANs (GENERATIVE ADVERSARIAL NETWORKS)
# =============================================================================

def gan_explanation():
    """Detailed explanation of GANs"""
    print("\n" + "=" * 70)
    print("GANs - GENERATIVE ADVERSARIAL NETWORKS")
    print("=" * 70)
    
    print("""
CONCEPT:
    Two neural networks compete in a game:
    - Generator (G): Creates fake data from noise
    - Discriminator (D): Distinguishes real from fake

ANALOGY:
    Generator = Art forger trying to create fake paintings
    Discriminator = Art expert trying to detect fakes
    Both improve through competition!

TRAINING:
    1. Sample real data x from dataset
    2. Sample noise z from prior (e.g., Gaussian)
    3. Generate fake data: G(z)
    4. Train D to maximize: log(D(x)) + log(1 - D(G(z)))
    5. Train G to minimize: log(1 - D(G(z))) or maximize log(D(G(z)))

LOSS FUNCTIONS:
    D_loss = -[log(D(x)) + log(1 - D(G(z)))]
    G_loss = -log(D(G(z)))  # Non-saturating version

GAN VARIANTS:
    - DCGAN: Deep Convolutional GAN
    - WGAN: Wasserstein GAN (stable training)
    - StyleGAN: Style-based generator
    - CycleGAN: Unpaired image-to-image translation
    - Pix2Pix: Paired image-to-image translation
    - BigGAN: Large-scale image generation

CHALLENGES:
    - Mode collapse: Generator produces limited variety
    - Training instability: Oscillating losses
    - Evaluation: Hard to measure quality objectively
    """)


def simple_gan_demo():
    """Demo simple GAN concept"""
    print("\n" + "=" * 70)
    print("SIMPLE GAN DEMO")
    print("=" * 70)
    
    if not HAS_TORCH:
        print("\nInstall PyTorch: pip install torch")
        print("""
Example GAN code:

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_size=784):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, img_size),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, img_size=784):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)
        """)
        return
    
    # Simple GAN implementation
    class Generator(nn.Module):
        def __init__(self, latent_dim=100, output_dim=784):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, output_dim),
                nn.Tanh()
            )
        
        def forward(self, z):
            return self.model(z)
    
    class Discriminator(nn.Module):
        def __init__(self, input_dim=784):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.model(x)
    
    G = Generator()
    D = Discriminator()
    
    print(f"Generator parameters: {sum(p.numel() for p in G.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in D.parameters()):,}")
    
    # Test
    z = torch.randn(1, 100)
    fake = G(z)
    prob = D(fake)
    print(f"\nNoise shape: {z.shape}")
    print(f"Generated shape: {fake.shape}")
    print(f"Discriminator output: {prob.item():.4f}")


# =============================================================================
# SECTION 4: VAEs (VARIATIONAL AUTOENCODERS)
# =============================================================================

def vae_explanation():
    """Detailed explanation of VAEs"""
    print("\n" + "=" * 70)
    print("VAEs - VARIATIONAL AUTOENCODERS")
    print("=" * 70)
    
    print("""
CONCEPT:
    Learn a latent representation of data that can be sampled
    to generate new data.

ARCHITECTURE:
    Encoder: x -> (mu, log_var)  # Parameters of latent distribution
    Sampling: z = mu + exp(0.5 * log_var) * epsilon  # Reparameterization
    Decoder: z -> x_reconstructed

LOSS FUNCTION:
    L = Reconstruction Loss + KL Divergence
    
    Reconstruction: How well can we reconstruct the input?
    KL Divergence: How close is latent distribution to prior N(0,1)?
    
    L = ||x - x_reconstructed||^2 + KL(q(z|x) || p(z))

REPARAMETERIZATION TRICK:
    Instead of sampling z ~ N(mu, sigma),
    Sample epsilon ~ N(0, 1)
    Compute z = mu + sigma * epsilon
    
    This allows gradients to flow through sampling!

ADVANTAGES:
    - Smooth latent space (interpolation works)
    - Principled probabilistic framework
    - Can compute likelihood

DISADVANTAGES:
    - Blurry outputs (compared to GANs)
    - KL term can dominate (posterior collapse)
    """)


# =============================================================================
# SECTION 5: DIFFUSION MODELS
# =============================================================================

def diffusion_explanation():
    """Detailed explanation of Diffusion Models"""
    print("\n" + "=" * 70)
    print("DIFFUSION MODELS")
    print("=" * 70)
    
    print("""
CONCEPT:
    Learn to reverse a gradual noising process.
    
    Forward: Gradually add noise until pure noise
    Reverse: Learn to denoise step by step

FORWARD PROCESS (Fixed):
    x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * noise
    
    After T steps, x_T is approximately pure Gaussian noise.

REVERSE PROCESS (Learned):
    x_{t-1} = model(x_t, t)  # Predict less noisy version
    
    The model learns to predict the noise added at each step.

TRAINING:
    1. Sample x_0 from data
    2. Sample t uniformly from [1, T]
    3. Sample noise epsilon ~ N(0, I)
    4. Compute x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    5. Train model to predict epsilon from x_t and t
    
    Loss = ||epsilon - model(x_t, t)||^2

GENERATION:
    1. Start with x_T ~ N(0, I)
    2. For t = T, T-1, ..., 1:
       - Predict noise: epsilon = model(x_t, t)
       - Compute x_{t-1} using reverse formula
    3. Return x_0

KEY MODELS:
    - DDPM (2020): Denoising Diffusion Probabilistic Models
    - DDIM (2020): Faster sampling
    - Stable Diffusion (2022): Latent diffusion + text conditioning
    - DALL-E 2/3 (2022-23): Text-to-image
    - Midjourney: Commercial text-to-image

ADVANTAGES:
    - High quality outputs
    - Stable training
    - Good mode coverage

DISADVANTAGES:
    - Slow generation (many steps)
    - High computational cost
    """)


# =============================================================================
# SECTION 6: INTERVIEW QUESTIONS
# =============================================================================

def interview_questions():
    """Common Generative AI interview questions"""
    print("\n" + "=" * 70)
    print("GENERATIVE AI INTERVIEW QUESTIONS")
    print("=" * 70)
    
    questions = [
        ("What is mode collapse in GANs?",
         "Generator produces limited variety of outputs, ignoring modes of the data distribution. Solutions: Wasserstein loss, minibatch discrimination, unrolled GANs."),
        ("Explain the reparameterization trick in VAEs.",
         "Instead of sampling z ~ N(mu, sigma) directly, sample epsilon ~ N(0,1) and compute z = mu + sigma * epsilon. This allows gradients to flow through the sampling operation."),
        ("Why do diffusion models produce higher quality images than GANs?",
         "Diffusion models have stable training (no adversarial dynamics), better mode coverage, and the iterative refinement process allows for high-quality details."),
        ("What is classifier-free guidance?",
         "Training a conditional diffusion model with random dropout of the condition. At inference, interpolate between conditional and unconditional predictions to control adherence to the condition."),
        ("How does Stable Diffusion work?",
         "Latent diffusion: encode image to latent space with VAE, apply diffusion in latent space (faster), decode back to image. Text conditioning via CLIP embeddings and cross-attention."),
    ]
    
    for i, (q, a) in enumerate(questions, 1):
        print(f"\nQ{i}: {q}")
        print(f"A: {a}")


# =============================================================================
# SECTION 7: COMMON PITFALLS
# =============================================================================

def common_pitfalls():
    """Common mistakes in Generative AI"""
    print("\n" + "=" * 70)
    print("COMMON GENERATIVE AI PITFALLS")
    print("=" * 70)
    
    pitfalls = [
        ("GAN training instability", "Use WGAN-GP, spectral normalization, or progressive growing"),
        ("Mode collapse", "Use minibatch discrimination, feature matching, or diffusion models"),
        ("VAE blurry outputs", "Use perceptual loss, adversarial loss, or switch to diffusion"),
        ("Slow diffusion sampling", "Use DDIM, DPM-Solver, or distillation"),
        ("Poor text-image alignment", "Increase classifier-free guidance scale"),
        ("Overfitting to training data", "Use data augmentation, regularization"),
    ]
    
    for mistake, solution in pitfalls:
        print(f"\nMistake: {mistake}")
        print(f"Solution: {solution}")


# =============================================================================
# MAIN PROGRAM
# =============================================================================

def main():
    print("=" * 70)
    print("   MODULE 07: GENERATIVE AI - COMPREHENSIVE 360 DEGREE COVERAGE")
    print("=" * 70)
    
    assistant = GenerativeAIAssistant()
    
    while True:
        print("\n" + "-" * 70)
        print("Choose a topic:")
        print()
        print("FUNDAMENTALS:")
        print("  1. What is Generative AI? (4W+H)")
        print("  2. History and Evolution")
        print()
        print("ARCHITECTURES:")
        print("  3. GANs Explained")
        print("  4. Simple GAN Demo")
        print("  5. VAEs Explained")
        print("  6. Diffusion Models Explained")
        print()
        print("DEEP DIVE:")
        print("  7. Interview Questions")
        print("  8. Common Pitfalls")
        print()
        print("GENAI FEATURES:")
        print("  9. AI Explanation (OpenAI)")
        print("  10. AI Explanation (Ollama - FREE)")
        print("  11. Generate Image (DALL-E)")
        print()
        print("  12. Run ALL Topics")
        print("  0. Exit")
        print("-" * 70)
        
        choice = input("\nEnter choice (0-12): ").strip()
        
        if choice == "0":
            print("\nHappy learning!")
            break
        elif choice == "1":
            explain_generative_ai_comprehensive()
        elif choice == "2":
            generative_ai_history()
        elif choice == "3":
            gan_explanation()
        elif choice == "4":
            simple_gan_demo()
        elif choice == "5":
            vae_explanation()
        elif choice == "6":
            diffusion_explanation()
        elif choice == "7":
            interview_questions()
        elif choice == "8":
            common_pitfalls()
        elif choice == "9":
            concept = input("Enter concept to explain: ").strip()
            print(assistant.explain_with_openai(concept))
        elif choice == "10":
            concept = input("Enter concept to explain: ").strip()
            print(assistant.explain_with_ollama(concept))
        elif choice == "11":
            prompt = input("Enter image prompt: ").strip()
            print(assistant.generate_image_with_dalle(prompt))
        elif choice == "12":
            explain_generative_ai_comprehensive()
            generative_ai_history()
            gan_explanation()
            simple_gan_demo()
            vae_explanation()
            diffusion_explanation()
            interview_questions()
            common_pitfalls()
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
