"""
Module 07: Generative AI
========================
Learn how AI creates new content - images, text, music, and more!
"""

# ============================================
# Generative AI Examples
# ============================================

code_examples = '''
"""
Generative AI Examples
======================
"""

import torch
import torch.nn as nn
import numpy as np

# ============================================
# 1. Simple GAN (Generative Adversarial Network)
# ============================================

class Generator(nn.Module):
    """Generator creates fake data from random noise"""
    def __init__(self, latent_dim=100, output_dim=784):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    """Discriminator distinguishes real from fake"""
    def __init__(self, input_dim=784):
        super(Discriminator, self).__init__()
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


def train_gan(generator, discriminator, dataloader, epochs=100):
    """Train GAN"""
    criterion = nn.BCELoss()
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
    
    for epoch in range(epochs):
        for real_data in dataloader:
            batch_size = real_data.size(0)
            
            # Train Discriminator
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)
            
            d_optimizer.zero_grad()
            
            # Real data
            real_output = discriminator(real_data)
            d_loss_real = criterion(real_output, real_labels)
            
            # Fake data
            z = torch.randn(batch_size, 100)
            fake_data = generator(z)
            fake_output = discriminator(fake_data.detach())
            d_loss_fake = criterion(fake_output, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            fake_output = discriminator(fake_data)
            g_loss = criterion(fake_output, real_labels)
            g_loss.backward()
            g_optimizer.step()


# ============================================
# 2. Variational Autoencoder (VAE)
# ============================================

class VAE(nn.Module):
    """VAE learns to encode and decode data"""
    def __init__(self, input_dim=784, latent_dim=20):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


# ============================================
# 3. Text Generation with GPT-style Model
# ============================================

from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(prompt, max_length=100):
    """Generate text using GPT-2"""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ============================================
# 4. Image Generation with Stable Diffusion
# ============================================

from diffusers import StableDiffusionPipeline

def generate_image(prompt):
    """Generate image from text using Stable Diffusion"""
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    
    image = pipe(
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5
    ).images[0]
    
    return image


# ============================================
# 5. Music Generation
# ============================================

def generate_music_notes():
    """Simple music generation using Markov chains"""
    import random
    
    # Simple note transitions
    transitions = {
        'C': ['D', 'E', 'G'],
        'D': ['C', 'E', 'F'],
        'E': ['D', 'F', 'G'],
        'F': ['E', 'G', 'A'],
        'G': ['F', 'A', 'C'],
        'A': ['G', 'B', 'C'],
        'B': ['A', 'C', 'D']
    }
    
    melody = ['C']
    for _ in range(15):
        current = melody[-1]
        next_note = random.choice(transitions[current])
        melody.append(next_note)
    
    return melody


print("Generative AI examples loaded!")
print("Install: pip install torch transformers diffusers")
'''

def main():
    print("=" * 60)
    print("Generative AI - Creating New Content")
    print("=" * 60)
    
    print("""
Generative AI creates NEW content that didn't exist before!

Types of Generative Models:
1. GANs (Generative Adversarial Networks)
   - Two networks compete: Generator vs Discriminator
   - Great for realistic images

2. VAEs (Variational Autoencoders)
   - Learn compressed representations
   - Good for variations of existing data

3. Diffusion Models (Stable Diffusion, DALL-E)
   - Start with noise, gradually denoise
   - State-of-the-art image generation

4. Transformers (GPT, Claude)
   - Predict next token
   - Excellent for text generation

5. Flow-based Models
   - Invertible transformations
   - Exact likelihood computation

Applications:
- Text: ChatGPT, Claude, Gemini
- Images: DALL-E, Midjourney, Stable Diffusion
- Music: Suno, Udio
- Video: Sora, Runway
- Code: GitHub Copilot, Cursor
    """)
    
    print("\nPython Code Examples:")
    print(code_examples)

if __name__ == "__main__":
    main()
