import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Sparkles, Wand2, Image, Code, RefreshCw } from 'lucide-react'

const Module07GenerativeAI = () => {
  const [generating, setGenerating] = useState(false)
  const [generatedArt, setGeneratedArt] = useState<string[]>([])
  const [ganStep, setGanStep] = useState(0)

  const generateArt = () => {
    setGenerating(true)
    const colors = ['from-pink-500 to-purple-500', 'from-blue-500 to-cyan-500', 'from-green-500 to-yellow-500', 'from-red-500 to-orange-500']
    setTimeout(() => {
      setGeneratedArt([...generatedArt, colors[Math.floor(Math.random() * colors.length)]])
      setGenerating(false)
    }, 1500)
  }

  useEffect(() => {
    const interval = setInterval(() => {
      setGanStep(prev => (prev + 1) % 5)
    }, 2000)
    return () => clearInterval(interval)
  }, [])

  const pythonCode = `# Generative AI - Creating New Content!
# GANs, VAEs, and Diffusion Models

import torch
import torch.nn as nn
import numpy as np

# ============================================
# Generative Adversarial Network (GAN)
# ============================================

class Generator(nn.Module):
    """
    The Generator creates fake data
    Like an artist trying to create convincing paintings!
    """
    def __init__(self, latent_dim=100, output_dim=784):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
            nn.Tanh()  # Output between -1 and 1
        )
    
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    """
    The Discriminator tries to spot fakes
    Like an art critic detecting forgeries!
    """
    def __init__(self, input_dim=784):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output probability (real or fake)
        )
    
    def forward(self, x):
        return self.model(x)

class GAN:
    """
    Complete GAN training system
    Generator and Discriminator compete to get better!
    """
    def __init__(self, latent_dim=100):
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()
        
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002)
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002)
        self.criterion = nn.BCELoss()
    
    def train_step(self, real_data):
        batch_size = real_data.size(0)
        
        # Labels
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        
        # ===== Train Discriminator =====
        self.d_optimizer.zero_grad()
        
        # Real data
        real_output = self.discriminator(real_data)
        d_loss_real = self.criterion(real_output, real_labels)
        
        # Fake data
        z = torch.randn(batch_size, self.latent_dim)
        fake_data = self.generator(z)
        fake_output = self.discriminator(fake_data.detach())
        d_loss_fake = self.criterion(fake_output, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.d_optimizer.step()
        
        # ===== Train Generator =====
        self.g_optimizer.zero_grad()
        
        # Try to fool discriminator
        fake_output = self.discriminator(fake_data)
        g_loss = self.criterion(fake_output, real_labels)  # Want discriminator to think it's real
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return d_loss.item(), g_loss.item()
    
    def generate(self, num_samples=1):
        """Generate new samples"""
        z = torch.randn(num_samples, self.latent_dim)
        with torch.no_grad():
            samples = self.generator(z)
        return samples

# ============================================
# Variational Autoencoder (VAE)
# ============================================

class VAE(nn.Module):
    """
    VAE learns to compress and reconstruct data
    Like learning to describe a picture with few words,
    then recreating it from that description!
    """
    def __init__(self, input_dim=784, latent_dim=20):
        super().__init__()
        
        # Encoder: compress input to latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)
        
        # Decoder: reconstruct from latent space
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
        """The reparameterization trick for backpropagation"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
    
    def generate(self, num_samples=1):
        """Generate new samples from random latent vectors"""
        z = torch.randn(num_samples, 20)
        with torch.no_grad():
            samples = self.decode(z)
        return samples

def vae_loss(recon_x, x, mu, log_var):
    """VAE loss = reconstruction loss + KL divergence"""
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_loss

# ============================================
# Simple Diffusion Model Concept
# ============================================

class SimpleDiffusion:
    """
    Diffusion models add noise then learn to remove it
    Like learning to clean up a messy picture!
    """
    def __init__(self, num_steps=100):
        self.num_steps = num_steps
        # Noise schedule
        self.betas = torch.linspace(0.0001, 0.02, num_steps)
        self.alphas = 1 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def add_noise(self, x, t):
        """Add noise to image at timestep t"""
        noise = torch.randn_like(x)
        alpha_t = self.alpha_cumprod[t]
        noisy_x = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise
        return noisy_x, noise
    
    def denoise_step(self, model, x_t, t):
        """One step of denoising"""
        # Model predicts the noise
        predicted_noise = model(x_t, t)
        
        # Remove predicted noise
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alpha_cumprod[t]
        
        x_t_minus_1 = (1 / torch.sqrt(alpha_t)) * (
            x_t - (self.betas[t] / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
        )
        
        return x_t_minus_1

# ============================================
# Using Hugging Face Diffusers
# ============================================

from diffusers import StableDiffusionPipeline
import torch

# Load Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# Generate image from text
def generate_image(prompt):
    """
    Text-to-image generation
    Describe what you want, AI creates it!
    """
    image = pipe(prompt).images[0]
    image.save("generated_image.png")
    return image

# Example
# image = generate_image("A beautiful sunset over mountains, digital art")

# ============================================
# Image-to-Image Generation
# ============================================

from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

def transform_image(image_path, prompt, strength=0.75):
    """
    Transform an existing image based on a prompt
    Like giving an artist a sketch to work from!
    """
    init_image = Image.open(image_path).convert("RGB")
    init_image = init_image.resize((512, 512))
    
    result = img2img_pipe(
        prompt=prompt,
        image=init_image,
        strength=strength
    ).images[0]
    
    return result

# ============================================
# Text Generation with GPT
# ============================================

from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def generate_text(prompt, max_length=100):
    """
    Generate text continuation
    AI writes the next part of the story!
    """
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example
# text = generate_text("Once upon a time in a magical forest,")`

  return (
    <div className="space-y-8">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-white mb-4">Generative AI</h1>
        <p className="text-xl text-purple-200">AI that creates new content - images, text, music, and more!</p>
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
                <Sparkles className="w-6 h-6 text-yellow-400" />
                What is Generative AI?
              </CardTitle>
            </CardHeader>
            <CardContent className="text-slate-300 space-y-4">
              <p className="text-lg">
                <strong className="text-yellow-400">Generative AI</strong> creates new content that never existed before! 
                Unlike AI that just classifies or predicts, generative AI can create images, write stories, 
                compose music, and even generate code.
              </p>
              <div className="bg-slate-900/50 p-4 rounded-lg border border-yellow-500/30">
                <h4 className="text-yellow-400 font-semibold mb-2">Think of it like this:</h4>
                <p>Imagine an artist who has seen millions of paintings. They can now create entirely new 
                paintings in any style! Generative AI works similarly - it learns from existing content 
                and creates new, original content.</p>
              </div>
            </CardContent>
          </Card>

          <div className="grid md:grid-cols-3 gap-4">
            {[
              { 
                title: 'GANs', 
                subtitle: 'Generative Adversarial Networks',
                desc: 'Two AI models compete: one creates, one judges. They make each other better!',
                icon: '🎭',
                color: 'bg-purple-500'
              },
              { 
                title: 'VAEs', 
                subtitle: 'Variational Autoencoders',
                desc: 'Compress data to its essence, then recreate variations from it.',
                icon: '🗜️',
                color: 'bg-blue-500'
              },
              { 
                title: 'Diffusion', 
                subtitle: 'Diffusion Models',
                desc: 'Add noise to images, then learn to remove it. Creates amazing art!',
                icon: '🌫️',
                color: 'bg-green-500'
              },
            ].map((item, i) => (
              <Card key={i} className="bg-slate-800/50 border-slate-700">
                <CardHeader>
                  <div className={`w-12 h-12 rounded-lg ${item.color} flex items-center justify-center mb-2 text-2xl`}>
                    {item.icon}
                  </div>
                  <CardTitle className="text-white text-lg">{item.title}</CardTitle>
                  <CardDescription className="text-slate-500 text-xs">{item.subtitle}</CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-slate-400 text-sm">{item.desc}</p>
                </CardContent>
              </Card>
            ))}
          </div>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">What Can Generative AI Create?</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-3">
                {[
                  { emoji: '🖼️', title: 'Images', desc: 'Art, photos, designs from text descriptions', examples: 'DALL-E, Midjourney, Stable Diffusion' },
                  { emoji: '📝', title: 'Text', desc: 'Stories, articles, code, conversations', examples: 'GPT-4, Claude, LLaMA' },
                  { emoji: '🎵', title: 'Music', desc: 'Songs, melodies, sound effects', examples: 'MusicLM, Jukebox' },
                  { emoji: '🎬', title: 'Video', desc: 'Animations, video clips, deepfakes', examples: 'Runway, Pika Labs' },
                  { emoji: '🗣️', title: 'Voice', desc: 'Speech synthesis, voice cloning', examples: 'ElevenLabs, Bark' },
                  { emoji: '🧬', title: '3D Models', desc: 'Objects, characters, environments', examples: 'Point-E, DreamFusion' },
                ].map((item, i) => (
                  <div key={i} className="flex items-start gap-3 p-3 bg-slate-900 rounded-lg">
                    <span className="text-2xl">{item.emoji}</span>
                    <div>
                      <h4 className="text-white font-medium">{item.title}</h4>
                      <p className="text-slate-400 text-sm">{item.desc}</p>
                      <p className="text-yellow-400 text-xs mt-1">{item.examples}</p>
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
                <Wand2 className="w-5 h-5 text-yellow-400" />
                How GANs Work - The Art Forger Game
              </CardTitle>
              <CardDescription className="text-slate-400">
                Watch the Generator and Discriminator compete!
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="relative bg-slate-900 rounded-lg p-6">
                <div className="grid md:grid-cols-3 gap-8 items-center">
                  <div className="text-center">
                    <div className={`w-24 h-24 mx-auto rounded-full bg-purple-500 flex items-center justify-center text-4xl mb-3 ${ganStep >= 1 ? 'ring-4 ring-purple-300 animate-pulse' : ''}`}>
                      🎨
                    </div>
                    <h4 className="text-purple-400 font-semibold">Generator</h4>
                    <p className="text-slate-400 text-sm">Creates fake images</p>
                    <p className="text-xs text-slate-500 mt-1">"Let me try to fool you!"</p>
                  </div>
                  
                  <div className="text-center">
                    <div className="space-y-2">
                      <div className={`h-16 w-16 mx-auto rounded-lg transition-all duration-500 ${
                        ganStep === 0 ? 'bg-slate-700' :
                        ganStep === 1 ? 'bg-gradient-to-br from-purple-500 to-pink-500 animate-pulse' :
                        ganStep === 2 ? 'bg-gradient-to-br from-purple-500 to-pink-500' :
                        ganStep === 3 ? 'bg-gradient-to-br from-purple-500 to-pink-500 ring-2 ring-red-500' :
                        'bg-gradient-to-br from-purple-500 to-pink-500 ring-2 ring-green-500'
                      }`}>
                        {ganStep >= 2 && <span className="text-2xl">🖼️</span>}
                      </div>
                      <p className="text-slate-400 text-sm">
                        {ganStep === 0 && 'Waiting...'}
                        {ganStep === 1 && 'Generating...'}
                        {ganStep === 2 && 'Image created!'}
                        {ganStep === 3 && 'Judging...'}
                        {ganStep === 4 && 'Learning!'}
                      </p>
                    </div>
                  </div>
                  
                  <div className="text-center">
                    <div className={`w-24 h-24 mx-auto rounded-full bg-blue-500 flex items-center justify-center text-4xl mb-3 ${ganStep >= 3 ? 'ring-4 ring-blue-300 animate-pulse' : ''}`}>
                      🔍
                    </div>
                    <h4 className="text-blue-400 font-semibold">Discriminator</h4>
                    <p className="text-slate-400 text-sm">Spots fakes</p>
                    <p className="text-xs text-slate-500 mt-1">"Is this real or fake?"</p>
                  </div>
                </div>
                
                <div className="mt-6 flex justify-center gap-2">
                  {[0, 1, 2, 3, 4].map(step => (
                    <div key={step} className={`w-3 h-3 rounded-full ${ganStep === step ? 'bg-yellow-400' : 'bg-slate-600'}`} />
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Image className="w-5 h-5 text-pink-400" />
                AI Art Generator Simulator
              </CardTitle>
              <CardDescription className="text-slate-400">
                Click to generate new AI art!
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button 
                onClick={generateArt} 
                disabled={generating}
                className="w-full bg-gradient-to-r from-pink-500 to-purple-500 hover:from-pink-600 hover:to-purple-600"
              >
                {generating ? (
                  <>
                    <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4 mr-2" />
                    Generate AI Art
                  </>
                )}
              </Button>
              
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {generatedArt.map((gradient, i) => (
                  <div 
                    key={i}
                    className={`aspect-square rounded-lg bg-gradient-to-br ${gradient} flex items-center justify-center animate-in fade-in`}
                  >
                    <span className="text-4xl">🎨</span>
                  </div>
                ))}
                {generatedArt.length === 0 && (
                  <div className="col-span-4 text-center py-8 text-slate-500">
                    Click the button to generate AI art!
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Diffusion Process Visualization</CardTitle>
              <CardDescription className="text-slate-400">
                How diffusion models create images from noise
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between gap-2">
                {['Pure Noise', 'Step 25', 'Step 50', 'Step 75', 'Final Image'].map((step, i) => (
                  <div key={i} className="text-center flex-1">
                    <div className={`aspect-square rounded-lg mb-2 flex items-center justify-center ${
                      i === 0 ? 'bg-gradient-to-br from-gray-600 to-gray-800' :
                      i === 1 ? 'bg-gradient-to-br from-gray-500 to-purple-900' :
                      i === 2 ? 'bg-gradient-to-br from-purple-700 to-blue-700' :
                      i === 3 ? 'bg-gradient-to-br from-purple-500 to-blue-500' :
                      'bg-gradient-to-br from-purple-400 to-pink-400'
                    }`}>
                      <span className="text-2xl">{i === 4 ? '🐱' : '🌫️'}</span>
                    </div>
                    <p className="text-slate-400 text-xs">{step}</p>
                  </div>
                ))}
              </div>
              <p className="text-slate-400 text-sm mt-4 text-center">
                Diffusion models start with random noise and gradually remove it to create an image!
              </p>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="code" className="space-y-6">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Code className="w-5 h-5 text-yellow-400" />
                Python Code: Generative AI
              </CardTitle>
              <CardDescription className="text-slate-400">
                Build GANs, VAEs, and use Stable Diffusion!
              </CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="bg-slate-900 p-4 rounded-lg overflow-x-auto text-sm max-h-96 overflow-y-auto">
                <code className="text-green-400">{pythonCode}</code>
              </pre>
              <div className="mt-4 p-4 bg-yellow-500/10 rounded-lg border border-yellow-500/30">
                <h4 className="text-yellow-400 font-semibold mb-2">Required Libraries:</h4>
                <code className="text-slate-300 text-sm bg-slate-800 px-2 py-1 rounded">
                  pip install torch diffusers transformers accelerate
                </code>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default Module07GenerativeAI
