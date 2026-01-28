import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Settings, Layers, Code, ArrowRight, Zap } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

const Module12FineTuning = () => {
  const [trainingProgress, setTrainingProgress] = useState(0)
  const [isTraining, setIsTraining] = useState(false)
  const [lossData, setLossData] = useState<{epoch: number, loss: number}[]>([])

  const simulateTraining = () => {
    setIsTraining(true)
    setTrainingProgress(0)
    setLossData([])
    
    let epoch = 0
    const interval = setInterval(() => {
      epoch++
      const loss = 2.5 * Math.exp(-0.3 * epoch) + 0.1 + Math.random() * 0.1
      setLossData(prev => [...prev, { epoch, loss: parseFloat(loss.toFixed(3)) }])
      setTrainingProgress(epoch * 10)
      
      if (epoch >= 10) {
        clearInterval(interval)
        setIsTraining(false)
      }
    }, 500)
  }

  const pythonCode = `# Fine-Tuning & Transfer Learning
# Customize pre-trained models for your specific task!

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM
)
from peft import LoraConfig, get_peft_model, TaskType

# ============================================
# Understanding Fine-Tuning
# ============================================

"""
Transfer Learning: Use knowledge from one task for another
Fine-Tuning: Adjust a pre-trained model for your specific task

Why fine-tune?
1. Faster training (start from learned features)
2. Less data needed (model already knows language)
3. Better performance (leverage pre-trained knowledge)

Types of fine-tuning:
1. Full fine-tuning: Update all parameters
2. Feature extraction: Freeze base, train new layers
3. LoRA/PEFT: Update only small adapter layers
"""

# ============================================
# Transfer Learning with PyTorch
# ============================================

from torchvision import models, transforms

def create_transfer_model(num_classes):
    """
    Use a pre-trained ResNet for your own classification task
    """
    # Load pre-trained model
    model = models.resnet50(pretrained=True)
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model

# Example: Fine-tune for 5 classes
model = create_transfer_model(num_classes=5)

# Only the new layers will be trained
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

# ============================================
# Fine-Tuning BERT for Classification
# ============================================

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx])
        }

def fine_tune_bert(train_texts, train_labels, num_labels=2):
    """
    Fine-tune BERT for text classification
    """
    # Load pre-trained model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    # Create dataset
    train_dataset = TextClassificationDataset(
        train_texts, train_labels, tokenizer
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    # Train!
    trainer.train()
    
    return model, tokenizer

# ============================================
# LoRA: Low-Rank Adaptation (Efficient Fine-Tuning)
# ============================================

def setup_lora_model(model_name="meta-llama/Llama-2-7b-hf"):
    """
    LoRA: Train only small adapter matrices
    
    Benefits:
    - 10-100x fewer trainable parameters
    - Much faster training
    - Can store multiple adapters for one base model
    """
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,                    # Rank of update matrices
        lora_alpha=32,          # Scaling factor
        lora_dropout=0.1,       # Dropout for regularization
        target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Check trainable parameters
    model.print_trainable_parameters()
    # Output: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06%
    
    return model

# ============================================
# QLoRA: Quantized LoRA (Even More Efficient)
# ============================================

from transformers import BitsAndBytesConfig

def setup_qlora_model(model_name):
    """
    QLoRA: LoRA + 4-bit quantization
    
    Can fine-tune 65B models on a single GPU!
    """
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load quantized model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Apply LoRA on top
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    return model

# ============================================
# Fine-Tuning with Custom Dataset
# ============================================

from datasets import load_dataset

def prepare_instruction_dataset():
    """
    Prepare data for instruction fine-tuning
    """
    # Load dataset
    dataset = load_dataset("databricks/dolly-15k")
    
    def format_instruction(example):
        """Format as instruction-response pairs"""
        if example["context"]:
            prompt = f"""### Instruction:
{example["instruction"]}

### Context:
{example["context"]}

### Response:
{example["response"]}"""
        else:
            prompt = f"""### Instruction:
{example["instruction"]}

### Response:
{example["response"]}"""
        return {"text": prompt}
    
    dataset = dataset.map(format_instruction)
    return dataset

# ============================================
# Training Loop for Fine-Tuning
# ============================================

from trl import SFTTrainer

def train_with_sft(model, tokenizer, dataset):
    """
    Supervised Fine-Tuning with TRL library
    """
    training_args = TrainingArguments(
        output_dir="./sft_output",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
    )
    
    trainer.train()
    
    # Save the fine-tuned model
    trainer.save_model("./fine_tuned_model")

# ============================================
# Merging LoRA Weights
# ============================================

def merge_and_save(model, output_path):
    """
    Merge LoRA weights into base model for deployment
    """
    # Merge LoRA weights
    merged_model = model.merge_and_unload()
    
    # Save merged model
    merged_model.save_pretrained(output_path)
    
    print(f"Merged model saved to {output_path}")

# ============================================
# Using Fine-Tuned Model
# ============================================

def generate_with_finetuned(model, tokenizer, prompt):
    """
    Generate text with your fine-tuned model
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example usage
prompt = "### Instruction:\\nExplain quantum computing\\n\\n### Response:\\n"
response = generate_with_finetuned(model, tokenizer, prompt)`

  return (
    <div className="space-y-8">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-white mb-4">Fine-Tuning & Transfer Learning</h1>
        <p className="text-xl text-purple-200">Customize AI models for your specific needs!</p>
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
                <Settings className="w-6 h-6 text-violet-400" />
                What is Fine-Tuning?
              </CardTitle>
            </CardHeader>
            <CardContent className="text-slate-300 space-y-4">
              <p className="text-lg">
                <strong className="text-violet-400">Fine-tuning</strong> is the process of taking a pre-trained 
                model and adapting it for your specific task. Instead of training from scratch, you start with 
                a model that already knows a lot and teach it your specific use case!
              </p>
              <div className="bg-slate-900/50 p-4 rounded-lg border border-violet-500/30">
                <h4 className="text-violet-400 font-semibold mb-2">Think of it like this:</h4>
                <p>Imagine hiring a chef who already knows how to cook. Instead of teaching them everything 
                from scratch, you just teach them your restaurant's special recipes. That's fine-tuning - 
                starting with existing knowledge and specializing it!</p>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Types of Fine-Tuning</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-3 gap-4">
                {[
                  { 
                    name: 'Full Fine-Tuning', 
                    desc: 'Update all model parameters', 
                    pros: 'Best performance',
                    cons: 'Expensive, needs lots of data',
                    trainable: '100%',
                    color: 'bg-blue-500'
                  },
                  { 
                    name: 'Feature Extraction', 
                    desc: 'Freeze base, train new layers', 
                    pros: 'Fast, less data needed',
                    cons: 'Limited adaptation',
                    trainable: '~5%',
                    color: 'bg-green-500'
                  },
                  { 
                    name: 'LoRA/PEFT', 
                    desc: 'Train small adapter layers', 
                    pros: 'Very efficient, good results',
                    cons: 'Slightly less flexible',
                    trainable: '~0.1%',
                    color: 'bg-purple-500'
                  },
                ].map((type, i) => (
                  <div key={i} className="bg-slate-900 p-4 rounded-lg">
                    <div className={`w-10 h-10 ${type.color} rounded-lg flex items-center justify-center mb-3`}>
                      <Layers className="w-5 h-5 text-white" />
                    </div>
                    <h4 className="text-white font-semibold mb-1">{type.name}</h4>
                    <p className="text-slate-400 text-sm mb-2">{type.desc}</p>
                    <div className="space-y-1 text-xs">
                      <p className="text-green-400">+ {type.pros}</p>
                      <p className="text-red-400">- {type.cons}</p>
                      <Badge variant="outline" className="text-violet-400 mt-2">
                        {type.trainable} trainable
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">When to Use What?</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {[
                  { scenario: 'Lots of data, need best performance', method: 'Full Fine-Tuning', icon: '📊' },
                  { scenario: 'Limited data, quick results', method: 'Feature Extraction', icon: '⚡' },
                  { scenario: 'Large model, limited GPU', method: 'LoRA/QLoRA', icon: '💾' },
                  { scenario: 'Multiple tasks, one base model', method: 'LoRA Adapters', icon: '🔄' },
                ].map((item, i) => (
                  <div key={i} className="flex items-center gap-3 p-3 bg-slate-900 rounded-lg">
                    <span className="text-2xl">{item.icon}</span>
                    <div className="flex-1">
                      <p className="text-slate-300 text-sm">{item.scenario}</p>
                    </div>
                    <ArrowRight className="w-4 h-4 text-slate-500" />
                    <Badge className="bg-violet-500">{item.method}</Badge>
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
                <Zap className="w-5 h-5 text-violet-400" />
                Training Simulation
              </CardTitle>
              <CardDescription className="text-slate-400">
                Watch how loss decreases during fine-tuning!
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button 
                onClick={simulateTraining} 
                disabled={isTraining}
                className="w-full bg-violet-600 hover:bg-violet-700"
              >
                {isTraining ? `Training... ${trainingProgress}%` : 'Start Fine-Tuning Simulation'}
              </Button>

              <div className="bg-slate-900 rounded-lg p-4">
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={lossData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="epoch" stroke="#9ca3af" label={{ value: 'Epoch', position: 'bottom', fill: '#9ca3af' }} />
                      <YAxis stroke="#9ca3af" label={{ value: 'Loss', angle: -90, position: 'left', fill: '#9ca3af' }} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569' }}
                        labelStyle={{ color: '#e2e8f0' }}
                      />
                      <Line type="monotone" dataKey="loss" stroke="#8b5cf6" strokeWidth={2} dot={{ fill: '#8b5cf6' }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <p className="text-slate-400 text-sm text-center mt-2">
                  Loss decreases as the model learns your specific task!
                </p>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">LoRA Architecture</CardTitle>
              <CardDescription className="text-slate-400">
                How LoRA adds small trainable matrices
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="bg-slate-900 p-6 rounded-lg">
                <div className="flex items-center justify-center gap-8">
                  <div className="text-center">
                    <div className="w-24 h-32 bg-blue-500/20 border-2 border-blue-500 rounded-lg flex items-center justify-center mb-2">
                      <span className="text-blue-400 font-mono text-sm">W</span>
                    </div>
                    <p className="text-slate-400 text-xs">Original Weights</p>
                    <p className="text-blue-400 text-xs">(Frozen)</p>
                  </div>
                  
                  <span className="text-slate-400 text-2xl">+</span>
                  
                  <div className="text-center">
                    <div className="flex gap-1 mb-2">
                      <div className="w-8 h-32 bg-purple-500/20 border-2 border-purple-500 rounded-lg flex items-center justify-center">
                        <span className="text-purple-400 font-mono text-xs">A</span>
                      </div>
                      <div className="w-24 h-8 bg-purple-500/20 border-2 border-purple-500 rounded-lg flex items-center justify-center self-center">
                        <span className="text-purple-400 font-mono text-xs">B</span>
                      </div>
                    </div>
                    <p className="text-slate-400 text-xs">LoRA Adapters</p>
                    <p className="text-purple-400 text-xs">(Trainable)</p>
                  </div>
                  
                  <span className="text-slate-400 text-2xl">=</span>
                  
                  <div className="text-center">
                    <div className="w-24 h-32 bg-green-500/20 border-2 border-green-500 rounded-lg flex items-center justify-center mb-2">
                      <span className="text-green-400 font-mono text-sm">W'</span>
                    </div>
                    <p className="text-slate-400 text-xs">Adapted Weights</p>
                    <p className="text-green-400 text-xs">(Fine-tuned)</p>
                  </div>
                </div>
                <p className="text-slate-400 text-sm text-center mt-4">
                  LoRA decomposes weight updates into low-rank matrices, reducing trainable parameters by 1000x!
                </p>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Parameter Comparison</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {[
                  { method: 'Full Fine-Tuning', params: 7000000000, trainable: 7000000000 },
                  { method: 'Feature Extraction', params: 7000000000, trainable: 350000000 },
                  { method: 'LoRA (r=8)', params: 7000000000, trainable: 4194304 },
                  { method: 'QLoRA (4-bit)', params: 7000000000, trainable: 4194304 },
                ].map((item, i) => (
                  <div key={i} className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span className="text-white">{item.method}</span>
                      <span className="text-violet-400">{(item.trainable / 1000000).toFixed(1)}M trainable</span>
                    </div>
                    <div className="h-4 bg-slate-700 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-gradient-to-r from-violet-500 to-purple-500"
                        style={{ width: `${Math.max((item.trainable / item.params) * 100, 1)}%` }}
                      />
                    </div>
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
                <Code className="w-5 h-5 text-violet-400" />
                Python Code: Fine-Tuning Models
              </CardTitle>
              <CardDescription className="text-slate-400">
                From transfer learning to LoRA and QLoRA!
              </CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="bg-slate-900 p-4 rounded-lg overflow-x-auto text-sm max-h-96 overflow-y-auto">
                <code className="text-green-400">{pythonCode}</code>
              </pre>
              <div className="mt-4 p-4 bg-violet-500/10 rounded-lg border border-violet-500/30">
                <h4 className="text-violet-400 font-semibold mb-2">Required Libraries:</h4>
                <code className="text-slate-300 text-sm bg-slate-800 px-2 py-1 rounded">
                  pip install transformers peft trl bitsandbytes datasets accelerate
                </code>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default Module12FineTuning
