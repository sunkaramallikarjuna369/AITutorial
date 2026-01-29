"""
=============================================================================
MODULE 1: INTRODUCTION TO ARTIFICIAL INTELLIGENCE - 360 DEGREE COVERAGE
=============================================================================

This comprehensive module covers:
- What is AI and its history
- Types of AI (Narrow, General, Super)
- AI vs Machine Learning vs Deep Learning
- Working with multiple GenAI models (OpenAI, Claude, Gemini, Ollama, HuggingFace)
- Real-world applications and examples
- Hands-on executable code

SETUP INSTRUCTIONS:
-------------------
1. Install required packages:
   pip install openai anthropic google-generativeai transformers torch requests

2. Set up API keys (create a .env file or export):
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   export GOOGLE_API_KEY="your-google-key"

3. For Ollama (local models - FREE, no API key needed):
   - Install Ollama: https://ollama.ai
   - Run: ollama pull llama2
   - Run: ollama pull mistral

=============================================================================
"""

import random
import os
from typing import Optional, Dict, Any

# =============================================================================
# GENAI MODELS CLASS - Use Multiple AI Providers
# =============================================================================

class GenAIModels:
    """
    A unified interface to work with multiple GenAI providers.
    Supports: OpenAI, Anthropic Claude, Google Gemini, Ollama (local), HuggingFace
    """
    
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.google_key = os.getenv("GOOGLE_API_KEY")
    
    def use_openai(self, prompt: str, model: str = "gpt-3.5-turbo") -> str:
        """
        Use OpenAI's GPT models (GPT-4, GPT-3.5-turbo)
        
        Example:
            ai = GenAIModels()
            response = ai.use_openai("Explain AI to a 5 year old")
        """
        try:
            from openai import OpenAI
            
            if not self.openai_key:
                return "Set OPENAI_API_KEY environment variable first"
            
            client = OpenAI(api_key=self.openai_key)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            return response.choices[0].message.content
        except ImportError:
            return "Install openai: pip install openai"
        except Exception as e:
            return f"Error: {e}"
    
    def use_claude(self, prompt: str, model: str = "claude-3-sonnet-20240229") -> str:
        """
        Use Anthropic's Claude models (Claude 3 Opus, Sonnet, Haiku)
        
        Example:
            ai = GenAIModels()
            response = ai.use_claude("What is machine learning?")
        """
        try:
            import anthropic
            
            if not self.anthropic_key:
                return "Set ANTHROPIC_API_KEY environment variable first"
            
            client = anthropic.Anthropic(api_key=self.anthropic_key)
            response = client.messages.create(
                model=model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except ImportError:
            return "Install anthropic: pip install anthropic"
        except Exception as e:
            return f"Error: {e}"
    
    def use_gemini(self, prompt: str, model: str = "gemini-pro") -> str:
        """
        Use Google's Gemini models
        
        Example:
            ai = GenAIModels()
            response = ai.use_gemini("Describe neural networks")
        """
        try:
            import google.generativeai as genai
            
            if not self.google_key:
                return "Set GOOGLE_API_KEY environment variable first"
            
            genai.configure(api_key=self.google_key)
            model_instance = genai.GenerativeModel(model)
            response = model_instance.generate_content(prompt)
            return response.text
        except ImportError:
            return "Install google-generativeai: pip install google-generativeai"
        except Exception as e:
            return f"Error: {e}"
    
    def use_ollama(self, prompt: str, model: str = "llama2") -> str:
        """
        Use Ollama for FREE local AI models (no API key needed!)
        
        Setup:
        1. Install Ollama from https://ollama.ai
        2. Run: ollama pull llama2 (or mistral, codellama, phi)
        3. Ollama runs on localhost:11434
        
        Example:
            ai = GenAIModels()
            response = ai.use_ollama("Hello!", model="mistral")
        """
        try:
            import requests
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get("response", "No response")
            return f"Ollama error: {response.status_code}"
        except Exception as e:
            return f"Start Ollama first: ollama serve. Error: {e}"
    
    def use_huggingface(self, prompt: str, model: str = "gpt2") -> str:
        """
        Use HuggingFace Transformers for local inference
        
        Example:
            ai = GenAIModels()
            response = ai.use_huggingface("Once upon a time")
        """
        try:
            from transformers import pipeline
            
            generator = pipeline("text-generation", model=model)
            result = generator(prompt, max_length=100, num_return_sequences=1)
            return result[0]["generated_text"]
        except ImportError:
            return "Install transformers: pip install transformers torch"
        except Exception as e:
            return f"Error: {e}"


def demo_genai_models():
    """Interactive demo to test different GenAI models"""
    print("\n" + "=" * 60)
    print("GenAI Models Demo - Test Multiple AI Providers")
    print("=" * 60)
    
    ai = GenAIModels()
    
    print("""
Available models:
1. OpenAI (GPT-4, GPT-3.5) - Requires OPENAI_API_KEY
2. Claude (Anthropic) - Requires ANTHROPIC_API_KEY  
3. Gemini (Google) - Requires GOOGLE_API_KEY
4. Ollama (Local) - FREE, no API key needed!
5. HuggingFace (Local) - FREE, runs on your machine
""")
    
    choice = input("Choose a model (1-5): ").strip()
    prompt = input("Enter your prompt: ").strip()
    
    if not prompt:
        prompt = "Explain artificial intelligence in simple terms."
    
    print("\nGenerating response...")
    
    if choice == "1":
        response = ai.use_openai(prompt)
    elif choice == "2":
        response = ai.use_claude(prompt)
    elif choice == "3":
        response = ai.use_gemini(prompt)
    elif choice == "4":
        response = ai.use_ollama(prompt)
    elif choice == "5":
        response = ai.use_huggingface(prompt)
    else:
        response = "Invalid choice"
    
    print(f"\nAI Response:\n{response}")

# ============================================
# What is AI? - Simple Examples
# ============================================

def simple_chatbot():
    """
    A simple rule-based chatbot - the most basic form of AI!
    
    This shows how AI can respond to user input using patterns.
    """
    print("=" * 50)
    print("Simple AI Chatbot")
    print("=" * 50)
    
    responses = {
        "hello": "Hi there! I'm a simple AI chatbot!",
        "how are you": "I'm doing great! I'm just a program, but thanks for asking!",
        "what is ai": "AI stands for Artificial Intelligence - it's when computers learn to do smart things!",
        "bye": "Goodbye! Have a great day!",
        "help": "I can chat with you! Try saying 'hello', 'what is ai', or 'bye'",
    }
    
    print("\nType 'quit' to exit")
    print("-" * 50)
    
    while True:
        user_input = input("\nYou: ").lower().strip()
        
        if user_input == "quit":
            print("Bot: Goodbye!")
            break
        
        # Find a matching response
        response = None
        for key in responses:
            if key in user_input:
                response = responses[key]
                break
        
        if response:
            print(f"Bot: {response}")
        else:
            print("Bot: I don't understand that. Try 'help' for options!")


def number_guessing_ai():
    """
    AI that learns to guess your number!
    
    This demonstrates how AI can use feedback to improve.
    """
    print("\n" + "=" * 50)
    print("Number Guessing AI")
    print("=" * 50)
    print("\nThink of a number between 1 and 100.")
    print("I'll try to guess it! Tell me 'higher', 'lower', or 'correct'")
    print("-" * 50)
    
    low, high = 1, 100
    attempts = 0
    
    while low <= high:
        guess = (low + high) // 2
        attempts += 1
        
        print(f"\nAttempt {attempts}: Is your number {guess}?")
        feedback = input("Enter 'higher', 'lower', or 'correct': ").lower().strip()
        
        if feedback == "correct":
            print(f"\n🎉 I guessed it in {attempts} attempts!")
            print("This is called BINARY SEARCH - a smart algorithm!")
            break
        elif feedback == "higher":
            low = guess + 1
        elif feedback == "lower":
            high = guess - 1
        else:
            print("Please enter 'higher', 'lower', or 'correct'")
            attempts -= 1
    else:
        print("Hmm, something went wrong. Did you change your number?")


def pattern_recognition():
    """
    Simple pattern recognition - a key AI concept!
    
    AI learns to recognize patterns in data.
    """
    print("\n" + "=" * 50)
    print("Pattern Recognition Demo")
    print("=" * 50)
    
    # Simple patterns
    patterns = [
        ([2, 4, 6, 8], 10, "Adding 2 each time"),
        ([1, 2, 4, 8], 16, "Doubling each time"),
        ([1, 1, 2, 3, 5], 8, "Fibonacci - add previous two"),
        ([1, 4, 9, 16], 25, "Square numbers"),
    ]
    
    print("\nI'll show you number patterns and predict the next number!")
    print("-" * 50)
    
    for sequence, answer, explanation in patterns:
        print(f"\nPattern: {sequence}")
        print(f"AI Prediction: The next number is {answer}")
        print(f"Explanation: {explanation}")


def recommendation_system():
    """
    Simple recommendation system - like Netflix or YouTube!
    
    AI suggests things based on what you like.
    """
    print("\n" + "=" * 50)
    print("Simple Recommendation System")
    print("=" * 50)
    
    # Movie database with genres
    movies = {
        "action": ["The Avengers", "John Wick", "Mad Max", "Die Hard"],
        "comedy": ["The Hangover", "Superbad", "Bridesmaids", "Anchorman"],
        "sci-fi": ["Interstellar", "The Matrix", "Inception", "Blade Runner"],
        "animation": ["Toy Story", "Finding Nemo", "Shrek", "Frozen"],
    }
    
    print("\nWhat genre do you like?")
    print("Options: action, comedy, sci-fi, animation")
    
    genre = input("\nYour choice: ").lower().strip()
    
    if genre in movies:
        print(f"\n🎬 Based on your love for {genre}, I recommend:")
        for movie in random.sample(movies[genre], min(3, len(movies[genre]))):
            print(f"   - {movie}")
        print("\nThis is how Netflix and YouTube recommend content!")
    else:
        print("I don't know that genre. Try: action, comedy, sci-fi, or animation")


# ============================================
# Types of AI
# ============================================

def explain_ai_types():
    """
    Explain the different types of AI
    """
    print("\n" + "=" * 50)
    print("Types of Artificial Intelligence")
    print("=" * 50)
    
    ai_types = [
        {
            "name": "Narrow AI (Weak AI)",
            "description": "AI designed for one specific task",
            "examples": ["Siri/Alexa", "Chess computers", "Spam filters", "Face recognition"],
            "status": "EXISTS TODAY ✓"
        },
        {
            "name": "General AI (Strong AI)",
            "description": "AI that can do any intellectual task a human can",
            "examples": ["Robots from movies", "HAL 9000", "Data from Star Trek"],
            "status": "DOESN'T EXIST YET"
        },
        {
            "name": "Super AI",
            "description": "AI smarter than all humans combined",
            "examples": ["Skynet", "The Matrix machines"],
            "status": "SCIENCE FICTION"
        }
    ]
    
    for ai_type in ai_types:
        print(f"\n📌 {ai_type['name']}")
        print(f"   {ai_type['description']}")
        print(f"   Examples: {', '.join(ai_type['examples'])}")
        print(f"   Status: {ai_type['status']}")


def ai_vs_human():
    """
    Compare AI and human capabilities
    """
    print("\n" + "=" * 50)
    print("AI vs Humans - Who's Better?")
    print("=" * 50)
    
    comparisons = [
        ("Math calculations", "AI wins! 🤖", "Computers calculate billions of times per second"),
        ("Creativity", "Humans win! 👤", "Humans create truly original ideas"),
        ("Pattern recognition", "AI wins! 🤖", "AI can find patterns in huge datasets"),
        ("Common sense", "Humans win! 👤", "AI struggles with everyday reasoning"),
        ("Playing games", "It depends! 🤝", "AI beats humans at chess, but humans are better at creative games"),
        ("Learning from few examples", "Humans win! 👤", "Humans can learn from just one example"),
    ]
    
    for task, winner, explanation in comparisons:
        print(f"\n{task}:")
        print(f"   Winner: {winner}")
        print(f"   Why: {explanation}")


# ============================================
# Main Program
# ============================================

def main():
    """
    Main function to run all demos
    """
    print("\n" + "=" * 60)
    print("   MODULE 1: INTRODUCTION TO ARTIFICIAL INTELLIGENCE")
    print("   360-Degree Coverage with Multiple GenAI Models")
    print("=" * 60)
    
    while True:
        print("\n" + "-" * 60)
        print("Choose a demo:")
        print("1. Simple Chatbot (Rule-based AI)")
        print("2. Number Guessing AI (Binary Search)")
        print("3. Pattern Recognition")
        print("4. Recommendation System")
        print("5. Types of AI (Narrow, General, Super)")
        print("6. AI vs Humans Comparison")
        print("7. GenAI Models Demo (OpenAI, Claude, Gemini, Ollama)")
        print("8. Run All Basic Demos")
        print("0. Exit")
        print("-" * 60)
        
        choice = input("\nEnter your choice (0-8): ").strip()
        
        if choice == "0":
            print("\nThanks for learning about AI! Goodbye!")
            break
        elif choice == "1":
            simple_chatbot()
        elif choice == "2":
            number_guessing_ai()
        elif choice == "3":
            pattern_recognition()
        elif choice == "4":
            recommendation_system()
        elif choice == "5":
            explain_ai_types()
        elif choice == "6":
            ai_vs_human()
        elif choice == "7":
            demo_genai_models()
        elif choice == "8":
            pattern_recognition()
            explain_ai_types()
            ai_vs_human()
            recommendation_system()
        else:
            print("Invalid choice. Please enter 0-8.")


if __name__ == "__main__":
    main()
