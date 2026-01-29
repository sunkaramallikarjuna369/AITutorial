"""
=============================================================================
MODULE 10: PROMPT ENGINEERING - COMPREHENSIVE 360 DEGREE COVERAGE
=============================================================================

This module provides COMPLETE coverage of Prompt Engineering including:
- 4W+H Explanations (What, Why, When, Where, How)
- Prompting Techniques (Zero-shot, Few-shot, CoT, etc.)
- Best Practices
- Real Examples with Multiple GenAI Models
- Interview Questions
- Common Pitfalls

SETUP:
------
For GenAI features:
pip install openai anthropic google-generativeai

=============================================================================
"""

import os
from typing import List, Dict, Optional

# =============================================================================
# GENAI INTEGRATION
# =============================================================================

class PromptEngineer:
    """
    Demonstrate prompt engineering with multiple GenAI models
    
    Supports: OpenAI, Anthropic Claude, Google Gemini, Ollama (local)
    """
    
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.google_key = os.getenv("GOOGLE_API_KEY")
    
    def call_openai(self, prompt: str, system: str = None, model: str = "gpt-3.5-turbo") -> str:
        """Call OpenAI API"""
        try:
            from openai import OpenAI
            if not self.openai_key:
                return "[Set OPENAI_API_KEY]"
            
            client = OpenAI(api_key=self.openai_key)
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=500
            )
            return response.choices[0].message.content
        except ImportError:
            return "[Install openai: pip install openai]"
        except Exception as e:
            return f"[OpenAI Error: {e}]"
    
    def call_claude(self, prompt: str, system: str = None) -> str:
        """Call Anthropic Claude API"""
        try:
            import anthropic
            if not self.anthropic_key:
                return "[Set ANTHROPIC_API_KEY]"
            
            client = anthropic.Anthropic(api_key=self.anthropic_key)
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=500,
                system=system or "You are a helpful assistant.",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except ImportError:
            return "[Install anthropic: pip install anthropic]"
        except Exception as e:
            return f"[Claude Error: {e}]"
    
    def call_gemini(self, prompt: str) -> str:
        """Call Google Gemini API"""
        try:
            import google.generativeai as genai
            if not self.google_key:
                return "[Set GOOGLE_API_KEY]"
            
            genai.configure(api_key=self.google_key)
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            return response.text
        except ImportError:
            return "[Install google-generativeai: pip install google-generativeai]"
        except Exception as e:
            return f"[Gemini Error: {e}]"
    
    def call_ollama(self, prompt: str, model: str = "llama2") -> str:
        """Call Ollama (FREE, local)"""
        try:
            import requests
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=60
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            return f"[Ollama error: {response.status_code}]"
        except Exception as e:
            return f"[Start Ollama: ollama serve. Error: {e}]"


# =============================================================================
# SECTION 1: COMPREHENSIVE PROMPT ENGINEERING EXPLANATION (4W+H)
# =============================================================================

def explain_prompt_engineering_comprehensive():
    """Comprehensive 360-degree explanation of Prompt Engineering"""
    print("=" * 70)
    print("PROMPT ENGINEERING - COMPREHENSIVE 360 DEGREE COVERAGE")
    print("=" * 70)
    
    print("""
WHAT IS PROMPT ENGINEERING?
===========================

Prompt Engineering is the art and science of crafting inputs (prompts)
to get optimal outputs from AI language models.

KEY INSIGHT:
    The same model can give vastly different results based on how you ask!
    
    Bad prompt: "Write code"
    Good prompt: "Write a Python function that takes a list of integers
                 and returns the sum of all even numbers. Include docstring
                 and type hints. Handle empty list edge case."

WHY IS PROMPT ENGINEERING IMPORTANT?
====================================

1. QUALITY: Better prompts = better outputs
2. CONSISTENCY: Structured prompts = consistent results
3. EFFICIENCY: Good prompts reduce iterations
4. COST: Fewer tokens = lower API costs
5. SAFETY: Proper prompts reduce harmful outputs

WHEN TO USE DIFFERENT TECHNIQUES?
=================================

ZERO-SHOT: Simple, well-defined tasks
    "Translate 'Hello' to French"

FEW-SHOT: Tasks needing specific format or style
    "Here are examples... Now do this..."

CHAIN-OF-THOUGHT: Complex reasoning, math, logic
    "Let's think step by step..."

ROLE PROMPTING: Specialized knowledge needed
    "You are an expert cardiologist..."

WHERE IS PROMPT ENGINEERING USED?
=================================

- Chatbots and virtual assistants
- Code generation (GitHub Copilot)
- Content creation (marketing, writing)
- Data extraction and analysis
- Customer support automation
- Education and tutoring

HOW TO ENGINEER EFFECTIVE PROMPTS?
==================================

1. BE SPECIFIC: Clear, detailed instructions
2. PROVIDE CONTEXT: Background information
3. USE EXAMPLES: Show desired format
4. STRUCTURE OUTPUT: Request specific format (JSON, markdown)
5. ITERATE: Test and refine prompts
6. USE SYSTEM PROMPTS: Set behavior and constraints
    """)


# =============================================================================
# SECTION 2: PROMPTING TECHNIQUES
# =============================================================================

def prompting_techniques():
    """Detailed explanation of prompting techniques"""
    print("\n" + "=" * 70)
    print("PROMPTING TECHNIQUES")
    print("=" * 70)
    
    techniques = [
        {
            "name": "Zero-Shot Prompting",
            "description": "Ask directly without examples",
            "when": "Simple, well-defined tasks",
            "example": '''Classify the sentiment as positive, negative, or neutral:
"This product is amazing!" 
Sentiment:'''
        },
        {
            "name": "Few-Shot Prompting",
            "description": "Provide examples before the task",
            "when": "Need specific format or style",
            "example": '''Classify sentiment:
"Great product!" -> positive
"Terrible service" -> negative
"It's okay" -> neutral
"Best purchase ever!" ->'''
        },
        {
            "name": "Chain-of-Thought (CoT)",
            "description": "Ask model to reason step by step",
            "when": "Complex reasoning, math, logic",
            "example": '''Solve step by step:
A store has 50 apples. They sell 23, receive 35, then sell 18.
How many apples remain?

Let's think step by step:'''
        },
        {
            "name": "Role Prompting",
            "description": "Assign a persona or expertise",
            "when": "Need specialized knowledge",
            "example": '''You are an expert Python developer with 15 years experience.
You write clean, efficient, well-documented code following PEP 8.

Write a function to find the longest palindrome in a string.'''
        },
        {
            "name": "Structured Output",
            "description": "Request specific output format",
            "when": "Need parseable, consistent output",
            "example": '''Extract information and return as JSON:
"John Smith, 35, software engineer in San Francisco"

Return: {"name": "", "age": 0, "job": "", "city": ""}'''
        },
        {
            "name": "ReAct (Reasoning + Acting)",
            "description": "Interleave reasoning and actions",
            "when": "Tasks requiring tool use or multi-step reasoning",
            "example": '''Answer using this format:
Thought: [reasoning]
Action: [Search/Calculate/Lookup]
Observation: [result]
Final Answer: [answer]

Question: What is the population of the capital of France?'''
        },
        {
            "name": "Self-Consistency",
            "description": "Generate multiple answers, pick most common",
            "when": "Need high confidence answers",
            "example": "Generate 5 answers with temperature=0.7, return majority"
        },
        {
            "name": "Tree of Thoughts",
            "description": "Explore multiple reasoning paths",
            "when": "Complex problems with multiple solutions",
            "example": "Consider 3 different approaches, evaluate each, choose best"
        },
    ]
    
    for tech in techniques:
        print(f"\n{'='*60}")
        print(f"{tech['name']}")
        print(f"{'='*60}")
        print(f"Description: {tech['description']}")
        print(f"When to use: {tech['when']}")
        print(f"\nExample:\n{tech['example']}")


# =============================================================================
# SECTION 3: BEST PRACTICES
# =============================================================================

def best_practices():
    """Prompt engineering best practices"""
    print("\n" + "=" * 70)
    print("PROMPT ENGINEERING BEST PRACTICES")
    print("=" * 70)
    
    print("""
1. BE SPECIFIC AND CLEAR
   Bad:  "Write about dogs"
   Good: "Write a 200-word informative paragraph about Golden Retrievers,
         covering their temperament, exercise needs, and suitability
         as family pets."

2. PROVIDE CONTEXT
   Bad:  "Fix this code"
   Good: "This Python function should calculate factorial but returns
         wrong results for n>10. Find and fix the bug:
         [code here]"

3. USE DELIMITERS
   Use ```, ---, or XML tags to separate sections:
   
   <context>
   Background information here
   </context>
   
   <task>
   What you want the model to do
   </task>

4. SPECIFY OUTPUT FORMAT
   "Return your answer as a JSON object with fields:
    - summary (string, max 100 words)
    - key_points (array of 3-5 strings)
    - sentiment (positive/negative/neutral)"

5. USE SYSTEM PROMPTS EFFECTIVELY
   System: "You are a helpful coding assistant. Always:
           - Explain your code
           - Include error handling
           - Follow best practices
           - Ask clarifying questions if needed"

6. BREAK COMPLEX TASKS INTO STEPS
   Instead of: "Analyze this data and create a report"
   Do:
   Step 1: "Summarize the key statistics"
   Step 2: "Identify trends and patterns"
   Step 3: "Generate recommendations"
   Step 4: "Format as a report"

7. USE NEGATIVE EXAMPLES
   "Do NOT include:
    - Personal opinions
    - Unverified claims
    - Technical jargon without explanation"

8. SET CONSTRAINTS
   "Your response must:
    - Be under 200 words
    - Use simple language (8th grade level)
    - Include at least 2 examples"
    """)


# =============================================================================
# SECTION 4: LIVE DEMOS
# =============================================================================

def live_demo_zero_shot(engineer: PromptEngineer):
    """Demo zero-shot prompting"""
    print("\n" + "=" * 70)
    print("LIVE DEMO: Zero-Shot Prompting")
    print("=" * 70)
    
    prompt = """Classify the sentiment of this review as positive, negative, or neutral:

Review: "This product exceeded my expectations! The quality is outstanding and customer service was incredibly helpful. Highly recommend!"

Sentiment:"""
    
    print(f"Prompt:\n{prompt}\n")
    print("Response (OpenAI):")
    print(engineer.call_openai(prompt))


def live_demo_few_shot(engineer: PromptEngineer):
    """Demo few-shot prompting"""
    print("\n" + "=" * 70)
    print("LIVE DEMO: Few-Shot Prompting")
    print("=" * 70)
    
    prompt = """Classify the sentiment of reviews:

Review: "Terrible quality, broke after one day."
Sentiment: negative

Review: "It's okay, nothing special."
Sentiment: neutral

Review: "Best purchase I've ever made!"
Sentiment: positive

Review: "The delivery was late but the product itself is decent."
Sentiment:"""
    
    print(f"Prompt:\n{prompt}\n")
    print("Response (OpenAI):")
    print(engineer.call_openai(prompt))


def live_demo_cot(engineer: PromptEngineer):
    """Demo chain-of-thought prompting"""
    print("\n" + "=" * 70)
    print("LIVE DEMO: Chain-of-Thought Prompting")
    print("=" * 70)
    
    prompt = """Solve this problem step by step:

A store has 50 apples. They sell 23 apples in the morning and receive 
a shipment of 35 apples. Then they sell 18 more apples. How many 
apples do they have now?

Let's think step by step:"""
    
    print(f"Prompt:\n{prompt}\n")
    print("Response (OpenAI):")
    print(engineer.call_openai(prompt))


def live_demo_role(engineer: PromptEngineer):
    """Demo role prompting"""
    print("\n" + "=" * 70)
    print("LIVE DEMO: Role Prompting")
    print("=" * 70)
    
    system = """You are an experienced Python developer with 15 years of experience.
You write clean, efficient, and well-documented code.
You always follow PEP 8 style guidelines and include type hints."""
    
    prompt = "Write a function to check if a string is a valid palindrome, ignoring spaces and punctuation."
    
    print(f"System:\n{system}\n")
    print(f"Prompt:\n{prompt}\n")
    print("Response (OpenAI):")
    print(engineer.call_openai(prompt, system=system))


# =============================================================================
# SECTION 5: INTERVIEW QUESTIONS
# =============================================================================

def interview_questions():
    """Common prompt engineering interview questions"""
    print("\n" + "=" * 70)
    print("PROMPT ENGINEERING INTERVIEW QUESTIONS")
    print("=" * 70)
    
    questions = [
        ("What is the difference between zero-shot and few-shot prompting?",
         "Zero-shot: No examples, just instructions. Few-shot: Provide examples before the task. Few-shot helps when you need specific format or style."),
        ("When would you use chain-of-thought prompting?",
         "For complex reasoning, math problems, or multi-step logic. It helps the model 'show its work' and reduces errors."),
        ("How do you reduce hallucinations in LLM outputs?",
         "Use RAG (retrieval), ask for citations, use lower temperature, provide factual context, ask model to say 'I don't know' when uncertain."),
        ("What is prompt injection and how do you prevent it?",
         "Malicious input that overrides system instructions. Prevent with: input validation, separate user/system content, output filtering."),
        ("How do you optimize prompts for cost?",
         "Be concise, use shorter examples, cache common prompts, use smaller models for simple tasks, batch requests."),
    ]
    
    for i, (q, a) in enumerate(questions, 1):
        print(f"\nQ{i}: {q}")
        print(f"A: {a}")


# =============================================================================
# SECTION 6: COMMON PITFALLS
# =============================================================================

def common_pitfalls():
    """Common prompt engineering mistakes"""
    print("\n" + "=" * 70)
    print("COMMON PROMPT ENGINEERING PITFALLS")
    print("=" * 70)
    
    pitfalls = [
        ("Vague instructions", "Be specific about what you want, format, length, style"),
        ("No examples for complex tasks", "Use few-shot prompting for non-trivial tasks"),
        ("Ignoring system prompts", "System prompts set behavior and constraints effectively"),
        ("Not iterating", "Test prompts, analyze failures, refine iteratively"),
        ("Overcomplicating prompts", "Start simple, add complexity only if needed"),
        ("Not handling edge cases", "Test with unusual inputs, empty strings, etc."),
    ]
    
    for mistake, solution in pitfalls:
        print(f"\nMistake: {mistake}")
        print(f"Solution: {solution}")


# =============================================================================
# MAIN PROGRAM
# =============================================================================

def main():
    print("=" * 70)
    print("   MODULE 10: PROMPT ENGINEERING - COMPREHENSIVE 360 DEGREE COVERAGE")
    print("=" * 70)
    
    engineer = PromptEngineer()
    
    while True:
        print("\n" + "-" * 70)
        print("Choose a topic:")
        print()
        print("FUNDAMENTALS:")
        print("  1. What is Prompt Engineering? (4W+H)")
        print("  2. Prompting Techniques")
        print("  3. Best Practices")
        print()
        print("LIVE DEMOS:")
        print("  4. Zero-Shot Demo (OpenAI)")
        print("  5. Few-Shot Demo (OpenAI)")
        print("  6. Chain-of-Thought Demo (OpenAI)")
        print("  7. Role Prompting Demo (OpenAI)")
        print()
        print("DEEP DIVE:")
        print("  8. Interview Questions")
        print("  9. Common Pitfalls")
        print()
        print("CUSTOM:")
        print("  10. Try Your Own Prompt (OpenAI)")
        print("  11. Try Your Own Prompt (Ollama - FREE)")
        print()
        print("  12. Run ALL Topics")
        print("  0. Exit")
        print("-" * 70)
        
        choice = input("\nEnter choice (0-12): ").strip()
        
        if choice == "0":
            print("\nHappy prompting!")
            break
        elif choice == "1":
            explain_prompt_engineering_comprehensive()
        elif choice == "2":
            prompting_techniques()
        elif choice == "3":
            best_practices()
        elif choice == "4":
            live_demo_zero_shot(engineer)
        elif choice == "5":
            live_demo_few_shot(engineer)
        elif choice == "6":
            live_demo_cot(engineer)
        elif choice == "7":
            live_demo_role(engineer)
        elif choice == "8":
            interview_questions()
        elif choice == "9":
            common_pitfalls()
        elif choice == "10":
            prompt = input("Enter your prompt: ").strip()
            print("\nResponse:")
            print(engineer.call_openai(prompt))
        elif choice == "11":
            prompt = input("Enter your prompt: ").strip()
            print("\nResponse:")
            print(engineer.call_ollama(prompt))
        elif choice == "12":
            explain_prompt_engineering_comprehensive()
            prompting_techniques()
            best_practices()
            live_demo_zero_shot(engineer)
            live_demo_few_shot(engineer)
            live_demo_cot(engineer)
            live_demo_role(engineer)
            interview_questions()
            common_pitfalls()
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
