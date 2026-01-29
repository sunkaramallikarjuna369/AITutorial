"""
Module 10: Prompt Engineering
=============================
The art of talking to AI effectively!
"""

def explain_prompt_engineering():
    print("=" * 60)
    print("Prompt Engineering")
    print("=" * 60)
    print("""
Prompt Engineering is crafting inputs to get better AI outputs!

Key Techniques:

1. ZERO-SHOT: Just ask directly
   "Translate 'Hello' to French"

2. FEW-SHOT: Give examples first
   "English: Hello -> French: Bonjour
    English: Goodbye -> French: Au revoir
    English: Thank you -> French: ?"

3. CHAIN-OF-THOUGHT: Ask to think step by step
   "Solve this problem step by step..."

4. ROLE PROMPTING: Assign a persona
   "You are an expert Python developer..."

5. STRUCTURED OUTPUT: Request specific format
   "Return your answer as JSON with fields: name, age, city"

Best Practices:
- Be specific and clear
- Provide context
- Use examples when helpful
- Break complex tasks into steps
- Iterate and refine
    """)

prompt_examples = '''
"""
Prompt Engineering Examples
===========================
"""

# ============================================
# 1. Zero-Shot Prompting
# ============================================

zero_shot = """
Classify the sentiment of this review as positive, negative, or neutral:

Review: "This product exceeded my expectations! Highly recommend."

Sentiment:
"""


# ============================================
# 2. Few-Shot Prompting
# ============================================

few_shot = """
Classify the sentiment of reviews:

Review: "Terrible quality, broke after one day."
Sentiment: negative

Review: "It's okay, nothing special."
Sentiment: neutral

Review: "Best purchase I've ever made!"
Sentiment: positive

Review: "The delivery was late but the product is decent."
Sentiment:
"""


# ============================================
# 3. Chain-of-Thought Prompting
# ============================================

chain_of_thought = """
Solve this problem step by step:

A store has 50 apples. They sell 23 apples in the morning and receive 
a shipment of 35 apples. Then they sell 18 more apples. How many 
apples do they have now?

Let's think step by step:
"""


# ============================================
# 4. Role Prompting
# ============================================

role_prompt = """
You are an experienced Python developer with 15 years of experience.
You write clean, efficient, and well-documented code.
You always follow PEP 8 style guidelines.

Task: Write a function to find the longest palindrome in a string.
"""


# ============================================
# 5. Structured Output
# ============================================

structured_output = """
Extract information from this text and return as JSON:

Text: "John Smith is a 35-year-old software engineer living in 
San Francisco. He has been working at Google for 5 years."

Return JSON with fields: name, age, occupation, city, company, years_employed
"""


# ============================================
# 6. System + User Messages
# ============================================

messages = [
    {
        "role": "system",
        "content": """You are a helpful coding assistant. 
        - Always explain your code
        - Include error handling
        - Follow best practices"""
    },
    {
        "role": "user",
        "content": "Write a function to read a CSV file safely"
    }
]


# ============================================
# 7. Prompt Templates
# ============================================

def create_prompt(task, context, examples=None):
    prompt = f"""
Task: {task}

Context: {context}
"""
    if examples:
        prompt += "\\nExamples:\\n"
        for ex in examples:
            prompt += f"- Input: {ex['input']}\\n  Output: {ex['output']}\\n"
    
    prompt += "\\nYour response:"
    return prompt


# ============================================
# 8. ReAct Prompting (Reasoning + Acting)
# ============================================

react_prompt = """
Answer the following question using this format:

Thought: [Your reasoning about what to do]
Action: [The action to take, e.g., Search, Calculate, Lookup]
Observation: [The result of the action]
... (repeat Thought/Action/Observation as needed)
Final Answer: [Your final answer]

Question: What is the capital of the country where the Eiffel Tower is located?
"""


# ============================================
# 9. Self-Consistency Prompting
# ============================================

def self_consistency(question, num_samples=5):
    """Generate multiple answers and pick the most common"""
    prompt = f"""
    {question}
    
    Think through this carefully and provide your answer.
    """
    
    # Generate multiple responses
    answers = []
    for _ in range(num_samples):
        response = generate(prompt, temperature=0.7)
        answers.append(extract_answer(response))
    
    # Return most common answer
    from collections import Counter
    return Counter(answers).most_common(1)[0][0]


# ============================================
# 10. Prompt Chaining
# ============================================

def analyze_and_improve_code(code):
    # Step 1: Analyze
    analysis_prompt = f"""
    Analyze this code for potential issues:
    
    ```python
    {code}
    ```
    
    List any bugs, inefficiencies, or style issues.
    """
    analysis = generate(analysis_prompt)
    
    # Step 2: Improve
    improve_prompt = f"""
    Original code:
    ```python
    {code}
    ```
    
    Issues found:
    {analysis}
    
    Provide improved code that fixes all issues.
    """
    improved = generate(improve_prompt)
    
    return improved


print("Prompt engineering examples loaded!")
'''

def main():
    explain_prompt_engineering()
    print("\nPrompt Examples:")
    print(prompt_examples)

if __name__ == "__main__":
    main()
