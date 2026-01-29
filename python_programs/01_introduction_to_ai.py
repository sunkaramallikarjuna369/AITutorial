"""
Module 01: Introduction to Artificial Intelligence
===================================================
Learn the basics of AI - what it is and how it works!

This program demonstrates fundamental AI concepts that even kids can understand.
"""

import random

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
    print("\n" + "🤖" * 25)
    print("\n   WELCOME TO INTRODUCTION TO AI!")
    print("\n" + "🤖" * 25)
    
    while True:
        print("\n" + "-" * 50)
        print("Choose a demo:")
        print("1. Simple Chatbot")
        print("2. Number Guessing AI")
        print("3. Pattern Recognition")
        print("4. Recommendation System")
        print("5. Types of AI")
        print("6. AI vs Humans")
        print("7. Run All Demos")
        print("0. Exit")
        print("-" * 50)
        
        choice = input("\nEnter your choice (0-7): ").strip()
        
        if choice == "0":
            print("\nThanks for learning about AI! Goodbye! 👋")
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
            pattern_recognition()
            explain_ai_types()
            ai_vs_human()
            recommendation_system()
        else:
            print("Invalid choice. Please enter 0-7.")


if __name__ == "__main__":
    main()
