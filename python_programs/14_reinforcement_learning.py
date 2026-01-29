"""
=============================================================================
MODULE 14: REINFORCEMENT LEARNING - COMPREHENSIVE 360 DEGREE COVERAGE
=============================================================================

This module provides COMPLETE coverage of Reinforcement Learning including:
- 4W+H Explanations (What, Why, When, Where, How)
- Core Concepts (MDP, Policy, Value Functions)
- Algorithms (Q-Learning, DQN, PPO)
- RLHF (RL from Human Feedback)
- Interview Questions
- Common Pitfalls

SETUP:
------
pip install numpy gymnasium

=============================================================================
"""

import os
import random
from typing import List, Dict, Tuple, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# =============================================================================
# GENAI INTEGRATION
# =============================================================================

class RLAssistant:
    """Use GenAI models for RL explanations"""
    
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
    
    def explain_with_openai(self, concept: str) -> str:
        try:
            from openai import OpenAI
            if not self.openai_key:
                return "[Set OPENAI_API_KEY]"
            client = OpenAI(api_key=self.openai_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"Explain {concept} in reinforcement learning with examples."}],
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[Error: {e}]"
    
    def explain_with_ollama(self, concept: str, model: str = "llama2") -> str:
        try:
            import requests
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": f"Explain {concept} in RL.", "stream": False},
                timeout=60
            )
            return response.json().get("response", "") if response.status_code == 200 else "[Ollama error]"
        except Exception as e:
            return f"[Start Ollama: ollama serve. Error: {e}]"


# =============================================================================
# SECTION 1: COMPREHENSIVE RL EXPLANATION (4W+H)
# =============================================================================

def explain_rl_comprehensive():
    """Comprehensive 360-degree explanation of RL"""
    print("=" * 70)
    print("REINFORCEMENT LEARNING - COMPREHENSIVE 360 DEGREE COVERAGE")
    print("=" * 70)
    
    print("""
WHAT IS REINFORCEMENT LEARNING?
===============================

RL is learning through interaction with an environment, receiving rewards
or penalties for actions, and learning to maximize cumulative reward.

KEY COMPONENTS:
    Agent: The learner/decision maker
    Environment: What agent interacts with
    State (s): Current situation
    Action (a): What agent can do
    Reward (r): Feedback signal
    Policy (pi): Strategy for choosing actions

THE RL LOOP:
    State -> Agent -> Action -> Environment -> Reward, New State -> ...

COMPARISON TO OTHER ML:
    Supervised: Learn from labeled examples
    Unsupervised: Find patterns in data
    Reinforcement: Learn from trial and error

WHY IS RL IMPORTANT?
====================

1. SEQUENTIAL DECISIONS: Optimal actions over time
2. NO LABELS NEEDED: Learn from rewards
3. EXPLORATION: Discover novel strategies
4. REAL-WORLD PROBLEMS: Games, robotics, trading

WHEN TO USE RL?
===============

USE RL FOR:
    - Games (Chess, Go, Atari, video games)
    - Robotics (manipulation, navigation)
    - Recommendation systems
    - Resource allocation
    - Trading strategies
    - RLHF for LLMs

DON'T USE RL FOR:
    - Problems with clear labeled data (use supervised)
    - One-shot decisions (no sequential aspect)
    - When simulation is impossible

WHERE IS RL USED?
=================

FAMOUS EXAMPLES:
    - AlphaGo: Beat world champion at Go
    - OpenAI Five: Beat pro Dota 2 players
    - ChatGPT: RLHF for alignment
    - Robotics: Boston Dynamics, Tesla FSD

HOW DOES RL WORK?
=================

MARKOV DECISION PROCESS (MDP):
    - States S, Actions A, Rewards R
    - Transition: P(s'|s,a)
    - Discount factor: gamma (0-1)
    
    Goal: Maximize expected cumulative reward
    G = r_0 + gamma*r_1 + gamma^2*r_2 + ...

VALUE FUNCTIONS:
    V(s): Expected return starting from state s
    Q(s,a): Expected return taking action a in state s

POLICY:
    pi(a|s): Probability of action a in state s
    Goal: Find optimal policy pi*
    """)


# =============================================================================
# SECTION 2: CORE ALGORITHMS
# =============================================================================

def rl_algorithms():
    """Core RL algorithms"""
    print("\n" + "=" * 70)
    print("RL ALGORITHMS")
    print("=" * 70)
    
    print("""
1. Q-LEARNING (Value-Based)
   =========================
   Learn Q(s,a) directly, act greedily
   
   Update rule:
   Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
   
   - Off-policy (can learn from any data)
   - Tabular (discrete states/actions)

2. DEEP Q-NETWORK (DQN)
   =====================
   Q-learning with neural network
   
   Key innovations:
   - Experience replay (store and sample transitions)
   - Target network (stable targets)
   - Epsilon-greedy exploration
   
   Loss: (Q(s,a) - (r + gamma * max_a' Q_target(s',a')))^2

3. POLICY GRADIENT
   ================
   Directly optimize policy parameters
   
   REINFORCE:
   gradient = E[sum_t grad(log pi(a_t|s_t)) * G_t]
   
   - On-policy
   - High variance

4. ACTOR-CRITIC
   =============
   Combine value and policy learning
   
   Actor: Policy network (chooses actions)
   Critic: Value network (evaluates actions)
   
   Advantage: A(s,a) = Q(s,a) - V(s)

5. PPO (Proximal Policy Optimization)
   ===================================
   State-of-the-art policy gradient
   
   Key idea: Limit policy updates to prevent instability
   
   Clipped objective:
   L = min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)
   
   where r_t = pi_new(a|s) / pi_old(a|s)
   
   Used in: ChatGPT (RLHF), robotics

6. SAC (Soft Actor-Critic)
   ========================
   Maximum entropy RL
   
   Objective: Maximize reward + entropy
   
   Benefits: Better exploration, more robust

ALGORITHM COMPARISON:
=====================

Algorithm    | Type         | On/Off Policy | Best For
-------------|--------------|---------------|------------------
Q-Learning   | Value        | Off           | Simple, discrete
DQN          | Value        | Off           | Atari games
REINFORCE    | Policy       | On            | Simple continuous
A2C/A3C      | Actor-Critic | On            | Parallel training
PPO          | Actor-Critic | On            | General purpose
SAC          | Actor-Critic | Off           | Continuous control
    """)


# =============================================================================
# SECTION 3: SIMPLE Q-LEARNING DEMO
# =============================================================================

def q_learning_demo():
    """Demo Q-learning on simple grid world"""
    print("\n" + "=" * 70)
    print("Q-LEARNING DEMO: GRID WORLD")
    print("=" * 70)
    
    if not HAS_NUMPY:
        print("\nInstall numpy: pip install numpy")
        return
    
    print("""
Grid World:
    S . . .
    . X . .
    . . . .
    . . . G
    
    S = Start, G = Goal, X = Wall
    Actions: 0=Up, 1=Down, 2=Left, 3=Right
    Reward: -1 per step, +10 at goal
    """)
    
    # Simple 4x4 grid world
    grid_size = 4
    n_states = grid_size * grid_size
    n_actions = 4  # Up, Down, Left, Right
    
    # Initialize Q-table
    Q = np.zeros((n_states, n_actions))
    
    # Parameters
    alpha = 0.1  # Learning rate
    gamma = 0.99  # Discount factor
    epsilon = 0.1  # Exploration rate
    
    goal_state = 15  # Bottom-right
    wall_state = 5   # (1,1)
    
    def get_next_state(state, action):
        row, col = state // grid_size, state % grid_size
        if action == 0:  # Up
            row = max(0, row - 1)
        elif action == 1:  # Down
            row = min(grid_size - 1, row + 1)
        elif action == 2:  # Left
            col = max(0, col - 1)
        elif action == 3:  # Right
            col = min(grid_size - 1, col + 1)
        
        next_state = row * grid_size + col
        if next_state == wall_state:
            return state  # Can't move into wall
        return next_state
    
    def get_reward(state):
        if state == goal_state:
            return 10
        return -1
    
    # Training
    print("Training Q-learning agent...")
    for episode in range(1000):
        state = 0  # Start
        
        for step in range(50):
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, n_actions - 1)
            else:
                action = np.argmax(Q[state])
            
            # Take action
            next_state = get_next_state(state, action)
            reward = get_reward(next_state)
            
            # Q-learning update
            Q[state, action] += alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )
            
            state = next_state
            if state == goal_state:
                break
    
    print("\nLearned Q-values (best action per state):")
    actions = ['U', 'D', 'L', 'R']
    for row in range(grid_size):
        line = ""
        for col in range(grid_size):
            state = row * grid_size + col
            if state == goal_state:
                line += " G "
            elif state == wall_state:
                line += " X "
            else:
                best_action = actions[np.argmax(Q[state])]
                line += f" {best_action} "
        print(line)
    
    print("\nAgent learned to navigate to goal!")


# =============================================================================
# SECTION 4: RLHF (RL FROM HUMAN FEEDBACK)
# =============================================================================

def rlhf_explanation():
    """RLHF for LLMs"""
    print("\n" + "=" * 70)
    print("RLHF - RL FROM HUMAN FEEDBACK")
    print("=" * 70)
    
    print("""
WHAT IS RLHF?
=============

RLHF trains language models to align with human preferences using
reinforcement learning.

THE RLHF PIPELINE:
==================

STEP 1: SUPERVISED FINE-TUNING (SFT)
    - Fine-tune base model on high-quality demonstrations
    - Human-written examples of desired behavior

STEP 2: REWARD MODEL TRAINING
    - Generate multiple responses to same prompt
    - Humans rank responses (A > B > C)
    - Train reward model to predict human preferences
    
    Loss: -log(sigmoid(r(preferred) - r(rejected)))

STEP 3: RL FINE-TUNING (PPO)
    - Use reward model as reward signal
    - Fine-tune LLM with PPO
    - KL penalty to stay close to SFT model
    
    Reward = RM(response) - beta * KL(pi || pi_sft)

WHY RLHF?
=========

1. ALIGNMENT: Model follows instructions, is helpful
2. SAFETY: Reduces harmful outputs
3. QUALITY: Improves response quality
4. HUMAN VALUES: Captures nuanced preferences

CHALLENGES:
===========

1. REWARD HACKING: Model exploits reward model flaws
2. EXPENSIVE: Requires human labelers
3. DISTRIBUTION SHIFT: Reward model trained on different data
4. KL COLLAPSE: Model becomes too similar to base

ALTERNATIVES TO RLHF:
=====================

DPO (Direct Preference Optimization):
    - Skip reward model
    - Directly optimize from preferences
    - Simpler, often works as well

Constitutional AI (CAI):
    - AI critiques and revises its own outputs
    - Less human labeling needed

RLAIF (RL from AI Feedback):
    - Use AI to generate preferences
    - Scalable but may inherit AI biases
    """)


# =============================================================================
# SECTION 5: INTERVIEW QUESTIONS
# =============================================================================

def interview_questions():
    """Common RL interview questions"""
    print("\n" + "=" * 70)
    print("RL INTERVIEW QUESTIONS")
    print("=" * 70)
    
    questions = [
        ("What is the exploration-exploitation tradeoff?",
         "Exploration: Try new actions to discover better strategies. Exploitation: Use known good actions. Balance with epsilon-greedy, UCB, or entropy bonus."),
        ("What is the difference between on-policy and off-policy?",
         "On-policy: Learn from actions taken by current policy (SARSA, PPO). Off-policy: Learn from any data, including old policies (Q-learning, DQN)."),
        ("Why is PPO popular for RLHF?",
         "Stable training (clipped objective), sample efficient, works well with neural networks, handles continuous actions. Good balance of simplicity and performance."),
        ("What is reward shaping?",
         "Adding intermediate rewards to guide learning. Risk: Can lead to unintended behavior if not careful. Better: Use potential-based shaping."),
        ("How does DQN handle continuous state spaces?",
         "Uses neural network to approximate Q(s,a). Experience replay and target networks stabilize training."),
    ]
    
    for i, (q, a) in enumerate(questions, 1):
        print(f"\nQ{i}: {q}")
        print(f"A: {a}")


# =============================================================================
# SECTION 6: COMMON PITFALLS
# =============================================================================

def common_pitfalls():
    """Common RL mistakes"""
    print("\n" + "=" * 70)
    print("COMMON RL PITFALLS")
    print("=" * 70)
    
    pitfalls = [
        ("Sparse rewards", "Use reward shaping, curiosity, or hierarchical RL"),
        ("Reward hacking", "Carefully design rewards; use multiple objectives"),
        ("Sample inefficiency", "Use off-policy methods, model-based RL, or demonstrations"),
        ("Hyperparameter sensitivity", "Use PPO (more robust); tune carefully"),
        ("Non-stationary environments", "Use experience replay, target networks"),
        ("Ignoring safety", "Add constraints, use safe RL methods"),
    ]
    
    for mistake, solution in pitfalls:
        print(f"\nMistake: {mistake}")
        print(f"Solution: {solution}")


# =============================================================================
# MAIN PROGRAM
# =============================================================================

def main():
    print("=" * 70)
    print("   MODULE 14: REINFORCEMENT LEARNING - COMPREHENSIVE COVERAGE")
    print("=" * 70)
    
    assistant = RLAssistant()
    
    while True:
        print("\n" + "-" * 70)
        print("Choose a topic:")
        print()
        print("FUNDAMENTALS:")
        print("  1. What is RL? (4W+H)")
        print("  2. RL Algorithms")
        print("  3. Q-Learning Demo")
        print("  4. RLHF for LLMs")
        print()
        print("DEEP DIVE:")
        print("  5. Interview Questions")
        print("  6. Common Pitfalls")
        print()
        print("GENAI FEATURES:")
        print("  7. AI Explanation (OpenAI)")
        print("  8. AI Explanation (Ollama - FREE)")
        print()
        print("  9. Run ALL Topics")
        print("  0. Exit")
        print("-" * 70)
        
        choice = input("\nEnter choice (0-9): ").strip()
        
        if choice == "0":
            print("\nHappy learning!")
            break
        elif choice == "1":
            explain_rl_comprehensive()
        elif choice == "2":
            rl_algorithms()
        elif choice == "3":
            q_learning_demo()
        elif choice == "4":
            rlhf_explanation()
        elif choice == "5":
            interview_questions()
        elif choice == "6":
            common_pitfalls()
        elif choice == "7":
            concept = input("Enter concept to explain: ").strip()
            print(assistant.explain_with_openai(concept))
        elif choice == "8":
            concept = input("Enter concept to explain: ").strip()
            print(assistant.explain_with_ollama(concept))
        elif choice == "9":
            explain_rl_comprehensive()
            rl_algorithms()
            q_learning_demo()
            rlhf_explanation()
            interview_questions()
            common_pitfalls()
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
