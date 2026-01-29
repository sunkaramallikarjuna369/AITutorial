"""
Module 14: Reinforcement Learning
=================================
Learn how AI learns from rewards - like training a dog!
"""

import random

def explain_rl():
    print("=" * 60)
    print("Reinforcement Learning")
    print("=" * 60)
    print("""
Reinforcement Learning = Learning from rewards and punishments!

Key Concepts:
- Agent: The learner (like a robot)
- Environment: The world it interacts with
- State: Current situation
- Action: What the agent can do
- Reward: Feedback (positive or negative)

The Goal: Maximize total reward over time!

Example: Training a dog
- State: Dog sees a ball
- Action: Fetch the ball
- Reward: Treat! (+1)
- Dog learns: Fetching = Good!

Famous RL Achievements:
- AlphaGo: Beat world champion at Go
- OpenAI Five: Beat pros at Dota 2
- AlphaStar: Grandmaster at StarCraft II
    """)

# ============================================
# Simple Q-Learning Agent
# ============================================

class QLearningAgent:
    """
    Q-Learning: Learn the value of state-action pairs
    
    Q(s,a) = Q(s,a) + α * (reward + γ * max(Q(s',a')) - Q(s,a))
    """
    
    def __init__(self, states, actions, learning_rate=0.1, discount=0.95, epsilon=0.1):
        self.q_table = {}
        for s in states:
            self.q_table[s] = {a: 0.0 for a in actions}
        
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.actions = actions
    
    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return max(self.q_table[state], key=self.q_table[state].get)
    
    def learn(self, state, action, reward, next_state):
        """Update Q-value"""
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())
        
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q


# ============================================
# Simple Grid World Environment
# ============================================

class GridWorld:
    """
    Simple 4x4 grid world
    
    S . . .
    . X . .
    . . . .
    . . . G
    
    S = Start, G = Goal, X = Obstacle
    """
    
    def __init__(self):
        self.size = 4
        self.start = (0, 0)
        self.goal = (3, 3)
        self.obstacle = (1, 1)
        self.state = self.start
    
    def reset(self):
        self.state = self.start
        return self.state
    
    def step(self, action):
        """Take action and return (next_state, reward, done)"""
        x, y = self.state
        
        if action == 'up':
            x = max(0, x - 1)
        elif action == 'down':
            x = min(self.size - 1, x + 1)
        elif action == 'left':
            y = max(0, y - 1)
        elif action == 'right':
            y = min(self.size - 1, y + 1)
        
        next_state = (x, y)
        
        # Check obstacle
        if next_state == self.obstacle:
            next_state = self.state
        
        self.state = next_state
        
        # Calculate reward
        if next_state == self.goal:
            return next_state, 10, True
        else:
            return next_state, -0.1, False
    
    def get_states(self):
        states = []
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) != self.obstacle:
                    states.append((i, j))
        return states


def train_agent():
    """Train Q-learning agent on grid world"""
    print("\n" + "=" * 60)
    print("Training Q-Learning Agent")
    print("=" * 60)
    
    env = GridWorld()
    actions = ['up', 'down', 'left', 'right']
    agent = QLearningAgent(env.get_states(), actions)
    
    episodes = 500
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 100:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            
            total_reward += reward
            state = next_state
            steps += 1
            
            if done:
                break
        
        if episode % 100 == 0:
            print(f"Episode {episode}: Steps={steps}, Reward={total_reward:.2f}")
    
    print("\nLearned Q-values for start state (0,0):")
    for action, value in agent.q_table[(0, 0)].items():
        print(f"  {action}: {value:.2f}")


code_examples = '''
"""
Advanced RL Examples
====================
"""

# ============================================
# Deep Q-Network (DQN)
# ============================================

import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
    
    def forward(self, x):
        return self.network(x)


# ============================================
# Stable Baselines3
# ============================================

from stable_baselines3 import PPO, DQN, A2C
import gymnasium as gym

# Create environment
env = gym.make("CartPole-v1")

# Train with PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Evaluate
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    if done:
        obs, _ = env.reset()


# ============================================
# Policy Gradient (REINFORCE)
# ============================================

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

def reinforce_update(policy, optimizer, rewards, log_probs):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + 0.99 * G
        returns.insert(0, G)
    
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    loss = 0
    for log_prob, G in zip(log_probs, returns):
        loss -= log_prob * G
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


print("RL examples loaded!")
print("Install: pip install stable-baselines3 gymnasium")
'''

def main():
    explain_rl()
    train_agent()
    print("\nAdvanced Code Examples:")
    print(code_examples)

if __name__ == "__main__":
    main()
