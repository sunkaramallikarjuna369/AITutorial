import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Gamepad2, Trophy, Brain, Code, ArrowUp, ArrowDown, ArrowLeft, ArrowRight } from 'lucide-react'

const Module14ReinforcementLearning = () => {
  const [agentPos, setAgentPos] = useState({ x: 0, y: 0 })
  const [score, setScore] = useState(0)
  const [moves, setMoves] = useState(0)
  const goalPos = { x: 3, y: 3 }
  const gridSize = 4

  const moveAgent = (direction: string) => {
    setAgentPos(prev => {
      let newX = prev.x
      let newY = prev.y
      
      if (direction === 'up' && prev.y > 0) newY--
      if (direction === 'down' && prev.y < gridSize - 1) newY++
      if (direction === 'left' && prev.x > 0) newX--
      if (direction === 'right' && prev.x < gridSize - 1) newX++
      
      const newPos = { x: newX, y: newY }
      setMoves(m => m + 1)
      
      if (newPos.x === goalPos.x && newPos.y === goalPos.y) {
        setScore(s => s + 100 - moves)
      }
      
      return newPos
    })
  }

  const resetGame = () => {
    setAgentPos({ x: 0, y: 0 })
    setScore(0)
    setMoves(0)
  }

  const pythonCode = `# Reinforcement Learning - Learning by Trial and Error!
# Teaching AI to make decisions through rewards

import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim

# ============================================
# Understanding RL Basics
# ============================================

"""
Key Concepts:
- Agent: The learner/decision maker
- Environment: The world the agent interacts with
- State: Current situation
- Action: What the agent can do
- Reward: Feedback signal (positive or negative)
- Policy: Strategy for choosing actions

Goal: Learn a policy that maximizes total reward!
"""

# ============================================
# Q-Learning (Classic Algorithm)
# ============================================

class QLearningAgent:
    """
    Q-Learning: Learn the value of state-action pairs
    
    Q(s, a) = expected future reward for taking action a in state s
    """
    
    def __init__(self, n_states, n_actions, learning_rate=0.1, 
                 discount=0.99, epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = learning_rate
        self.gamma = discount  # How much to value future rewards
        self.epsilon = epsilon  # Exploration rate
        self.n_actions = n_actions
    
    def choose_action(self, state):
        """
        Epsilon-greedy: Explore vs Exploit
        - With probability epsilon: random action (explore)
        - Otherwise: best known action (exploit)
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, done):
        """
        Q-Learning update rule:
        Q(s,a) = Q(s,a) + lr * (reward + gamma * max(Q(s')) - Q(s,a))
        """
        current_q = self.q_table[state, action]
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        
        # Update Q-value
        self.q_table[state, action] += self.lr * (target - current_q)

# Training loop
def train_q_learning(env, agent, episodes=1000):
    rewards_history = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        
        rewards_history.append(total_reward)
        
        if episode % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")
    
    return rewards_history

# ============================================
# Deep Q-Network (DQN)
# ============================================

class DQN(nn.Module):
    """
    Deep Q-Network: Use neural network to approximate Q-values
    
    For complex environments where Q-table is too large!
    """
    
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.001):
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = []  # Experience replay buffer
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.action_dim = action_dim
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:
            self.memory.pop(0)
    
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def replay(self):
        """Learn from random batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        states = torch.FloatTensor([b[0] for b in batch])
        actions = torch.LongTensor([b[1] for b in batch])
        rewards = torch.FloatTensor([b[2] for b in batch])
        next_states = torch.FloatTensor([b[3] for b in batch])
        dones = torch.FloatTensor([b[4] for b in batch])
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Update network
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        """Copy weights to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

# ============================================
# Policy Gradient (REINFORCE)
# ============================================

class PolicyNetwork(nn.Module):
    """
    Policy Gradient: Directly learn the policy
    
    Instead of learning Q-values, learn which actions to take!
    """
    
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)

class REINFORCEAgent:
    def __init__(self, state_dim, action_dim, lr=0.01):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = 0.99
        
        self.saved_log_probs = []
        self.rewards = []
    
    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state_tensor)
        
        # Sample action from probability distribution
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        
        self.saved_log_probs.append(dist.log_prob(action))
        return action.item()
    
    def learn(self):
        """Update policy after episode ends"""
        # Calculate discounted returns
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Policy gradient loss
        policy_loss = []
        for log_prob, G in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        
        # Clear episode data
        self.saved_log_probs = []
        self.rewards = []

# ============================================
# Using Stable Baselines3 (Easy RL)
# ============================================

from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env

# Create environment
env = make_vec_env("CartPole-v1", n_envs=4)

# Train with PPO (state-of-the-art algorithm)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Save and load
model.save("ppo_cartpole")
model = PPO.load("ppo_cartpole")

# Test the trained agent
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()

# ============================================
# Custom Environment
# ============================================

import gym
from gym import spaces

class SimpleGridWorld(gym.Env):
    """
    Custom environment: Agent navigates a grid to reach goal
    """
    
    def __init__(self, size=5):
        super().__init__()
        self.size = size
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Box(
            low=0, high=size-1, shape=(2,), dtype=np.float32
        )
        self.goal = np.array([size-1, size-1])
    
    def reset(self):
        self.agent_pos = np.array([0, 0])
        return self.agent_pos.astype(np.float32)
    
    def step(self, action):
        # Move agent
        if action == 0 and self.agent_pos[1] > 0:  # up
            self.agent_pos[1] -= 1
        elif action == 1 and self.agent_pos[1] < self.size-1:  # down
            self.agent_pos[1] += 1
        elif action == 2 and self.agent_pos[0] > 0:  # left
            self.agent_pos[0] -= 1
        elif action == 3 and self.agent_pos[0] < self.size-1:  # right
            self.agent_pos[0] += 1
        
        # Check if reached goal
        done = np.array_equal(self.agent_pos, self.goal)
        reward = 1.0 if done else -0.01
        
        return self.agent_pos.astype(np.float32), reward, done, {}
    
    def render(self):
        grid = np.zeros((self.size, self.size), dtype=str)
        grid[:] = '.'
        grid[self.goal[1], self.goal[0]] = 'G'
        grid[self.agent_pos[1], self.agent_pos[0]] = 'A'
        print('\\n'.join([' '.join(row) for row in grid]))
        print()

# Train on custom environment
env = SimpleGridWorld()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)`

  return (
    <div className="space-y-8">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-white mb-4">Reinforcement Learning</h1>
        <p className="text-xl text-purple-200">Teaching AI through trial, error, and rewards!</p>
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
                <Gamepad2 className="w-6 h-6 text-teal-400" />
                What is Reinforcement Learning?
              </CardTitle>
            </CardHeader>
            <CardContent className="text-slate-300 space-y-4">
              <p className="text-lg">
                <strong className="text-teal-400">Reinforcement Learning (RL)</strong> is how AI learns by 
                interacting with an environment and receiving rewards or penalties. It's like training a 
                pet - good behavior gets treats, bad behavior doesn't!
              </p>
              <div className="bg-slate-900/50 p-4 rounded-lg border border-teal-500/30">
                <h4 className="text-teal-400 font-semibold mb-2">Think of it like this:</h4>
                <p>Imagine learning to play a video game without any instructions. You try different 
                buttons, see what happens, and learn from the score. That's exactly how RL works - 
                the AI tries actions, gets feedback, and improves over time!</p>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">The RL Loop</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col items-center">
                <div className="flex items-center gap-4 mb-4">
                  <div className="bg-teal-500 text-white px-6 py-3 rounded-lg text-center">
                    <p className="font-bold">Agent</p>
                    <p className="text-xs">The learner</p>
                  </div>
                  <div className="flex flex-col items-center">
                    <span className="text-teal-400">Action →</span>
                    <span className="text-orange-400">← State, Reward</span>
                  </div>
                  <div className="bg-orange-500 text-white px-6 py-3 rounded-lg text-center">
                    <p className="font-bold">Environment</p>
                    <p className="text-xs">The world</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Key RL Concepts</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {[
                { term: 'State', desc: 'Current situation the agent is in', icon: '📍', color: 'blue' },
                { term: 'Action', desc: 'What the agent can do', icon: '🎮', color: 'green' },
                { term: 'Reward', desc: 'Feedback signal (+/-)', icon: '🏆', color: 'yellow' },
                { term: 'Policy', desc: 'Strategy for choosing actions', icon: '🧠', color: 'purple' },
                { term: 'Value', desc: 'Expected future rewards', icon: '💎', color: 'pink' },
              ].map((item, i) => (
                <div key={i} className={`flex items-center gap-3 p-3 bg-${item.color}-500/10 rounded-lg border border-${item.color}-500/30`}>
                  <span className="text-2xl">{item.icon}</span>
                  <div>
                    <span className="text-white font-medium">{item.term}: </span>
                    <span className="text-slate-400">{item.desc}</span>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Popular RL Algorithms</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-4">
                {[
                  { name: 'Q-Learning', type: 'Value-based', desc: 'Learn value of state-action pairs', color: 'bg-blue-500' },
                  { name: 'DQN', type: 'Deep RL', desc: 'Q-Learning with neural networks', color: 'bg-purple-500' },
                  { name: 'Policy Gradient', type: 'Policy-based', desc: 'Directly learn the policy', color: 'bg-green-500' },
                  { name: 'PPO', type: 'Actor-Critic', desc: 'State-of-the-art, stable training', color: 'bg-orange-500' },
                ].map((algo, i) => (
                  <div key={i} className="bg-slate-900 p-4 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <div className={`w-8 h-8 ${algo.color} rounded flex items-center justify-center`}>
                        <Brain className="w-4 h-4 text-white" />
                      </div>
                      <div>
                        <p className="text-white font-semibold">{algo.name}</p>
                        <Badge variant="outline" className="text-slate-400 text-xs">{algo.type}</Badge>
                      </div>
                    </div>
                    <p className="text-slate-400 text-sm">{algo.desc}</p>
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
                <Trophy className="w-5 h-5 text-yellow-400" />
                Grid World Game
              </CardTitle>
              <CardDescription className="text-slate-400">
                Navigate the agent (🤖) to the goal (🎯)!
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-center">
                <div className="bg-slate-900 p-4 rounded-lg">
                  <div className="grid grid-cols-4 gap-1">
                    {Array(gridSize * gridSize).fill(0).map((_, i) => {
                      const x = i % gridSize
                      const y = Math.floor(i / gridSize)
                      const isAgent = agentPos.x === x && agentPos.y === y
                      const isGoal = goalPos.x === x && goalPos.y === y
                      
                      return (
                        <div 
                          key={i}
                          className={`w-12 h-12 rounded flex items-center justify-center text-2xl ${
                            isAgent ? 'bg-teal-500' :
                            isGoal ? 'bg-yellow-500' :
                            'bg-slate-700'
                          }`}
                        >
                          {isAgent && '🤖'}
                          {isGoal && !isAgent && '🎯'}
                        </div>
                      )
                    })}
                  </div>
                </div>
              </div>

              <div className="flex justify-center gap-2">
                <div className="grid grid-cols-3 gap-1">
                  <div />
                  <Button size="sm" onClick={() => moveAgent('up')} className="bg-teal-600">
                    <ArrowUp className="w-4 h-4" />
                  </Button>
                  <div />
                  <Button size="sm" onClick={() => moveAgent('left')} className="bg-teal-600">
                    <ArrowLeft className="w-4 h-4" />
                  </Button>
                  <Button size="sm" onClick={resetGame} variant="outline">
                    Reset
                  </Button>
                  <Button size="sm" onClick={() => moveAgent('right')} className="bg-teal-600">
                    <ArrowRight className="w-4 h-4" />
                  </Button>
                  <div />
                  <Button size="sm" onClick={() => moveAgent('down')} className="bg-teal-600">
                    <ArrowDown className="w-4 h-4" />
                  </Button>
                  <div />
                </div>
              </div>

              <div className="flex justify-center gap-8">
                <div className="text-center">
                  <p className="text-slate-400 text-sm">Moves</p>
                  <p className="text-2xl font-bold text-white">{moves}</p>
                </div>
                <div className="text-center">
                  <p className="text-slate-400 text-sm">Score</p>
                  <p className="text-2xl font-bold text-yellow-400">{score}</p>
                </div>
              </div>

              <p className="text-slate-400 text-sm text-center">
                In RL, the agent learns to find the shortest path to maximize rewards!
              </p>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Exploration vs Exploitation</CardTitle>
              <CardDescription className="text-slate-400">
                The fundamental trade-off in RL
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-blue-500/10 p-4 rounded-lg border border-blue-500/30">
                  <h4 className="text-blue-400 font-semibold mb-2">🔍 Exploration</h4>
                  <p className="text-slate-300 text-sm">Try new actions to discover better strategies</p>
                  <p className="text-slate-500 text-xs mt-2">"What if I try something different?"</p>
                </div>
                <div className="bg-green-500/10 p-4 rounded-lg border border-green-500/30">
                  <h4 className="text-green-400 font-semibold mb-2">💰 Exploitation</h4>
                  <p className="text-slate-300 text-sm">Use known good actions to maximize reward</p>
                  <p className="text-slate-500 text-xs mt-2">"I know this works, let's do it!"</p>
                </div>
              </div>
              <p className="text-slate-400 text-sm mt-4 text-center">
                ε-greedy: With probability ε explore randomly, otherwise exploit best known action
              </p>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">RL Applications</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-3 gap-3">
                {[
                  { app: 'Game Playing', example: 'AlphaGo, Atari', icon: '🎮' },
                  { app: 'Robotics', example: 'Walking, grasping', icon: '🤖' },
                  { app: 'Autonomous Driving', example: 'Navigation, control', icon: '🚗' },
                  { app: 'Trading', example: 'Stock trading bots', icon: '📈' },
                  { app: 'Recommendations', example: 'Content suggestions', icon: '🎯' },
                  { app: 'Resource Management', example: 'Data centers, energy', icon: '⚡' },
                ].map((item, i) => (
                  <div key={i} className="bg-slate-900 p-3 rounded-lg text-center">
                    <span className="text-3xl">{item.icon}</span>
                    <p className="text-white font-medium mt-2">{item.app}</p>
                    <p className="text-slate-500 text-xs">{item.example}</p>
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
                <Code className="w-5 h-5 text-teal-400" />
                Python Code: Reinforcement Learning
              </CardTitle>
              <CardDescription className="text-slate-400">
                From Q-Learning to Deep RL with Stable Baselines!
              </CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="bg-slate-900 p-4 rounded-lg overflow-x-auto text-sm max-h-96 overflow-y-auto">
                <code className="text-green-400">{pythonCode}</code>
              </pre>
              <div className="mt-4 p-4 bg-teal-500/10 rounded-lg border border-teal-500/30">
                <h4 className="text-teal-400 font-semibold mb-2">Required Libraries:</h4>
                <code className="text-slate-300 text-sm bg-slate-800 px-2 py-1 rounded">
                  pip install gym stable-baselines3 torch numpy
                </code>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default Module14ReinforcementLearning
