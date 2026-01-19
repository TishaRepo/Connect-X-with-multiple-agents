# Connect-X with Multiple RL Agents

> **Kaggle Competition Project**: Build a Reinforcement Learning agent to compete against other participants in the [Kaggle Connect X](https://www.kaggle.com/c/connectx) simulation competition.

## 🎯 Project Objective

This project implements multiple RL (Reinforcement Learning) agents to compete in the Kaggle Connect X competition. The approach follows a structured learning path:

1. **Understand the game mechanics** and environment
2. **Create baseline agents** (Random and Rule-Based) for comparison
3. **Experiment with RL algorithms**:
   - Deep Q-Network (DQN)
   - Proximal Policy Optimization (PPO)
4. **Train and evaluate** agents against various opponents
5. **Compare performance** to select the best submission

## 🎮 About Connect X

Connect X is a classic "connect N in a row" game similar to Connect Four:

| Property | Value |
|----------|-------|
| **Board Size** | 7 columns × 6 rows |
| **Objective** | Connect 4 pieces in a row |
| **Win Conditions** | Horizontal, Vertical, or Diagonal |
| **Players** | 2 players (alternating turns) |
| **Actions** | Drop a piece in columns 0-6 |
| **Rewards** | +1 (win), -1 (loss), 0 (draw) |

## 🤖 Implemented Agents

### Baseline Agents

#### 1. Random Agent
A simple baseline that randomly selects a valid column. Used to establish minimum performance expectations.

#### 2. Rule-Based Agent
A heuristic-based agent implementing basic game strategy:
1. ✅ Play winning move if available
2. 🛡️ Block opponent's winning move
3. 📍 Prefer center column (strategic advantage)
4. 🎲 Random fallback for other situations

**Performance**: ~92% win rate against Random Agent

### RL Agents

#### 3. DQN (Deep Q-Network) Agent
A value-based deep reinforcement learning agent:

| Component | Implementation |
|-----------|----------------|
| **Network** | 4-layer fully connected (input → 128 → 128 → 128 → 7) |
| **Experience Replay** | Buffer size: 10,000 |
| **Target Network** | Soft update every 100 steps |
| **Exploration** | ε-greedy (1.0 → 0.01, decay: 0.995) |
| **Action Masking** | Invalid moves masked with -∞ |
| **Gradient Clipping** | Max norm: 1.0 |

#### 4. PPO (Proximal Policy Optimization) Agent
A policy gradient agent with actor-critic architecture:

| Component | Implementation |
|-----------|----------------|
| **Architecture** | Shared backbone + Actor/Critic heads |
| **Hidden Size** | 256 (shared), 128 (heads) |
| **Clipping** | ε = 0.2 |
| **Entropy Bonus** | 0.01 (encourages exploration) |
| **K-Epochs** | 4 passes per update |
| **Update Frequency** | Every 10 episodes |

## 📁 Project Structure

```
Connect-X-with-multiple-agents/
├── ConnectX_RL_Agent.ipynb   # Main notebook with all implementations
├── README.md                  # Project documentation
└── LICENSE                    # MIT License
```

## 🛠️ Requirements

- Python 3.10+
- PyTorch (for neural networks)
- NumPy
- Matplotlib & Seaborn (visualization)
- kaggle-environments (competition environment)

```bash
pip install numpy torch matplotlib seaborn kaggle-environments
```

## 🚀 Getting Started

### 1. Clone and Setup
```bash
git clone https://github.com/yourusername/Connect-X-with-multiple-agents.git
cd Connect-X-with-multiple-agents
pip install -r requirements.txt  # or install dependencies manually
```

### 2. Run the Notebook
Open `ConnectX_RL_Agent.ipynb` in Jupyter and execute cells sequentially:
- Cells 1-5: Environment setup and baseline agents
- Cells 6-9: DQN implementation and training
- Cells 10-13: PPO implementation and training
- Cells 14+: Evaluation and visualization

### 3. Training Examples

**DQN Agent (Two-Phase Training)**:
```python
# Phase 1: Learn basics against random opponent
dqn_agent = train_dqn_agent(episodes=500, opponent='random')

# Phase 2: Refine against stronger opponent
dqn_agent_final = train_dqn_agent(episodes=500, opponent='rule_based', epsilon_start=0.3)
```

**PPO Agent**:
```python
# Phase 1: Train against random
ppo_agent = train_ppo_agent(episodes=300, opponent='random')

# Phase 2: Train against rule-based
ppo_agent_final = train_ppo_agent(episodes=300, opponent='rule_based')
```

### 4. Evaluate Performance
```python
# Compare agents head-to-head
compare_agents(dqn_agent_wrapper, ppo_agent_wrapper, "DQN", "PPO", n_rounds=100)

# Visualize a game
visualize_game(dqn_agent_wrapper, rule_based_agent, "DQN", "Rule-based")
```

## 📊 Expected Performance

| Agent | vs Random | vs Rule-Based |
|-------|:---------:|:-------------:|
| **Rule-Based** | ~92% | — |
| **DQN** | 60-80% | 30-50% |
| **PPO** | 70-90% | 40-60% |

*Win rates after standard training. Results may vary based on training duration and hyperparameters.*

## � Technical Implementation

### State Encoding
The board is encoded as a 3-channel binary representation (126 features total):
```
Channel 1: Agent's pieces     (6×7 = 42 values)
Channel 2: Opponent's pieces  (6×7 = 42 values)
Channel 3: Empty spaces       (6×7 = 42 values)
```

### Reward Shaping
| Outcome | Reward |
|---------|--------|
| Win | +1.0 |
| Loss | -1.0 |
| Draw | +0.5 |
| Ongoing | 0.0 |

### Training Hyperparameters

**DQN**:
- Learning Rate: 0.001
- Batch Size: 64
- Discount (γ): 0.99
- Epsilon Decay: 0.995
- Replay Buffer: 10,000

**PPO**:
- Learning Rate: 0.0003
- Discount (γ): 0.99
- Clip Range: 0.2
- K-Epochs: 4
- Gradient Clip: 0.5

## 🏆 Kaggle Submission

To submit your trained agent to the Kaggle competition:

1. Export your trained agent as a Python function
2. Create a `submission.py` file with your agent
3. Submit via Kaggle Notebooks or API

```python
# Example submission wrapper
def my_agent(obs, config):
    state = encode_board(obs, config)
    valid_actions = get_valid_actions(obs, config)
    action = trained_agent.act(state, valid_actions, eps=0.0)
    return int(action)
```

## 🔑 Key Improvements in This Implementation

- ✅ **Proper action masking** - Only valid moves are considered
- ✅ **Correct target network updates** - Stable Q-learning
- ✅ **Balanced reward shaping** - Clear win/loss signals
- ✅ **Fixed tensor dimensions** - PPO critic outputs 1D tensors
- ✅ **Gradient clipping** - Prevents training instabilities
- ✅ **Curriculum learning** - Progressive difficulty increase

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Kaggle Connect X Competition](https://www.kaggle.com/c/connectx)
- [kaggle-environments](https://github.com/Kaggle/kaggle-environments) library
- PyTorch team for the deep learning framework