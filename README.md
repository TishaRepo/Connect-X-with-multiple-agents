# Connect-X with Multiple RL Agents

> Building Reinforcement Learning agents for the [Kaggle Connect X](https://www.kaggle.com/c/connectx) competition.

## What's This Project About?

This project tackles the Kaggle Connect X challenge—a Connect Four-style game where you build AI agents to compete. I implemented and compared multiple approaches:

- **Baseline agents**: Random and Rule-Based (for benchmarking)
- **Deep Q-Network (DQN)**: Value-based RL
- **Proximal Policy Optimization (PPO)**: Policy-based RL

The goal was to understand both algorithms, make them work correctly for this game, and see which one performs better.

## The Game

Connect X is played on a 7×6 board. Players take turns dropping pieces, trying to connect 4 in a row (horizontally, vertically, or diagonally). Simple rules, but surprisingly deep strategy.

## Quick Start

```bash
# Install dependencies
pip install numpy torch matplotlib seaborn kaggle-environments

# Open the notebook
jupyter notebook ConnectX_RL_Agent.ipynb
```

Run the cells in order—the notebook handles everything from training to evaluation.

## Results

| Agent | vs Random | vs Rule-Based |
|-------|:---------:|:-------------:|
| Rule-Based | ~92% | — |
| DQN | 60-80% | 30-50% |
| PPO | 70-90% | 40-60% |

## How the Algorithms Work

### DQN (Deep Q-Network)

DQN learns a Q-function that estimates "how good is taking action A in state S?"

```
State → Neural Network → Q-values for each column → Pick highest (valid) one
```

**Key ideas:**
- **Experience Replay**: Store past moves, sample randomly to break correlations
- **Target Network**: Separate network for stable training targets
- **ε-Greedy**: Start random (explore), gradually become greedy (exploit)

### PPO (Proximal Policy Optimization)

PPO directly learns a policy (probability distribution over actions) using actor-critic architecture.

```
State → Shared Layers → Actor (action probs) + Critic (state value)
```

**Key ideas:**
- **Actor**: Decides what to do
- **Critic**: Evaluates how good the situation is
- **Clipped Updates**: Prevents the policy from changing too drastically

## My Implementation Approach

Here's what makes this implementation work well:

### 1. Smart State Encoding

Instead of feeding raw board values, I encode it as 3 binary channels:
```python
my_pieces    = where(board == my_mark)      # Where I am
opp_pieces   = where(board == opp_mark)     # Where opponent is  
empty_spaces = where(board == 0)            # What's available
```
This way, the agent always sees the game from its own perspective—works for both Player 1 and Player 2.

### 2. Action Masking

You can't drop a piece in a full column. I mask invalid actions:
- **DQN**: Set Q-value to `-∞` for full columns
- **PPO**: Zero out probabilities, then renormalize

No wasted learning on impossible moves.

### 3. Reward Shaping

```python
Win:  +1.0
Loss: -1.0
Draw: +0.5  # Better than losing!
```

I intentionally avoided step penalties—they can make the agent rush into bad decisions.

### 4. Curriculum Learning

Training in two phases works better:
1. **vs Random**: Learn the basics (how to win)
2. **vs Rule-Based**: Learn defense (how to block)

Starting against a strong opponent just leads to constant losses and slow learning.

### 5. Bug Fixes

A few things that tutorials often get wrong:
- Target network should update every N steps, not every step
- PPO critic output needs `.squeeze(-1)` to match reward dimensions
- Gradient clipping prevents training explosions

## Project Structure

```
├── ConnectX_RL_Agent.ipynb   # Everything is here
├── README.md
└── LICENSE
```

## Kaggle Submission

To submit your trained agent:

```python
def my_agent(obs, config):
    state = encode_board(obs, config)
    valid_actions = get_valid_actions(obs, config)
    action = trained_agent.act(state, valid_actions, eps=0.0)
    return int(action)
```

## What I Learned

- Action masking is crucial for games with invalid moves
- Curriculum learning really helps—don't throw the agent into the deep end
- DQN is simpler to implement; PPO is more stable once you get it right
- Small bugs (like tensor dimensions) can completely break training

## License

MIT License - see [LICENSE](LICENSE) file.

---

Built for the Kaggle Connect X competition. Feel free to use, modify, and improve!