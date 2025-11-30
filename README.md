# Multi-Snake DQN (Shared Policy)

This is a **Python + PyTorch** implementation of a multi-snake (1 vs 1) reinforcement learning game using a **shared DQN (Deep Q-Network)**.

Two snakes share the same policy network and compete to eat food, learning to avoid collisions while maximizing their scores.

---

## Features

- Two snakes play simultaneously (**shared policy network**)  
- Uses **DQN with Target Network and Replay Buffer** for training  
- Supports **checkpoint saving and loading**, allowing training to resume  
- Optional **rendering for visualization** or faster headless training  
- Reward shaping: rewards not only for eating food but also for approaching it  

---

## Requirements

- Python 3.9+  
- PyTorch 1.13+  
- Pygame  
- Numpy  

Install dependencies with:

```bash
pip install torch torchvision torchaudio pygame numpy
