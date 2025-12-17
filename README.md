# Multi-Agent Snake DQN: Hunter vs. Gatherer

This project implements a Multi-Agent Reinforcement Learning environment using **Pygame** and **PyTorch (Deep Q-Learning)**. It features an **Asymmetric Adversarial** gameplay mechanism where two agents compete with distinct objectives and reward functions.

## 🎮 Game Mechanics

The game operates on a fixed-duration timer (default: 5 minutes). Agents respawn immediately upon death while retaining their scores until the time runs out.

### Roles & Objectives
1.  **Snake 1 (Gatherer / Blue)**
    * **Goal**: Consume as much food (Red) as possible.
    * **Behavior**: Grows in length upon eating food.
    * **Perception**: Focused on locating food coordinates relative to itself.

2.  **Snake 2 (Hunter / Green)**
    * **Goal**: Intercept, block, or kill Snake 1.
    * **Behavior**:
        * **Blocking**: Successfully forcing S1 to crash into S2's body results in a score increase and growth for S2.
        * **Head-on Collision**: If heads collide, S2 wins (hunter takes down prey).
        * **Food**: Eating food removes it from the map but **does not increase S2's length**. This prevents the hunter from becoming too long and clumsy, allowing it to focus on combat agility.
    * **Perception**: Focused on locating Snake 1's head coordinates.

## 🧠 RL Architecture

### State Space (15 Inputs)
Each agent perceives the environment through 15 boolean inputs relative to its own perspective:
1.  **Danger Perception (3)**: Obstacles (wall or body) straight ahead, left, or right.
2.  **Current Direction (4)**: Moving Left, Right, Up, or Down (One-hot encoding).
3.  **Target Direction (4)**:
    * For S1: Relative direction to Food.
    * For S2: Relative direction to S1's Head.
4.  **Opponent Direction (4)** (Currently used by S2): The moving direction of the opponent.

### Reward Function

| Event | Snake 1 (Gatherer) | Snake 2 (Hunter) |
| :--- | :--- | :--- |
| **S1 Eats Food** | **+100** | **-100** (Enemy scored) |
| **S1 Dies (Wall/Self)** | **-50** | N/A |
| **S2 Dies (Wall/Self)** | N/A | **-50** |
| **S2 Blocks S1** | **-100** (Blocked) | **+200** (Successful Block) |
| **Head-on Collision** | **-100** (Killed) | **+100** (Successful Kill) |
| **Distance Shaping** | (Optional) Small + | (Optional) Small + |

### Model Structure
* **Algorithm**: Deep Q-Network (DQN) with Experience Replay & Target Network.
* **Network**: 3-Layer MLP (Linear -> ReLU -> Linear -> ReLU -> Linear).
* **Optimizer**: Adam (Learning Rate = 1e-3).
* **Loss Function**: MSE Loss.

## 🛠️ Installation & Usage

### 1. Requirements
Ensure you have Python 3.8+ and the following libraries installed:
```bash
pip install pygame torch numpy
