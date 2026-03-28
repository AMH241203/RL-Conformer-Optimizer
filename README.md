# 🧬 Molecular Conformer Optimization via Reinforcement Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![RDKit](https://img.shields.io/badge/RDKit-Chemistry-green)
![StableBaselines3](https://img.shields.io/badge/Stable_Baselines3-RL-orange)

A deep reinforcement learning system that teaches itself the laws of molecular physics — no hardcoded heuristics, no lookup tables. Just a PPO agent, a force field, and trial and error until it finds the lowest-energy 3D conformation of an organic molecule.

---

## 🎥 The Optimization Engine in Action

<!-- `conformer_optimization_smooth.mp4` -->

What you're watching above is zero-shot physical optimization. The agent starts from a high-energy, sterically hindered configuration and — through pure gradient descent on a reward signal from a quantum chemistry force field — works its way to the global energy minimum. In the case of ethane, that's the anti-staggered conformation: the arrangement every physical chemistry textbook tells you is most stable, rediscovered here entirely from scratch.

---

## 🧠 How It Works

This project reframes molecular conformation as a continuous control problem. The molecule is an environment; the agent's job is to find the configuration that makes a physics simulator happiest.

### The Environment (Custom Gymnasium)

Three design decisions define this environment:

- **State:** A flattened 1D tensor of every atom's $(X, Y, Z)$ coordinates. The agent sees the molecule as raw geometry, nothing more.
- **Actions:** A continuous vector in $[-1.0,\ 1.0]$, scaled to $[-\pi,\ \pi]$ radians. Each dimension corresponds to a rotatable bond — the agent decides how much to twist each dihedral angle per step.
- **Reward:** RDKit's MMFF94 force field evaluates the molecule after every action and returns its potential energy. The agent's reward is the *negative* of that energy, so the only way to score higher is to make the molecule physically happier.

No manual features. No domain-specific loss terms. The entire signal the agent ever receives is: *"this conformation costs X kcal/mol."*

### The Agent (PPO + MLP Actor-Critic)

The agent is a standard Proximal Policy Optimization setup from Stable Baselines3, using a Multi-Layer Perceptron actor-critic. There's nothing exotic here by design — the interesting part is what the agent figures out on its own. Given enough environment interactions, it infers the geometry of steric clashes and torsional strain from scratch, learning rules that took decades of experimental chemistry to formalize.

---

## 🚀 Running It Locally

RDKit carries C++ dependencies that are easiest to manage through Conda. Using pip alone tends to cause headaches.

```bash
# Clone the repo
git clone https://github.com/AMH241203/RL-Conformer-Optimizer.git
cd RL-Conformer-Optimizer

# Create and activate a Conda environment
conda create -n conformer_rl python=3.10 -y
conda activate conformer_rl

# Install RDKit via conda-forge (handles the C++ bindings)
conda install -c conda-forge rdkit -y

# Install the rest
pip install torch numpy gymnasium stable-baselines3
```

> **Why Conda?** RDKit's Python bindings wrap a C++ core. Conda-forge pre-builds these against a consistent ABI, which saves you from a surprisingly unpleasant compilation experience.

---

## 📁 Project Structure

```
RL-Conformer-Optimizer/
│
├── .gitignore                 
├── README.md                         
│
├── src/                       
│   └── env.py
│
├── scripts/                   
│   ├── train.py                
│   ├── evaluate.py          
│   └── visualize.py          
│
├── models/                     
│   └── ppo_conformer_model.zip
│
├── visualizations/           
│   ├── start.sdf               
│   ├── optimized.sdf           
│   └── conformer_optimization_smooth.mp4
│
└── conformer_tensorboard/
    └── PPO_1/
        └── events.out.tfevents...
```

---

## 🔬 Why This Is Interesting

Conformer search is a real problem in drug discovery. The 3D shape of a molecule determines how it docks to a protein, and there can be millions of possible conformations for a flexible ligand. Classical methods (grid search, MMFF minimization, Monte Carlo sampling) are well-studied but fundamentally heuristic. Training an RL agent on a force field is a small step toward learning geometry-aware policies that might generalize across molecular scaffolds — something rule-based solvers can't do.

This project is a proof of concept on a simple molecule, but the environment and training loop are designed to scale.

---
