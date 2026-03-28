import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from src.env import ConformerEnv

def evaluate_agent():
    env = ConformerEnv(smiles="CCCC")
    
    # Load the trained PyTorch weights
    model = PPO.load("ppo_conformer_model")
    
    obs, info = env.reset()
    print(f"Initial Energy: {info.get('energy'):.2f}")
    
    # Let the AI take 10 steps to find the lowest energy conformer
    for step in range(10):
        # The model predicts the best action based on the observation
        # deterministic=True means it takes the absolute best action, no random exploration
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step + 1} | Action Taken: {action} | New Energy: {info.get('energy'):.2f}")

if __name__ == "__main__":
    evaluate_agent()
    