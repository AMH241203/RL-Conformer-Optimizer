import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.env import ConformerEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

def train_agent():
    print("Initializing Conformer Environment...")
    # Using Butane for the initial test (1 rotatable bond)
    env = ConformerEnv(smiles="CCCC") 
    
    # 1. Sanity Check
    # This SB3 utility checks your env to ensure it strictly follows the 
    # Gymnasium API. If your state or action spaces are mismatched, it fails here.
    check_env(env, warn=True)
    print("Environment passed Gymnasium API check!")

    # 2. Initialize the PPO Agent
    # MlpPolicy tells it to build standard PyTorch feed-forward networks
    # verbose=1 prints training metrics to the console
    print("Building PyTorch PPO Agent...")
    model = PPO(
        policy="MlpPolicy", 
        env=env, 
        learning_rate=0.0003, # Standard starting LR for PPO
        verbose=1, 
        tensorboard_log="./conformer_tensorboard/" # For Phase 3 evaluation
    )

    # 3. The Training Loop
    # The agent will take 10,000 actions in the environment to learn
    # You will eventually increase this to 100k or 1M for complex molecules
    timesteps = 100000
    print(f"Starting training for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps)

    # 4. Save the Weights
    model_path = "ppo_conformer_model"
    model.save(model_path)
    print(f"Training complete! PyTorch model saved to {model_path}.zip")

if __name__ == "__main__":
    train_agent()