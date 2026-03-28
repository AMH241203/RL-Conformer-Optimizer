from src.env import ConformerEnv

def run_random_baseline():
    print("Initializing environment...")
    env = ConformerEnv(smiles="CCCC") # Butane has 1 rotatable bond
    
    obs, info = env.reset()
    print(f"Initial Observation Shape: {obs.shape}")
    
    epochs = 10
    print("\nRunning random actions...")
    for step in range(epochs):
        # Sample a valid random action from our defined Box space
        action = env.action_space.sample() 
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step + 1} | Action: {action} | Energy: {info.get('energy', 'N/A'):.2f} | Reward: {reward:.2f}")

if __name__ == "__main__":
    run_random_baseline()