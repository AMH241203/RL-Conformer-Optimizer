import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.env import ConformerEnv
from stable_baselines3 import PPO # type: ignore
from rdkit import Chem


def generate_morph_files():
    os.makedirs("visualizations", exist_ok=True)
    
    env = ConformerEnv(smiles="CCCC")
    model = PPO.load("models/ppo_conformer_model")    
    obs, info = env.reset()
    
    # 1. Save the initial twisted state
    writer_start = Chem.SDWriter("visualizations/start.sdf")
    env.mol.SetProp("_Name", f"Start_Energy_{info.get('energy'):.2f}")
    writer_start.write(env.mol)
    writer_start.close()
    
    # 2. Let the AI take its 1 perfect step
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 3. Save the completely solved state
    writer_end = Chem.SDWriter("visualizations/optimized.sdf")
    env.mol.SetProp("_Name", f"Optimized_Energy_{info.get('energy'):.2f}")
    writer_end.write(env.mol)
    writer_end.close()
    
    print("Success!")

if __name__ == "__main__":
    generate_morph_files()