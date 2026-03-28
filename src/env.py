import gymnasium as gym
from gymnasium import spaces
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

class ConformerEnv(gym.Env):
    def __init__(self, smiles="CCCC"): 
        super(ConformerEnv, self).__init__()
        
        self.smiles = smiles
        self.mol = Chem.AddHs(Chem.MolFromSmiles(self.smiles))
        AllChem.EmbedMolecule(self.mol, randomSeed=42)
        self.conf = self.mol.GetConformer()
        
        rot_smarts = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
        rotatable_bond_matches = self.mol.GetSubstructMatches(rot_smarts)
        
        self.dihedrals = []
        for match in rotatable_bond_matches:
            j, k = match[0], match[1] 
            atoms = self._get_dihedral_atoms(j, k)
            if atoms:
                self.dihedrals.append(atoms)
                
        self.num_bonds = len(self.dihedrals)
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_bonds,), dtype=np.float32
        )
        
        num_atoms = self.mol.GetNumAtoms()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_atoms * 3,), dtype=np.float32
        )
        
        # NEW: Episode limits
        self.max_steps = 50 
        self.current_step = 0

    def _get_dihedral_atoms(self, j, k):
        i = -1
        for neighbor in self.mol.GetAtomWithIdx(j).GetNeighbors():
            idx = neighbor.GetIdx()
            if idx != k:
                i = idx
                break
                
        l = -1
        for neighbor in self.mol.GetAtomWithIdx(k).GetNeighbors():
            idx = neighbor.GetIdx()
            if idx != j:
                l = idx
                break
                
        if i == -1 or l == -1:
            return None
        return (i, j, k, l)

    def _get_obs(self):
        return self.conf.GetPositions().flatten().astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0 # Reset the step counter
        
        AllChem.EmbedMolecule(self.mol, randomSeed=seed if seed else 42)
        self.conf = self.mol.GetConformer()
        
        # BUG FIX: Calculate initial energy to pass back in info dict
        obs = self._get_obs()
        ff_props = AllChem.MMFFGetMoleculeProperties(self.mol)
        ff = AllChem.MMFFGetMoleculeForceField(self.mol, ff_props)
        initial_energy = ff.CalcEnergy() if ff is not None else 0.0
        
        return obs, {"energy": initial_energy}

    def step(self, action):
        self.current_step += 1 
        
        for idx, (i, j, k, l) in enumerate(self.dihedrals):
            angle_rad = float(action[idx]) * np.pi 
            rdMolTransforms.SetDihedralRad(self.conf, i, j, k, l, angle_rad)
            
        obs = self._get_obs()
        
        ff_props = AllChem.MMFFGetMoleculeProperties(self.mol)
        ff = AllChem.MMFFGetMoleculeForceField(self.mol, ff_props)
        
        if ff is None:
            return obs, -100.0, True, False, {"error": "Invalid force field"}
            
        energy = ff.CalcEnergy()
        reward = -energy 
        
        terminated = False 
        # NEW: End the episode if we hit the max steps
        truncated = self.current_step >= self.max_steps 
        
        return obs, reward, terminated, truncated, {"energy": energy}