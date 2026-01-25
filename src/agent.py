from stable_baselines3 import PPO
from .env import SupplyChainEnv

def train_agent():
    env = SupplyChainEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("../models/supply_chain_ppo")
    return model