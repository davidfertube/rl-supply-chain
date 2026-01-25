import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SupplyChainEnv(gym.Env):
    def __init__(self):
        super(SupplyChainEnv, self).__init__()
        
        # Action: Order Quantity (0 to 100)
        self.action_space = spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)
        
        # Observation: [Current Stock, Forecast Demand, Incoming Orders]
        self.observation_space = spaces.Box(low=0, high=10000, shape=(3,), dtype=np.float32)
        
        self.state = 100 # Initial stock
        
    def step(self, action):
        order_qty = action[0]
        demand = np.random.poisson(20) # Stochastic demand
        
        # State transition
        self.state = max(0, self.state + order_qty - demand)
        
        # Reward function
        holding_cost = self.state * 0.1
        stockout_penalty = 10.0 if self.state == 0 else 0
        reward = -(holding_cost + stockout_penalty)
        
        done = False
        info = {"demand": demand}
        
        return np.array([self.state, demand, order_qty], dtype=np.float32), reward, done, False, info

    def reset(self, seed=None):
        self.state = 100
        return np.array([self.state, 0, 0], dtype=np.float32), {}