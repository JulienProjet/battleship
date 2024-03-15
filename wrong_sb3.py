from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, MultiDiscrete, Tuple, Dict
"""
import gym
from gym import Env
from gym.spaces import Discrete, Box, MultiDiscrete
"""
import numpy as np
import random
import numpy as np
import warnings

from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks


#warnings.filterwarnings("ignore")

class BattleShipEnv(gym.Env):
    def __init__(self, size=10, ships=[5,4,3,3,2]):
        super(BattleShipEnv, self).__init__()

        # Define action and observation space
        #self.action_space = Discrete(size*size)
        self.action_space = MultiDiscrete([size,size])
        #self.observation_space = Box(low=0, high=3, shape=(3, size, size), dtype=int)
        self.observation_space = Dict({
            'grid': Box(low=0, high=1, shape=(3, size, size), dtype=int),  # Grid observation
            'ships_remaining': Box(low=0, high=max(ships), shape=(len(ships),), dtype=int)  # Ships remaining
        })
        # Initialize state
        self.state = np.zeros((4, size, size), dtype=int)
        #self.observation = np.zeros((3, size, size), dtype=int) 
        self.size = size
        self.ships = ships
        self.nb_steps = 0
        self.nb_invalid_move = 0
        self.total_invalid_moves = 0
        
        #self.ships_remaining = np.zeros((size,), dtype=int)
        self.ships_remaining = np.array(ships)
        
    def action_masks(self):
        # Generate action mask
        action_mask = np.ones(self.action_space.nvec, dtype=bool)

        # Iterate over board to mask invalid actions
        for row in range(self.size):
            for col in range(self.size):
                if self.state[0, row, col] != 0:
                    action_mask[row, col] = False

        return action_mask
    
    def step(self, action):
        self.nb_steps += 1
        reward = -0.1

        #row = action // self.size
        #col = action % self.size
        row,col = action
        
        if self.state[0, row, col] == 0:
            self.state[0, row, col] = 2  # Miss
            self.state[2, row, col] = 1  # Add this line
            reward -= 1
            
        elif self.state[0, row, col] == 1:
            self.state[0, row, col] = 3  # Hit
            self.state[1, row, col] = 1  # Add this line
            reward += 10
            
            ship_size = self.get_ship_size(row, col)
            if ship_size in self.ships and not (self.state[0] == ship_size).any():
                self.ships_remaining[self.ships.index(ship_size)] = 0
                
        else:
            self.nb_invalid_move +=1
            self.state[3, row, col] = 1
            #self.total_invalid_moves += 1
            reward = -50  # Invalid move
            #done = False
            #info = {'total_invalid_moves': self.total_invalid_moves}
            #return self.state, reward, done, False, info

        done = not (self.state[0] == 1).any()  # Game is over if no ships are left
        if done :
            reward += 15
            print("\nGame over, number of steps:", self.nb_steps)
            print("Number of invalid moves:", self.nb_invalid_move)
        
        self.observation = {'grid': np.stack([self.state[1], self.state[2], self.state[3]]), 'ships_remaining': self.ships_remaining}
        
        info = {'total_invalid_moves': self.nb_invalid_move}
        return self.observation, reward, done, False, info

    def reset(self,seed=None):
        self.state = np.zeros((4, self.size, self.size), dtype=int)
        self.observation = {'grid': np.zeros((3, self.size, self.size), dtype=int), 'ships_remaining': np.array(self.ships)}
        self.nb_steps = 0
        self.nb_invalid_move = 0
        for ship in self.ships:
            self.place_ship(ship)
        info = {}
        return self.observation, info

    def render(self, mode='human'):
        # Render the environment to the screen
        #print(self.state)
        for row in self.state:
            print(' '.join(['~' if cell == 0 else 'S' if cell == 1 else 'X' if cell == 3 else 'O' for cell in row]))
        print()



    def place_ship(self, ship_size):
        # Place a ship of given size on the board
        while True:
            direction = np.random.choice(["vertical", "horizontal"])
            if direction == "horizontal":
                row = np.random.randint(self.size)
                col = np.random.randint(self.size - ship_size + 1)
                if self.state[0, row, col:col+ship_size].sum() == 0:  # Access the first layer of self.state
                    self.state[0, row, col:col+ship_size] = 1
                    break
            else:  # direction == "vertical"
                row = np.random.randint(self.size - ship_size + 1)
                col = np.random.randint(self.size)
                if self.state[0, row:row+ship_size, col].sum() == 0:  # Access the first layer of self.state
                    self.state[0, row:row+ship_size, col] = 1
                    break
                
    def get_ship_size(self, row, col):
        # Return the size of the ship at the given position
        ship_size = 1
        for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            for i in range(1, self.size):
                if (0 <= row + i*direction[0] < self.size and
                    0 <= col + i*direction[1] < self.size and
                    self.state[0, row + i*direction[0], col + i*direction[1]] in [1, 3]):
                    ship_size += 1
                else:
                    break
        return ship_size       
        
# Créer l'environnement
#env = BattleShipEnv(size=10,ships=[5,4,4,3,2])
env = BattleShipEnv(size=6,ships=[4,3])
#env = BattleShipEnv(size=4,ships=[3])
from stable_baselines3.common.monitor import Monitor
env = Monitor(env, "./ppo_battleship_tensorboard/")

check_env(env)

# Il est généralement bon de vectoriser l'environnement
env = DummyVecEnv([lambda: env])

#env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)


import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Neural network for predicting action values
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int=128):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.spaces['grid'].shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.spaces['grid'].sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        grid_obs = observations['grid']
        return self.linear(self.cnn(grid_obs))
    
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
)
        
# Initialize agent
model = MaskablePPO("MultiInputPolicy", env, gamma=0.4, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./ppo_battleship_tensorboard/")

# Créer l'agent
#model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_battleship_tensorboard/")

from stable_baselines3.common.callbacks import BaseCallback
import tensorflow as tf
import os
from stable_baselines3.common.results_plotter import load_results, ts2xy

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(TensorboardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              self.logger.record('rollout/mean_reward', float(mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"New best mean reward: {mean_reward} - Saving best model to {self.save_path}.zip")
                  self.model.save(self.save_path)

        return True

# Create the callback: check every 1000 steps
callback = TensorboardCallback(check_freq=1000, log_dir="./ppo_battleship_tensorboard/")

# Entraîner l'agent
model.learn(total_timesteps=100000, callback=callback)
print('done')
# Sauvegarder l'agent entraîné
model.save("ppo_battleship")

# Charger l'agent entraîné
model = PPO.load("ppo_battleship")

from tensorboardX import SummaryWriter

writer = SummaryWriter(logdir="./ppo_battleship_tensorboard/")

# Tester l'agent
obs = env.reset()
print('test')
episode_lengths = []
episode_length = 0
total_won_games = 0
total_lost_games = 0
invalid_moves_per_game = 0  # Add this line
nb_episodes = 0 
episode_reward = 0 
for i in range(50000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    #print(obs)  # pour voir une partie !
    #env.render()
    episode_length += 1
    invalid_moves_per_game += info[0]['total_invalid_moves']  # Add this line
    episode_reward += rewards
    if dones:
        nb_episodes += 1
        episode_lengths.append(episode_length)
        episode_length = 0
        print("Invalid moves this game:", invalid_moves_per_game)  # Add this line
        invalid_moves_per_game = 0  # Add this line
        if 'total_invalid_moves' in info[0] and info[0]['total_invalid_moves'] > 0:
            total_lost_games += 1
        else:
            total_won_games += 1
        writer.add_scalar('Total_invalid_moves', info[0]['total_invalid_moves']/nb_episodes, i)
        writer.add_scalar('Reward', episode_reward, i)
        episode_reward = 0
        
    if len(episode_lengths) > 0:
        writer.add_scalar('Average_episode_length', np.mean(episode_lengths), i)
    
    print(i)
writer.close()

for i in range(30):
    print("Step number:", i)
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(obs)  # pour voir une partie !
    
print("boucle finie")
print("Total won games:", total_won_games)
print("Total lost games:", total_lost_games)