
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, MultiDiscrete, Tuple, Dict
import numpy as np
import random
import numpy as np
import warnings

from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy, MaskableActorCriticCnnPolicy
from stable_baselines3.common.env_checker import check_env

#warnings.filterwarnings("ignore")

class BattleShipEnv(gym.Env):
    def __init__(self, size=10, ships=[5,4,3,3,2]):
        super(BattleShipEnv, self).__init__()

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = MultiDiscrete(np.array([size,size]))
        #self.action_space = Discrete(size * size)
        self.observation_space = Box(low=0, high=255, shape=(2, size, size), dtype=np.uint8)

        # Initialize state
        self.state = np.zeros((3,size, size), dtype=int)
        self.size = size
        self.ships = ships
        self.nb_steps = 0
        self.nb_invalid_move = 0

    def step(self, action):
        # Execute one time step within the environment
        self.nb_steps += 1
        #action = np.unravel_indeaction, (self.size, self.size))
        reward = -0.1
        #row = action // self.size
        #col = action % self.size
        row = action[0]
        col = action[1]
        done=False

        if self.state[0, row, col] == 0:
            self.state[0, row, col] = 2  # Miss
            self.state[2, row, col] = 1  # Add this line
            reward -= 1
            
        elif self.state[0, row, col] == 1:
            self.state[0, row, col] = 3  # Hit
            self.state[1, row, col] = 1  # Add this line
            reward += 10
        else:
            #reward = -1  # Invalid move
            #return self.step(self.action_space.sample())
            self.nb_invalid_move +=1
            reward = -50  # Invalid move
            #done = False
            #info = {}
            #return self.state, reward, done, False, info

        done = (self.state[0] == 1).sum() == 0  # Game is over if no ships are left
        if done == True :
            #done=True
            reward += 15
            print("\nGame over, number of steps:", self.nb_steps)
            print("Number of invalid moves:", self.nb_invalid_move)
        
        info = {'total_invalid_moves': self.nb_invalid_move}
        #return self.state[1:], reward, done, False, info
        normalized_state = (self.state * 255).astype(np.uint8)
        return np.stack([normalized_state[1], normalized_state[2]], axis=0), reward, done, False, info
    
    def reset(self, seed=None, **kwargs):
        #print("Before reset:", self.state)
        # Reset the state of the environment to an initial state
        self.state = np.zeros((3,self.size, self.size), dtype=int)
        self.nb_steps = 0
        self.nb_invalid_move = 0
        for ship in self.ships:
            self.place_ship(ship)
        #print("After reset:", self.state)
        #return self.state[1:]
        normalized_state = (self.state * 255).astype(np.uint8)
        return np.stack([normalized_state[1], normalized_state[2]], axis=0)
    
    def render(self, mode='human'):
        # Render the environment to the screen
        print(self.state)


    def valid_action_mask(self):
        # Generate flat action mask
        mask = np.ones(self.size * self.size, dtype=np.float32)
        mask[self.state.flatten() > 1] = 0  # Set to 0 for cells that have been hit or missed
        return mask
    

    def place_ship(self, ship_size):
        # Place a ship of given size on the board
        while True:
            direction = np.random.choice(["vertical", "horizontal"])
            if direction == "horizontal":
                row = np.random.randint(self.size)
                col = np.random.randint(self.size - ship_size + 1)
                if self.state[0,row, col:col+ship_size].sum() == 0:
                    self.state[0,row, col:col+ship_size] = 1
                    break
            else:  # direction == "vertical"
                row = np.random.randint(self.size - ship_size + 1)
                col = np.random.randint(self.size)
                if self.state[0,row:row+ship_size, col].sum() == 0:
                    self.state[0,row:row+ship_size, col] = 1
                    break

#env = BattleShipEnv()

#vec_env = make_vec_env(BattleShipEnv, n_envs=1)
env = BattleShipEnv(size=84,ships=[5,4,3,3,2])
#env = BattleShipEnv(size=4,ships=[3])
#env = BattleShipEnv(size=7,ships=[4,3])
#env = BattleShipEnv(size=8,ships=[5,4,3])
#model = A2C("MlpPolicy", vec_env, verbose=1)


from stable_baselines3.common.monitor import Monitor
env = Monitor(env, "./ppo_battleship_tensorboard/")

#check_env(env)

env = DummyVecEnv([lambda: env])

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.valid_action_mask()

#check_env(env)

#env = ActionMasker(env, mask_fn)

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

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):  # Change features_dim to 64
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # Modify the CNN architecture
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_space.shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
        )
        # Calculate the output shape after the convolutional layers
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, *observation_space.shape)).shape[1]
        # Use the calculated shape for the input of the linear layer
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs, features_extractor_class=CustomCNN, features_extractor_kwargs=dict(features_dim=128))

#model = PPO(CustomPolicy, env, verbose=1, tensorboard_log="./ppo_battleship_tensorboard/")

#model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_battleship_tensorboard/")
model.learn(total_timesteps=500000,callback=callback)

model.save("best_battleship4_multi_cnn")
#model.load("a2c_battleship")
model = PPO.load("best_battleship4_multi_cnn")

#eval_env = make_vec_env(BattleShipEnv, n_envs=1)
#env = ActionMasker(env, mask_fn)


from tensorboardX import SummaryWriter

writer = SummaryWriter(logdir="./ppo_battleship_tensorboard/")

# Tester l'agent
obs = env.reset()
print(obs)
print('test')
episode_lengths = []
episode_length = 0
total_won_games = 0
total_lost_games = 0
invalid_moves_per_game = 0  # Add this line
nb_episodes = 0 
episode_reward = 0 
for i in range(50000):
    action, _states = model.predict(obs)#, action_masks=np.stack(env.valid_action_mask()))
    obs, rewards, dones, info = env.step(action)
    #print(obs)  # pour voir une partie !
    #print(obs, rewards, dones, info)
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
