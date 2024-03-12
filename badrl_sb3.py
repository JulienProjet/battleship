from gym import Env
from gym.spaces import Discrete, Box, MultiDiscrete
import numpy as np
import random
import gym
import numpy as np
import warnings

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

warnings.filterwarnings("ignore")

class BattleShipEnv(gym.Env):
    def __init__(self, size=10, ships=[5,4,3,3,2]):
        super(BattleShipEnv, self).__init__()

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = MultiDiscrete(np.array([size,size]))
        self.observation_space = Box(low=0, high=3, shape=(size, size), dtype=int)

        # Initialize state
        self.state = np.zeros((size, size), dtype=int)
        self.size = size
        self.ships = ships
        self.nb_steps = 0
        self.nb_invalid_move = 0

    def step(self, action):
        # Execute one time step within the environment
        self.nb_steps += 1

        #row = action // self.size
        #col = action % self.size
        row = action[0]
        col = action[1]

        if self.state[row, col] == 0:
            self.state[row, col] = 2  # Miss
            reward = -1
        elif self.state[row, col] == 1:
            self.state[row, col] = 3  # Hit
            reward = 100
        else:
            #reward = -1  # Invalid move
            #return self.step(self.action_space.sample())
            self.nb_invalid_move +=1
            reward = -10  # Invalid move
            done = False
            info = {}
            return self.state, reward, done, info

        done = (self.state == 1).sum() == 0  # Game is over if no ships are left
        if done :
            print("\nGame over, number of steps:", self.nb_steps)
            print("Number of invalid moves:", self.nb_invalid_move)

        return self.state, reward, done, {}

    def reset(self):
        #print("Before reset:", self.state)
        # Reset the state of the environment to an initial state
        self.state = np.zeros((self.size, self.size), dtype=int)
        self.nb_steps = 0
        self.nb_invalid_move = 0
        for ship in self.ships:
            self.place_ship(ship)
        #print("After reset:", self.state)
        return self.state

    def render(self, mode='human'):
        # Render the environment to the screen
        print(self.state)



    def place_ship(self, ship_size):
        # Place a ship of given size on the board
        while True:
            direction = np.random.choice(["vertical", "horizontal"])
            if direction == "horizontal":
                row = np.random.randint(self.size)
                col = np.random.randint(self.size - ship_size + 1)
                if self.state[row, col:col+ship_size].sum() == 0:
                    self.state[row, col:col+ship_size] = 1
                    break
            else:  # direction == "vertical"
                row = np.random.randint(self.size - ship_size + 1)
                col = np.random.randint(self.size)
                if self.state[row:row+ship_size, col].sum() == 0:
                    self.state[row:row+ship_size, col] = 1
                    break

#env = BattleShipEnv()

vec_env = make_vec_env(BattleShipEnv, n_envs=4)

model = A2C("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=1000000)