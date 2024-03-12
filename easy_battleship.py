#%%
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.optimizers.legacy import Adam
from keras.layers.convolutional import Conv2D, MaxPooling2D
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import warnings
from rl.policy import EpsGreedyQPolicy

warnings.filterwarnings("ignore")

class EasyBattleShipEnv(gym.Env):
    def __init__(self, size=10, ships=[3,2]):
        super(EasyBattleShipEnv, self).__init__()

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = Discrete(size)
        self.observation_space = Box(low=0, high=3, shape=(size,1), dtype=int)

        # Initialize state
        self.state = np.zeros((size,1), dtype=int)
        self.size = size
        self.ships = ships
        self.nb_steps = 0
        self.nb_invalid_move = 0

    def step(self, action):
        # Execute one time step within the environment
        self.nb_steps += 1

        if self.state[action] == 0:
            self.state[action] = 2  # Miss
            reward = -1
        elif self.state[action] == 1:
            self.state[action] = 3  # Hit
            reward = 1
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
        self.state = np.zeros((size,1), dtype=int)
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
            start_idx = np.random.randint(self.size)
            end_idx = start_idx + ship_size
            if self.state[start_idx:end_idx].sum() == 0:
                self.state[start_idx:end_idx] = 1
                break


def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,states[0])))
    model.add(Dense(256, activation='relu', input_shape=states))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model



size = 10  # or whatever the size of your grid is

env = EasyBattleShipEnv(size=size)
states = env.observation_space.shape # (10,1)
actions = env.action_space.n # 10

print("States :",states, "Actions :",actions)

model = build_model(states, actions)


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    #policy = EpsGreedyQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn
print('start')
dqn = build_agent(model, actions)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

_ = dqn.test(env, nb_episodes=1, visualize=False)
# pour voir une partie !


scores = dqn.test(env, nb_episodes=20, visualize=False)
print(np.mean(scores.history['episode_reward']))
# tests des scores

#dqn.save_weights('dqn_weights.h5f', overwrite=True)
# à décommenter si on veut sauvegarder les poids du modèle
# %%
