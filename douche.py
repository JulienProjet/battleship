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

class BattleShipEnv(gym.Env):
    def __init__(self, size=10, ships=[5,4,3,3,2]):
        super(BattleShipEnv, self).__init__()

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = Discrete(size * size) #Regarder pour donner le numéro de ligne & colonne en 2 inputs MultiDiscrete
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
        row = action // self.size
        col = action % self.size

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
        self.state = np.zeros((size, size), dtype=int)
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

class ShowerEnv(Env):
    def __init__(self):
        # Actions we can take, down, stay, up
        self.action_space = Discrete(3)
        # Temperature array
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        # Set start temp
        self.state = 38 + random.randint(-3,3)
        # Set shower length
        self.shower_length = 60
        
    def step(self, action):
        # Apply action
        # 0 -1 = -1 temperature
        # 1 -1 = 0 
        # 2 -1 = 1 temperature 
        self.state += action -1 
        # Reduce shower length by 1 second
        self.shower_length -= 1 
        
        # Calculate reward
        if self.state >=37 and self.state <=39: 
            reward =1 
        else: 
            reward = -1 
        
        # Check if shower is done
        if self.shower_length <= 0: 
            done = True
        else:
            done = False
        
        # Apply temperature noise
        #self.state += random.randint(-1,1)
        # Set placeholder for info
        info = {}
        
        # Return step information
        return self.state, reward, done, info

    def render(self):
        # Implement viz
        pass
    
    def reset(self):
        # Reset shower temperature
        self.state = 38 + random.randint(-4,4)
        # Reset shower time
        self.shower_length = 60 
        return self.state

env = ShowerEnv()

def create_model(input_shape, action_space):
    model = Sequential()
    #model.add(Reshape((input_shape[1], input_shape[2], input_shape[0]), input_shape=input_shape))
    #model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(action_space, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=1e-3))
    return model

def build_model(states, actions):
    model = Sequential()    
    model.add(Dense(256, activation='relu', input_shape=states))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

size = 10  # or whatever the size of your grid is
input_shape = (1,size, size) # les réseaux de neurones tu coco faut une dimension en plus jsp quoi
states = env.observation_space.shape # (10, 10) (le plateau quoi)
actions = env.action_space.n # 100 !



#model = create_model(input_shape, actions)
model = build_model(states, actions)
print(model.summary())



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