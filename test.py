import gym
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
#from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


#Création de l'environnement
env = gym.make("CartPole-v1", render_mode = "human")

states = env.observation_space.shape[0] #Nombre d'états possibles, ici 4
actions = env.action_space.n #Nombre d'actions possibles, ici 2 (droite ou gauche)

print(states, actions)

#Création du modèle de l'agent
model = Sequential()
model.add(Flatten(input_shape=(1,states)))
model.add(Dense(24, activation = "relu"))
model.add(Dense(24, activation = "relu"))
model.add(Dense(actions, activation = "linear"))

agent = DQNAgent(
    model = model,
    memory = SequentialMemory(limit = 50000, window_length=1), #Mémoire de l'agent
    policy = BoltzmannQPolicy(), #Politique de prise de décision
    nb_actions = actions, #Nombre d'actions possibles
    nb_steps_warmup = 10, #Nombre d'étapes avant de commencer à apprendre
    target_model_update = 0.01 #Taux de mise à jour du modèle cible
)

#Entraînement de l'agent
agent.compile(tf.keras.optimizers.legacy.Adam(learning_rate=1e-3), metrics = ["mae"])
agent.fit(env, nb_steps = 100000, visualize = False, verbose = 1) 

#Test de l'agent
results = agent.test(env, nb_episodes = 10, visualize = True)
print(results.history["episode_reward"])