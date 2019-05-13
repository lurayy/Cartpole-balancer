import gym 
import random 
import numpy as np 
from deepq import DeepQ

env = gym.make("CartPole-v0")

time = 20
steps = 10000
exploration_rate = 1
leraning_rate = 0.0025
discount_factor = 0.99
learn_start = 128
memory_size = 1000000

deepQ = DeepQ(4, 2, memory_size, discount_factor, leraning_rate, learn_start)
deepQ.loadModel()
avg_steps = []

for epoch in range(time):
    observation =env.reset()
    for step in range(steps):
        env.render()
        qValues = deepQ.getQValues(observation)

        action  = deepQ.selectAction(qValues, exploration_rate)

        new_observation, reward, done, info = env.step(action)
        observation = new_observation
        if done: 
                avg_steps.append(step)
                print("Done after {} Steps.".format(step))
                break
    exploration_rate *= 0.995
    # explorationRate -= (2.0/epochs)
    exploration_rate = max (0.05, exploration_rate)
env.close()