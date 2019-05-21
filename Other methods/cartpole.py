import gym 
import random 
import numpy as np 
from deepq import DeepQ

env = gym.make('CartPole-v0')

epochs = 100000
steps = 10000
udpate_target_network = 1000
initial_update_target_network = 1000
exploration_rate = 1
mini_batch_size = 128
learn_start = 128
leraning_rate = 0.0025
discount_factor = 0.99
memory_size = 1000000

last_100_scores = [0] * 100
last_100_scores_index = 0 
last_100_filled = False

deepQ = DeepQ(4, 2, memory_size, discount_factor, leraning_rate, learn_start)

deepQ.initNetwork([30,30,30])

step_counter = 0

for epoch in range(epochs):
    observation = env.reset()
    # print(exploration_rate)
    total_scores = []
    for t in range(steps):
        # env.render()
        qValues = deepQ.getQValues(observation)
        
        action  = deepQ.selectAction(qValues, exploration_rate)

        new_observation, reward, done, info = env.step(action)

        if step_counter >= learn_start:
            deepQ.learnOnMiniBatch(mini_batch_size)
        
        observation = new_observation

        if done:
            last_100_scores[last_100_scores_index] = t 
            if last_100_scores_index >= 100:
                last_100_filled = True
                last_100_scores_index = 0
            # print(last_100_filled)
            if not last_100_filled:
                print("Episode ",epoch," finished after {} timesteps".format(t+1))
            else :
                print( "Episode ",epoch," finished after {} timesteps".format(t+1)," last 100 average: ",(sum(last100Scores)/len(last100Scores)))
            break
        
        step_counter += 1
        if step_counter < udpate_target_network:
            current_update_target_network = initial_update_target_network
        else: 
            current_update_target_network = udpate_target_network
        if step_counter % current_update_target_network == 0:
            deepQ.updateTargetNetwork()
            print ("updating target network")
    if (epoch % 1000 == 0):
        print("saving current model and weight to the disk")
        deepQ.saveModel()
        print("Model saved")

    exploration_rate *= 0.995
    # explorationRate -= (2.0/epochs)
    exploration_rate = max (0.05, exploration_rate)


print("saving current model and weight to the disk")
deepQ.saveModel()
print("Model saved")