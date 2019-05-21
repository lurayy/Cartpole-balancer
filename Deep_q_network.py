import gym 
import tensorflow  as tf 
import numpy as np 
from collections import deque
import random


learning_rate = 0.001
discount_rate = 0.95
memory_size = 1000000
batch_size = 20
exploration_max = 1.0
exploration_min = 0.01
exploration_decay = 0.995

class Agent:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = exploration_max
        self.action_space = action_space
        self.memory = deque(maxlen = memory_size)

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(24, input_shape = (observation_space,), activation = 'relu'))
        self.model.add(tf.keras.layers.Dense(24,activation = 'relu'))
        self.model.add(tf.keras.layers.Dense(self.action_space, activation = "linear"))
        self.model.compile(loss= "mse", optimizer = tf.keras.optimizers.Adam(lr = learning_rate))
        
        # json_file = open("models/dqn_with_er.json",'r')
        # loaded_model_json = json_file.read()
        # json_file.close()
        # self.model = tf.keras.models.model_from_json(loaded_model_json)
        # self.model.load_weights("models/dqn_with_er.h5")
        
    def save_to_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def get_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])
    
    def experience_replay(self):
        if len(self.memory)< batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            q_update = reward                                                                       #if done, final q value will the final reward
            if not done:
                q_update = reward + discount_rate*np.amax(self.model.predict(next_state)[0])
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose = 0)
        self.exploration_rate *= exploration_decay
        self.exploration_rate = max(exploration_min, self.exploration_rate)

    def save(self):
        model_json = self.model.to_json()
        with open("models/dqn_with_er.json","w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("models/dqn_with_er.h5")
        print("***************************** Model Saved ******************************")


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    observation_space = env.observation_space.shape[0]
    print("Observation Space = ",observation_space)
    action_space = env.action_space.n 
    agent = Agent(observation_space, action_space)
    run =  0
    render = False
    while True:
        run += 1 
        state = env.reset()
        if render == True:
            env.render()
        state = np.reshape(state, [1,observation_space])
        step = 0
        while True:
            step += 1
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            reward = reward if not done else -reward
            next_state = np.reshape(next_state, [1,observation_space])
            agent.save_to_memory(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("Number of runs : ",run," with exploration : ",agent.exploration_rate, " with score : ", step)
                break
            agent.experience_replay()
        if(run % 10 == 0) : 
            agent.save()
         if(run >90):
             render= True
        if (run == 100):
            break
    env.close()