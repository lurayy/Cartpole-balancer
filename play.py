import gym 
import tensorflow as tf 
import numpy as np 

class Agent:

    def __init__(self, observation_space, action_space):
        self.action_space = action_space

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(24, input_shape = (observation_space,), activation = 'relu'))
        self.model.add(tf.keras.layers.Dense(24,activation = 'relu'))
        self.model.add(tf.keras.layers.Dense(self.action_space, activation = "linear"))
        
        json_file = open("models/dqn_with_er.json",'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = tf.keras.models.model_from_json(loaded_model_json)
        self.model.load_weights("models/dqn_with_er.h5")
        
    def get_action(self,state):
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n 
    agent = Agent(observation_space, action_space)
    for episode in range(2):
        state = env.reset()
        state = np.reshape(state, [1,observation_space])
        for step in range(500):
            env.render()
            action = agent.get_action(state)
            state, reward, done, info = env.step(action)
            state = np.reshape(state, [1,observation_space])
            if done:
                print("For episode ",episode," Score : ",step)
                break
    env.close()