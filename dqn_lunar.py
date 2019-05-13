import gym
import random
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 
from collections import deque, namedtuple


env = gym.make('LunarLander-v2')
env.seed(0)
print('State Shape : ',env.observation_space.Shape)
print('Number Of action : ',env.action_space.Shape)
agent = Agent(state_size = int(env.observation_space.Shape), action_size = int(env.action_space.Shape), seed =0)

state = env.reset()
for _ in range(200):
    action = agent.act(200)
    env.render()
    state, reward, done, _ = env.step(action)
    if done: 
        break

env.close()

#eps = value of epsilon, for epsilon-greedy action selection - for exploration
#eps_decay = multiplicative factor (per episode) for decreasing epsilon 
def train(n_episodes = 2000, max_t = 1000, eps_start = 1.0, eps_end = 0.01, eps_decay = 0.995):
    scores  = []
    scores_window = deque(maxlen = 100) # last 100 scores 
    eps = eps_start
    for i_episode in range(1, n_episodes+1):
        state =env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state,eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            # torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            #save the session
            break
        return scores

def plot_scores(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

def show_smart_agent():
#    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    for i in range(3):
        state = env.reset()
        for j in range(300):
            action = agent.act(state)
            env.render()
            state, reward, done, _ = env.step(action)
            if done:
                break
    env.close()


# Deep neural network that maps state to action values 
class QNetwork():
    def __init__(self, state_size, action_size, seed):
        tf.set_random_seed(seed)
        self.state_size = state_size
        self.X = tf.placeholder("float",[None,state_size])
        self.input_layer = tf.keras.layers.Dense(state_size, activation = tf.nn.relu)(self.X)
        self.hidden_layer_one = tf.keras.layers.Dense(state_size, activation = tf.nn.relu)(self.input_layer)
        self.output_layer = tf.keras.layers.Dense(action_size, activation = tf.nn.relu)(self.hidden_layer_one)

    def forward(self, state, sess): 
        Y = sess.run(self.output_layer, feed_dict = {self.X: state.reshape(1,self.state_size)})
        return Y 


#Coding the Agent 

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor for rewards   
TAU = 1e-3              # for soft update of traget parameters (weights)
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network (weights)

class Agent():

    def __init__(self,state_size,action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed

        self.qnetwork_local = QNetwork(state_size, action_size, seed)
        self.qnetwork_target = QNetwork(state_size, action_size, seed)
        # self.optimizer = tf.keras.optimizers.Adam(self.qnetwork_local.parameters(), lr = LR)


class ReplayBuffer: 
    def __init__(self,action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experience = random.sample(self.memory, k = self.batch_size)

        # states = 