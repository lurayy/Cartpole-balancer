import tensorflow as tf 
import gym 
import numpy as np 

n_states = 40
iter_max = 10000
init_lr = 1.0
min_lr = 0.003
gamma = 1.0
t_max = 10000
eps = 0.02


def run_episode(env, policy = None, render = False):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    for _ in range ( t_max):
        if render:
            env.render()
        if policy is None:
            action = env.action_space.sample()
        else: 
            a,b = obs_to_state(env, obs)
            action = policy[a][b]
        obs, reward, done, _ = env.step(action)
        total_reward += gamma** step_idx * reward
        step_idx += 1 
        if done:
            break
    return total_reward

def obs_to_state(env, obs):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low)/n_states
    a = int((obs[0] - env_low[0])/env_dx(0))
    b = int((obs[1] - env_low[1])/env_dx[1])
    return a, b

if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    q_table = np.zeros((n_states, n_states, 3))
    print(q_table)