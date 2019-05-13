import gym 
import tensorflow as tf 
import numpy as  np 

num_inputs = 4
num_hidden_first = 4
num_hidden_second = 4

num_ouput = 1

initializer = tf.contrib.layers.variance_scaling_initializer()
X = tf.placeholder(tf.float32,shape=[None,num_inputs])
Y = tf.placeholder(tf.float32,shape=[None,num_ouput])

hl_1 = tf.layers.Dense(num_hidden_first, activation = 'relu', kernel_initializer = initializer)(X)
hl_2 = tf.layers.Dense(num_hidden_second, activation = 'relu', kernel_initializer = initializer)(hl_1)

output = tf.layers.Dense(num_ouput, activation = 'sigmoid', kernel_initializer= initializer)(hl_2)
probabilities = tf.concat(axis = 1, values = [output, 1-output])

action = tf.multinomial(probabilities, num_samples = 1 )

init = tf.global_variables_initializer()

env = gym.make('CartPole-v0')

epi = 100
step_limit = 100
avg_steps = []

with tf.Session() as sess:
    init.run()

    for i_epi in range(epi):
        obs = env.reset()
        env.render()

        for step in range(step_limit):
            action_val = action.eval(feed_dict = {X: obs.reshape(1,num_inputs)})
            obs,reward,done,info = env.step(action_val[0][0])

            if done: 
                avg_steps.append(step)
                print("Done after {} Steps.".format(step))
                

print("After {} episode, average steps per game was {}".format(epi, np.mean(avg_steps)))
env.close()
