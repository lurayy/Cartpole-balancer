import gym 
import tensorflow as tf 
import numpy as  np 

num_inputs = 4
num_hidden_first = 4
num_hidden_second = 4

num_ouput = 1

# mnist = tf.keras.datasets.mnist
# (train_images, train_labels),(test_images, test_labels) = mnist.load_data()

# num_inputs = 784 #28x28 pix
# num_hidden_first = 512 
# num_hidden_second = 256
# num_hidden_third = 128
# num_output = 10 

initializer = tf.contrib.layers.variance_scaling_initializer()
X = tf.placeholder(tf.float32,shape=[None,num_inputs])
Y = tf.placeholder(tf.float32,shape=[None,num_ouput])

# inputs = tf.keras.Input(tf.float32,)

hl_1 = tf.layers.Dense(num_hidden_first, activation = 'relu', kernel_initializer = initializer)(X)
hl_2 = tf.layers.Dense(num_hidden_second, activation = 'relu', kernel_initializer = initializer)(hl_1)
# hl_3 = tf.layers.Dense(num_hidden_third, activation = 'relu', kernel_initializer = initializer)(hl_2)

output = tf.layers.Dense(num_ouput, activation = 'sigmoid', kernel_initializer= initializer)(hl_2)
probabilities = tf.concat(axis = 1, values = [output, 1-output])

# action = tf.multinomial(probabilities, num_samples = 1 )
action = np.argmax(probabilities)
init = tf.global_variables_initializer()

env = gym.make('CartPole-v0')

epi = 10
step_limit = 500


with tf.Session() as sess:
    obs = env.reset()
    env.render()
    sess.run(init)
    print(obs.reshape(1,num_inputs  ))
    for i_epi in range(epi):
        for step in range(step_limit):
            prob = probabilities.eval(feed_dict = { X: obs.reshape(1,num_inputs)})
            action = np.argmax(prob)
            print(action)
            obs,reward,done,info = env.step(action)
        
        if done:
            print("Done")
            break

env.close()
