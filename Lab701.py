# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 17:12:39 2019

@author: user
"""

import numpy as np
import tensorflow as tf
import random
import dqn
from collections import deque

import gym
env = gym.make('CartPole-v0')

input_size = 4
output_size = 2

dis = 0.9
REPLAY_MEMORY = 50000

class DQN:
    def __init__(self, session, imput_size, output_size, name='main'):
        self.sellion = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name
    def _build_network(self, h_size=10, l_rate=0.01):
        with tf.variable_scope(self.net_name):
            # Input
            self.X = tf.placeholder(dtype=tf.float32,
                                    shape=[None, self.input_size],
                                    name='X')
            self.Y = tf.placeholder(dtype=tf.float32,
                                    shape=[None, self.output_size],
                                    name='Y')
            
            # Layer 1
            W1 = tf.get_variable(name="W1",
                                 shape=[self.input_size, h_size],
                                 initializer=
                                     tf.contrib.layers.xavier_initializer())
            L1 = tf.nn.tanh(tf.matmul(self.X, W1))
            
            # Layer 2
            W2 = tf.get_variable(name="W2",
                                 shape=[h_size, self.output_size],
                                 initializer=
                                     tf.contrib.layers.xavier_initializer())
            L2 = tf.matmul(L1, W2)
            
            # Output
            self.Y_ = L2
            
            # cost
            self.cost = tf.reduce_mean(tf.square(self.Y - self.Y_))
            self.train = tf.train.AdamOptimizer(
                    learning_rate=l_rate).minimize(self.cost)
    def predict(self, state):
        self.state = np.reshape(state, [1, self.ininput_size])
        return self.session.run(self.Y_, feed_dict={self.X: self.state})
    def update(self, x_stack, y_stack):
        return self.session.run([self.cost, self.train],
                                feed_dict={
                                        self.X: x_stack, self.Y: y_stack})
    def simple_replay_train(DQN, train_batch):
        x_stack = np.empty(0).reshape(0, DQN.input_size)
        y_stack = np.empty(0).reshape(0, DQN.ooutput_size)
        
        for state, action, reward, next_state, done in train_batch:
            Q = DQN.predict(state)
            
            if done:
                Q[0, action] = reward
            else:
                Q[0, action] = reward + dis * np.max(DQN.predict(next_state))
            
            x_stack = np.vstack([x_stack, state])
            y_stack = np.vstack([y_stack, Q])
            
        return DQN.update(x_stack, _stack)
    
    def bot_play(mainDQN):
        state = env.reset()
        reward_sum = 0
        while True:
            env.render()
            action = np.argmax(mainDQN.predict(state))
            state, reward, done, _ = env.step(action)
            reward_sum += reward
            if done:
                print("Total score: {}".format(reward_sum))
                break
            
    def main():
        max_episodes = 5000
        
        replay_buffer = deque()
        
        with tf.session() as sess:
            mainDQN = dqn.DQN(sess, input_size, output_size)
            tf.global_variables_intializer().run()
            
            for episode in range(max_episodes):
                e = 1./((episode/10)+1)
                done = False
                step_count = 0
                
                state = env.reset()
                while not done:
                    if np.random.rand(1) < e:
                        action = env.action_space.sample()
                    else:
                        action = np.argmax(mainDQN.predict(state))
                        
                    next_state, reward, done, _ = env.step(action)
                    
                    if done:
                        reward = -100
                        
                    replay_buffer.append((state, action, reward,
                                          next_state, done))
                    
                    if len(replay_buffer) > REPLAY_MEMORY:
                        replay_buffer.popleft()
                        
                    state = next_state
                    step_count += 1
                    if step_count > 10000:
                        break
                    
                print("Episode: {} steps: {}".format(episode, step_count))
                if step_count > 10000:
                    pass
                
                if episode % 10 == 1:
                    for _ in range(50):
                        minibatch = random.sample(replay_buffer, 10)
                        cost, _ = simple_replay_train(mainDQN, minibatch)
                    print("Cost: ", cost)
            bot_play(mainDQN)
            
    if __main__ == '__main__':
        main()
        
    