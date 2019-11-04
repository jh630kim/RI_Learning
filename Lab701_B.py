# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 17:12:39 2019

@author: user
"""

import numpy as np
import tensorflow as tf
import random
import gym
from collections import deque
import _model_dqn_701_A as m

env = gym.make('CartPole-v0')
# env = gym.wrappers.Monitor(env, 'gym-results/', force=True)

INPUT_SIZE = env.observation_space.shape[0]     # 4
OUTPUT_SIZE = env.action_space.n                # 2

DISCOUNT_RATE = 0.9
REPLAY_MEMORY = 50000
MAX_EPISODE = 500
BATCH_SIZE = 64
MIN_E = 0.0     # minimum epsilon for E-greedy
EPSILON_DECAYING_EPISODE = MAX_EPISODE * 0.01


def bot_play(dqn: m.DQN) -> None:
    """Runs a single episode with rendering and prints a reward

    Args:
        dqn (dqn.DQN): DQN Agent
    """
    _state = env.reset()
    _reward_sum = 0
    while True:
        env.render()
        _action = np.argmax(dqn.predict(_state))
        _state, _reward, _done, _ = env.step(action)
        _reward_sum += _reward
        if done:
            print("Total score: {}".format(_reward_sum))
            break


# def simple_replay_train(DQN, train_batch):
def train_sample_replay(dqn: m.DQN, train_batch: list) -> list:
    """Prepare X_batch, y_batch and train them

    Recall our loss function is
        target = reward + discount * max Q(s',a)
                 or reward if done early
        Loss function: [target - Q(s, a)]^2

    Hence,
        X_batch is a state list
        y_batch is reward + discount * max Q
                   or reward if terminated early

    Args:
        dqn (m.DQN): DQN Agent to train & run
        train_batch (list): Mini batch of Sample Replay memory
            Each element is a tuple of (state, action, reward, next_state, done)

    Returns:
        loss: Returns a list of cost and train
    """
    x_stack = np.empty(0).reshape(0, dqn.input_size)
    y_stack = np.empty(0).reshape(0, dqn.output_size)

    for _state, _action, _reward, _next_state, _done in train_batch:
        Q_pred = dqn.predict(_state)

        if done:
            Q_pred[0, _action] = reward
        else:
            Q_pred[0, _action] = reward + DISCOUNT_RATE * np.max(dqn.predict(_next_state))

        x_stack = np.vstack([x_stack, _state])
        # x_batch = np.vstack([x[0] for x in train_batch])
        y_stack = np.vstack([y_stack, Q_pred])

    return dqn.update(x_stack, y_stack)

# (1) replay 메모리를 생성
replay_buffer = deque(maxlen=REPLAY_MEMORY)

with tf.Session() as sess:
    # (2) 네트워크 구성
    mainDQN = m.DQN(sess, INPUT_SIZE, OUTPUT_SIZE)
    mainDQN.build_network(h_size=10, l_rate=0.01)
    sess.run(tf.global_variables_initializer())

    for episode in range(MAX_EPISODE):
        e = 1./((episode/10)+1)
        done = False
        step_count = 0
        state = env.reset()

        while not done:
            if np.random.rand() < e:
                action = env.action_space.sample()
            else:
                action = np.argmax(mainDQN.predict(state))

            next_state, reward, done, _ = env.step(action)

            if done:
                reward = -1

            replay_buffer.append((state, action, reward, next_state, done))

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
                cost, _ = train_sample_replay(mainDQN, minibatch)
            print("Cost: ", cost)
    bot_play(mainDQN)

# https://mclearninglab.tistory.com/35
