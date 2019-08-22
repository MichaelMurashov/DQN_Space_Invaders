import gym
from gym import wrappers
import time
import random
from tqdm import tqdm
from keras.models import load_model

import config
from model import *
from ring_buffer import ReplayBuffer
from preprocess import *

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable INFO massages from Tensorflow


def get_epsilon_for_iteration(iter_num):
    if iter_num is 0:
        return config.initial_epsilon
    elif iter_num > config.final_iteration_num:
        return config.final_epsilon
    else:
        return -0.0168*(iter_num**2) + 3.0867*iter_num + 41.3745


def q_iteration(env, model, target_model, curr_state, iter_num, memory):
    # Choose epsilon based on the iteration
    epsilon = get_epsilon_for_iteration(iter_num)

    # Choose the action
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = choose_best_action(model, curr_state)

    new_state = []
    is_terminal = False
    reward = 0
    # Play one game iteration (note: according to the next paper, you should actually play 4 times here)
    for i in range(config.num_frames):
        new_frame, new_reward, is_terminal, _ = env.step(action)
        reward += new_reward
        new_state.append(new_frame)

    new_state = preprocess_state(new_state)
    memory.append(curr_state, action, reward, new_state, is_terminal)

    # Fit the model nn sample batch
    if len(memory) > config.min_frames_for_train:
        curr_states, actions, rewards, next_states, is_terminals = memory.sample_batch(config.batch_size)
        fit_batch(model, target_model, curr_states, actions, rewards, next_states, is_terminals)

    return new_state, is_terminal, reward


def main(mode, num):
    if mode is 'random':

        list_rewards = []
        for i in range(1):
            env = gym.make('SpaceInvaders-v4')
            env = gym.wrappers.Monitor(env, "./gym-results")

            env.reset()
            total_reward = 0

            is_done = False
            while not is_done:
                _, reward, is_done, _ = env.step(env.action_space.sample())
                total_reward += reward

            # print(total_reward)
            list_rewards.append(total_reward)
            env.close()

        print(max(list_rewards))

    elif mode is 'train':
        model = atari_model(config.num_actions)
        target_model = load_model(f'checkpoints/0.h5', custom_objects={'huber_loss': huber_loss})

        env = gym.make('SpaceInvaders-v4')

        memory = ReplayBuffer(config.num_iterations)

        rewards = []

        for i in range(config.num_iterations):
            env.reset()

            state = []
            for j in range(config.num_frames):
                frame, _, _, _ = env.step(env.action_space.sample())
                state.append(frame)
            state = preprocess_state(state)

            is_terminal = False
            reward = 0
            start = time.time()
            while not is_terminal:
                state, is_terminal, new_reward = q_iteration(env, model, target_model, state, i, memory)
                reward += new_reward

            rewards.append(reward)
            game_time = time.time() - start
            if i % 10 == 0 and i is not 0:
                print(f'time: {game_time}, mean reward: {np.mean(rewards)}')

            if i % 1000 == 0:
                print(f'SCORE {reward}')
                model.save(f'checkpoints/{i}.h5')
                target_model = load_model(f'checkpoints/{i}.h5', custom_objects={'huber_loss': huber_loss})
                # target_model = load_model(f'checkpoints/{i}.h5')

    elif mode is 'test':
        model = load_model(f'checkpoints/{num}.h5', custom_objects={'huber_loss': huber_loss})

        env = gym.make('SpaceInvaders-v4')
        # env = wrappers.Monitor(env, "./gym-results")
        env.reset()

        state = []
        for j in range(config.num_frames):
            frame, _, _, _ = env.step(env.action_space.sample())
            state.append(frame)
        state = preprocess_state(state)

        is_done = False
        reward = 0
        while not is_done:
            action = choose_best_action(model, state)
            state = []
            for i in range(config.num_frames):
                new_state, new_reward, is_done, _ = env.step(action)
                env.render()
                time.sleep(0.03)
                state.append(new_state)
                reward += new_reward
            state = preprocess_state(state)

        print(f'REWARD = {reward}')


if __name__ == '__main__':
    act = 'test'
    main(act, 13000)
