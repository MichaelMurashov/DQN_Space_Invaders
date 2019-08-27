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
        # return -0.0168*(iter_num**2) + 3.0867*iter_num + 41.3745
        return -0.0001*(iter_num) + 1.0001


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
    if mode == 'random':

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

    elif mode == 'train':
        model = atari_model(config.num_actions)
        target_model = load_model(f'checkpoints/0.h5', custom_objects={'huber_loss': huber_loss})

        env = gym.make('SpaceInvaders-v4')

        memory = ReplayBuffer(config.buffer_size)

        rewards, game_times = [], []

        start = time.time()
        for i in range(config.num_iterations):
            env.reset()

            state = []
            for j in range(config.num_frames):
                frame, _, _, _ = env.step(env.action_space.sample())
                state.append(frame)
            state = preprocess_state(state)

            is_terminal = False
            reward = 0
            start_game = time.time()
            while not is_terminal:
                state, is_terminal, new_reward = q_iteration(env, model, target_model, state, i, memory)
                reward += new_reward

            game_times.append(np.float16(time.time()-start_game))
            rewards.append(np.uint16(reward))

            if i % 5 == 0 and i is not 0:
                print(f'#{i} mean game time: {round(np.mean(game_times[-6:-1]), 2)}, mean reward: {round(np.mean(rewards[-6:-1]), 2)}')

            if i % 500 == 0:
                print('======== Save checkpoint ========')
                print(f'Elapsed time {round(time.time()-start, 2)}s')
                print(f'mean game time: {round(np.mean(game_times[-501:-1]), 2)}, mean reward: {round(np.mean(rewards[-501:-1]), 2)}')
                print('=================================')
                model.save(f'checkpoints/{i}.h5')
                target_model = load_model(f'checkpoints/{i}.h5', custom_objects={'huber_loss': huber_loss})

    elif mode == 'test':
        print(f'Runing test mode with model after {num} epochs')
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
                new_frame, new_reward, is_done, _ = env.step(action)
                env.render()
                time.sleep(0.03)
                state.append(new_frame)
                reward += new_reward
            state = preprocess_state(state)

        print(f'REWARD = {reward}')

    else:
        print('Error mode!')

if __name__ == '__main__':
    act = sys.argv[1]
    num = sys.argv[2] if len(sys.argv) is 3 else -1
    main(act, num)
