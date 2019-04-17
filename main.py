import gym
from gym import wrappers


def main():
    list_rewards = []
    for i in range(100):
        env = gym.make('SpaceInvaders-v0')
        env = wrappers.Monitor(env, "./gym-results")

        env.reset()
        total_reward = 0

        is_done = False
        while not is_done:
            _, reward, is_done, _ = env.step(env.action_space.sample())
            total_reward += reward

        list_rewards.append(total_reward)
        env.close()

    print(max(list_rewards))
    print(list_rewards.index(max(list_rewards)))


if __name__ == '__main__':
    main()
