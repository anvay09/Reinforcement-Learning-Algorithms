import gym
import itertools
from tiles3 import IHT, tiles
import numpy as np
import math
import matplotlib.pyplot as plt
import os

def plot_rewards(episode_rewards):
    episode_rewards = np.array(episode_rewards)
    average_rewards = [np.mean(episode_rewards[:i]) for i in range(0, len(episode_rewards))]
    plt.plot(episode_rewards, color='green')
    plt.plot(average_rewards, color='red')
    plt.xlabel('episode')
    plt.ylabel('episode rewards')
    plt.yscale('log')
    plt.show()

def mytiles(features, iht, numTilings, action):
    scaleFactor_p = 5/(4.8)
    scaleFactor_v = 10/(4)
    scaleFactor_a = 10/(0.836)
    scaleFactor_vt = 10/(4)

    return np.array(tiles(iht, numTilings, list((features[0]*scaleFactor_p, features[1]*scaleFactor_v, features[2]*scaleFactor_a, features[3]*scaleFactor_vt)), [action]))

def Q(w, x):
    return np.sum(w[x])

class Agent:
    def decide(self, features, iht, num_tilings, w, epsilon, num_actions):
        if np.random.random() > epsilon:
            Q_vals = [0] * num_actions
            for a in range(num_actions):
                x = mytiles(features, iht, num_tilings, a)
                Q_vals[a] = Q(w, x)
            
            action = np.argmax(Q_vals)
            
        else:
            action = np.random.randint(0, num_actions)

        return action 

agent = Agent()

def play_once(env, agent, num_actions, num_tilings, learning_rate, gamma, epsilon, w, iht, render=False, verbose=False):
    observation, _= env.reset()
    episode_reward = 0.

    for step in itertools.count():
        if render:
            env.render()

        old_features = list(observation)
        action = agent.decide(old_features, iht, num_tilings, w, epsilon, num_actions)
        observation, reward, terminated, truncated, info_dict = env.step(action)
        features = list(observation)
        
        # print(features)

        episode_reward += reward
        if episode_reward >= 10000:
            break

        if terminated:
            reward = -1
        else:
            reward = 0
        
        next_action = agent.decide(features, iht, num_tilings, w, epsilon, num_actions)

        x_t = mytiles(old_features, iht, num_tilings, action)
        x_t1 = mytiles(features, iht, num_tilings, next_action)
        
        Q_t = Q(w, x_t)
        Q_tp1 = Q(w, x_t1)
        
        if terminated:
            target = reward
            w[x_t] += learning_rate * (target - Q_t)

            break

        else:
            target = reward + gamma * Q_tp1
            w[x_t] += learning_rate * (target - Q_t)

    if verbose:
        print('get {} rewards in {} steps'.format(episode_reward, step + 1))
        
    return episode_reward, w


if __name__ == '__main__':
    episode_rewards = []
    agent = Agent()
    weight_file = 'weights_CartPole-v0_4.npy'
    env = gym.make('CartPole-v0', render_mode="human")

    num_features = 4
    num_actions = 2
    feature_ranges = [[-2.4, 2.4], [-2, 2], [-0.418, 0.418], [-2, 2]]
    num_tilings = 32
    learning_rate = 0.1 / num_tilings
    gamma = 1
    epsilon = 0
    maxSize = 500000
    iht = IHT(maxSize)

    if os.path.isfile(weight_file):
        print('Loading saved weights...')
        with open(weight_file, 'rb') as f:
            w = np.load(f, allow_pickle = True)
    else:
        w = np.zeros(maxSize)

    for i in range(200):
        print('Episode', i)
        e, w = play_once(env, agent, num_actions, num_tilings, learning_rate, gamma, epsilon, w, iht, True, True)
        episode_rewards.append(e)
        
        print('Saving weights...')
        with open(weight_file, 'wb') as f:
            np.save(f, w, allow_pickle = True)

        print('average episode rewards = {}'.format(np.mean(episode_rewards[-100:])))

    plot_rewards(episode_rewards)