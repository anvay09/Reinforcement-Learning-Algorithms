import gym
import itertools
import numpy as np
import math
import matplotlib.pyplot as plt
from tiles3 import IHT, tiles
import os

def plot_rewards(episode_rewards):
    episode_rewards = np.array(episode_rewards)
    episode_rewards *= -1
    average_rewards = [np.mean(episode_rewards[:i]) for i in range(0, len(episode_rewards))]
    plt.plot(episode_rewards, color='green')
    plt.plot(average_rewards, color='red')
    plt.xlabel('episode')
    plt.ylabel('episode rewards')
    plt.yscale('log')
    plt.show()

class Agent:
    def decide(self, features, iht, num_tilings, w, epsilon, num_actions):
        if np.random.random() > epsilon:
            Q_vals = [0] * num_actions
            for a in range(num_actions):
                x = mytiles(features, iht, num_tilings, a)
                Q_vals[a] = Q(w, x)
            
            # print(Q_vals)
            action = np.argmax(Q_vals)
            
        else:
            action = np.random.randint(0, num_actions)

        return action 
        
def play_once(env, agent, num_actions, num_tilings, learning_rate, gamma, epsilon, lam, w, iht, render=False, verbose=False):
    observation, _= env.reset()
    episode_reward = 0.
    z = np.zeros(len(w))

    for step in itertools.count():
        if render:
            env.render()

        old_features = list(observation)
        action = agent.decide(old_features, iht, num_tilings, w, epsilon, num_actions)
        observation, reward, terminated, truncated, info_dict = env.step(action)
        features = list(observation)
        
        episode_reward += reward
        
        x_t = mytiles(old_features, iht, num_tilings, action)
        delta = reward

        for ind in x_t:
            delta -= w[ind]
            z[ind] += 1

        if terminated:
            w += learning_rate * delta * z
            break

        else:
            next_action = agent.decide(features, iht, num_tilings, w, epsilon, num_actions)
            x_t1 = mytiles(features, iht, num_tilings, next_action)
            for ind in x_t1:
                delta += gamma * w[ind]

            w += learning_rate * delta * z
            z = gamma * lam * z
            
    if verbose:
        print('get {} rewards in {} steps'.format(episode_reward, step + 1))
        
    return episode_reward, w

def mytiles(features, iht, numTilings, action):
    scaleFactor_p = 8/(1.7)
    scaleFactor_v = 8/(0.14)

    return np.array(tiles(iht, numTilings, list((features[0]*scaleFactor_p, features[1]*scaleFactor_v)), [action]))

def Q(w, x):
    return np.sum(w[x])

if __name__ == '__main__':
    episode_rewards = []
    agent = Agent()
    weight_file = 'weights_mountain_car_sarsa_lambda_0.92.npy'
    env = gym.make('MountainCar-v0', render_mode="human")

    num_features = 2
    num_actions = 3
    feature_ranges = [[-1.2, 0.6], [-0.07, 0.07]]
    num_tilings = 32
    learning_rate = 0.5 / num_tilings
    gamma = 1
    epsilon = 0
    lam = 0.9
    maxSize = 8192
    iht = IHT(maxSize)

    if os.path.isfile(weight_file):
        print('Loading saved weights...')
        with open(weight_file, 'rb') as f:
            w = np.load(f, allow_pickle = True)
    else:
        w = np.zeros(maxSize)

    for i in range(100):
        print('Episode', i)
        e, w = play_once(env, agent, num_actions, num_tilings, learning_rate, gamma, epsilon, lam, w, iht, True, True)
        episode_rewards.append(e)
        
        print('Saving weights...')
        with open(weight_file, 'wb') as f:
            np.save(f, w, allow_pickle = True)

        print('average episode rewards = {}'.format(np.mean(episode_rewards[-100:])))

    plot_rewards(episode_rewards)



