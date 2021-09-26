# https://deeplearningcourses.com/c/deep-reinforcement-learning-in-python
# https://www.udemy.com/deep-reinforcement-learning-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import gym
import numpy as np
import matplotlib.pyplot as plt


def get_action(s, w):
    return 1 if s.dot(w) > 0 else 0


def play_one_episode(env, params):
    observation = env.reset()
    done = False
    t = 0

    while not done and t < 10000:
        # env.render()
        t += 1
        action = get_action(observation, params)
        observation, reward, done, info = env.step(action)
        if done:
            break
    return t


if __name__ == "__main__":
    # Tạo môi trường cartpole
    env = gym.make('CartPole-v0')

    #
    time_run = 100
    w_best = None
    max_episode_length = 0
    for time in range(time_run):
        w = np.random.rand(4)
        episode_length = np.empty(100)
        for episode in range(100):
            # Chơi một episode để lấy ra số lần chạy được đến khi terminate
            episode_length[episode] = play_one_episode(env, w)
        # Lấy trung bình của các lần chơi
        avg_episode_length = episode_length.mean()
        print(avg_episode_length)
        # Nếu số lần chơi trung bình lớn hơn bị gán lại
        if avg_episode_length > max_episode_length:
            max_episode_length = avg_episode_length
            w_best = w
    print('w_best:' , w_best)
    print('max_length:', max_episode_length)