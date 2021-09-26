# https://deeplearningcourses.com/c/deep-reinforcement-learning-in-python
# https://www.udemy.com/deep-reinforcement-learning-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future
#
# This takes 4min 30s to run in Python 2.7
# But only 1min 30s to run in Python 3.5!
#
# Note: gym changed from version 0.7.3 to 0.8.0
# MountainCar episode length is capped at 200 in later versions.
# This means your agent can't learn as much in the earlier episodes
# since they are no longer as long.

import gym
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gym import wrappers
from datetime import datetime
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor


# SGDRegressor defaults:
# loss='squared_loss', penalty='l2', alpha=0.0001,
# l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=True,
# verbose=0, epsilon=0.1, random_state=None, learning_rate='invscaling',
# eta0=0.01, power_t=0.25, warm_start=False, average=False

# Inspired by https://github.com/dennybritz/reinforcement-learning
class FeatureTransformer:
    def __init__(self, env, n_components=500):
        N = 10000
        self.feature_transformer = FeatureUnion([
            ('rbf1', RBFSampler(gamma=5, n_components=n_components)),
            ('rbf2', RBFSampler(gamma=2, n_components=n_components)),
            ('rbf3', RBFSampler(gamma=3, n_components=n_components)),
            ('rbf4', RBFSampler(gamma=0.5, n_components=n_components)),
        ])

        # Tạo ra các sample để tiến hành scale và fit với Feature Transform
        samples = np.array([env.observation_space.sample() for i in range(N)])
        # Tiến hành fit scaler
        self.scaler = StandardScaler()
        self.scaler.fit(samples)

        # Tiến hành fit Feature Tranform
        self.feature_transformer.fit(self.scaler.transform(samples))

    def transform(self, observations):
        scaled = self.scaler.transform(observations)
        return self.feature_transformer.transform(scaled)


# Holds one SGDRegressor for each action
class Model:
    def __init__(self, env, feature_transformer, learning_rate):
        self.env = env
        self.feature_transformer = feature_transformer
        # Khởi tạo các model tương ứng với các action
        self.models = []
        for i in range(env.action_space.n):
            model = SGDRegressor(learning_rate=learning_rate)
            model.fit(self.feature_transformer.transform([self.env.reset()]), [0])
            self.models.append(model)

    def predict(self, s):
        x = self.feature_transformer.transform([s])
        result = np.stack([model.predict(x) for model in self.models]).T
        return result

    def update(self, s, a, G):
        x = self.feature_transformer.transform([s])
        self.models[a].partial_fit(x, [G])

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))


# returns a list of states_and_rewards, and the total reward
def play_one(model, env, eps, gamma, render=False):
    # Reset env sau mỗi lần chơi
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0
    while not done and iters < 10000:
        # Chọn action với eps
        action = model.sample_action(observation, eps)
        # Lưu lại state trước để tiến hành cập nhật
        prev_observation = observation

        # Nhận giá trị mới sau lần thử
        observation, reward, done, info = env.step(action)

        # update the model
        next = model.predict(observation)
        # assert(next.shape == (1, env.action_space.n))
        G = reward + gamma * np.max(next[0])
        model.update(prev_observation, action, G)

        totalreward += reward
        iters += 1
        if render:
            env.render()

    return totalreward


def view_agent(model, env):
    # Reset env sau mỗi lần chơi
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0
    while not done and iters < 10000:
        # Chọn action với eps
        action = np.argmax(model.predict(observation))

        # Nhận giá trị mới sau lần thử
        observation, reward, done, info = env.step(action)

        totalreward += reward

        env.render()
    print('reward:', totalreward)


def plot_cost_to_go(env, estimator, num_tiles=20):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)
    # both X and Y will be of shape (num_tiles, num_tiles)
    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))
    # Z will also be of shape (num_tiles, num_tiles)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z,
                           rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost-To-Go == -V(s)')
    ax.set_title("Cost-To-Go Function")
    fig.colorbar(surf)
    plt.show()


def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t - 100):(t + 1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()


def main(show_plots=True):
    # Khởi tạo môi trường
    env = gym.make('MountainCar-v0')

    # Khởi tạo FeatureTransformer chuyển State sang Feature
    ft = FeatureTransformer(env)

    # Khởi tạo RL model
    model = Model(env, ft, "constant")
    gamma = 0.99

    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)

    # Số lần thử
    N = 1000
    # Khởi tạo totalrewards để plot
    totalrewards = np.empty(N)

    # Khởi tạo eps-greedy chọn action
    eps = None

    # Thử
    for n in range(N):
        # eps = 1.0/(0.1*n+1)
        # Giảm eps với lần thử cao hơn
        eps = 0.1 * (0.97 ** n)
        if n == 199:
            print("eps:", eps)
        # eps = 1.0/np.sqrt(n+1)
        # Lưu lại reward nhận được sau mỗi lần chơi
        totalreward = play_one(model, env, eps, gamma)

        # Set lại vào total reward
        totalrewards[n] = totalreward
        if (n + 1) % 100 == 0:
            print("episode:", n, "total reward:", totalreward)
    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", -totalrewards.sum())

    if show_plots:
        plt.plot(totalrewards)
        plt.title("Rewards")
        plt.show()

        plot_running_avg(totalrewards)

        # plot the optimal state-value function
        plot_cost_to_go(env, model)
    while True:
        view_agent(model, env)


if __name__ == '__main__':
    # for i in range(10):
    #   main(show_plots=False)
    main()
