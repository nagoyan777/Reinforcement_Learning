import random
import numpy as np


class EpsilonGreedyAgent():
    def __init__(self, epsilon):

        self.epsilon = epsilon
        self.V = []

    def policy(self):
        actions = range(len(self.V))
        if random.random() < self.epsilon:
            return random.choice(actions)
        else:
            return np.argmax(self.V)

    def play(self, env):
        # Initialize estimateion.
        N = [0] * len(env)
        self.V = [0] * len(env)

        env.reset()
        done = False
        rewards = []
        while not done:
            action = self.policy()
            reward, done = env.step(action)
            rewards.append(reward)

            n = N[action]
            average = self.V[action]
            new_average = (average * n + reward) / (n + 1)
            N[action] += 1
            self.V[action] = new_average
        return rewards
