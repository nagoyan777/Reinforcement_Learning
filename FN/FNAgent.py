import os, io, re, numpy as np
from collections import namedtuple, deque
import tensorflow as tf
from PIL import Image
from tensorflow.python import keras as K

import matplotlib.pyplot as plt

Experience = namedtuple("Experience", ["s", "a", "r", "n_s", "d"])

class FNAgent():
    def __init__(self, epsilon, actions):
        self.epsilon = epsilon
        self.actions = actions
        self.model = None
        self.estimate_probs = False
        self.initialized = False
    def save(self, model_path):
        self.model.save(model_path, overwrite=True, include_optimizer=False)

    @classmethod
    def load(cls, env, model_path, epsilon=0.0001):
        actions = list(range(env.action_space.n))
        agent = cls(epsilon, actions)
        agent.model = K.models.load_model(model_path)
        agent.initialized = True
        return agent

    def initialize(self, experiences):
        raise NotImplementedError("Errro")

    def estimzte(self, s):
        raise NotImplementedError("")

    def update(self, experiences, gamma):
        raise NotImplementedError("")

    def policy(self, s):
        if np.random.random() < self.epsilon or not self.initialized:
            return np.random.randint(len(self.actions))
        else:
            estimates = self.estimate(s)
            if self.estimate_probs:
                action = np.random.choice(self.actions, size=1, p=estimates)[0]

                return action

            else:
                return np.argmax(estimates)

    def play(self, env, episode_count=5, render=True):
        for e in range(episode_count):
            s = env.reset()
            done = False
            episode_reward = 0
            while not done:
                if render: 
                    env.render()
                a = self.policy(s)
                n_state, reward, done, info = env.step(a)
                episode_reward += reward
                s = n_state

            else:
                print("Get reward {}.".format(episode_reward))

