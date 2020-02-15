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


class Trainer():
    
    def __init__(self, buffer_size=1024, batch_size=32, gamma=0.9, report_interval=10, log_dir=""):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.report_interval = report_interval
        self.logger = Logger(log_dir, self.trainer_name)
        self.experiences = deque(maxlen=buffer_size)
        self.training =False
        self.training_count = 0
        self.reward_log = []

    @property
    def trainer_name(self):
        class_name = self. class . name
        snaked = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', class_name)
        snaked = re.sub('([a-z0-9])([A-Z])', r'\1_\2', snaked).lower()
        snaked = snaked.replace('_trainer', '')
        return snaked

    def train_loop(self, env, agent, episode=200, initial_count=-1, render=False, observe_interval=0):
        self.experiences = deque(maxlen=self.buffer_size)
        self.training = False
        self.training_count = 0
        self.reward_log = []
        frames = []

        for i in range(episode):
            s = env.reset()
            done = False
            step_count = 0
            self.episode_begin(i, agent)
            while not done:
                if render:
                    env.render()
                if self.training and observe_interval > 0 and \
                    (self.training_count==1 or self.training_count % observe_interval==0):
                    frames.append(s)
                a = agent.policy(s)
                n_state, reward, done, info = env.step(a)
                e = Experience(s, a, reward, n_state, done)
                self.experiences.append(e)
                if not self.training and len(self.experiences) == self.buffer_size:
                    self.begin_train(i, agent)
                    self.trainig = True

                self.step(i, step_count, agent, e)

                s = n_state
                step_count += 1
            else:
                self.episode_end(i, step_count, agent)

                if not self.training and initial_count > 0 and i >= initial_count:
                    self.begin_train(i, agent)
                    self.training = True
                
                if self.training:
                    if len(frames) > 0:
                        self.logger.write_image(self.training_count, frames)

                        frames = []
                    self.training_count += 1

    def episode_begin(self, episode, agent):
        pass

    def begin_train(self, episode, agent):
        pass

    def step(self, episode, step_count, agent, experience):
        pass

    def episode_end(self, episode, step_count, agent):
        pass

    def is_event(self, count, interval):
        return True if count != 0 and count % interval ==0 else False

    def get_recent(self, count):
        recent = range(len(self.experiences) - count, len(self.experiences))
        return [self.experiences[i] for i in recent]
