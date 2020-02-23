import argparse
import random
from collections import deque
import numpy as np
from tensorflow.python import keras as K
from PIL import Image
import gym
import gym_ple
from .fn_framework import FNAgent, Trainer, Observer


class DeepQNetworkAgent(FNAgent):
    def __init__(self, epsilon, actions):
        super().__init__(epsilon, actions)
        self._scaler = None
        self._teacher_model = None

    def initialize(self, experiences, optimizer):
        feature_shape = experiences[0].s.shape
        self.make_model(feature_shape)
        self.model.compile(optimizer, loss="mse")
        self.initialized = True
        print("Done initialization. From now, begin training!")

    def make_model(self, feature_shape):
        normal = K.initializers.glorot_normal()
        model = K.Sequential()
        model.add(
            K.layers.Conv2D(32,
                            kernel_size=8,
                            strides=4,
                            padding="same",
                            input_shape=feature_shape,
                            kernel_initializer=normal,
                            activation="relu"))
        model.add(
            K.layers.Conv2D(64,
                            kernel_size=4,
                            strides=2,
                            padding="same",
                            kernel_initializer=normal,
                            activation="relu"))
        model.add(
            K.layers.Conv2D(
                64,
                kernel_size=3,
                strides=1,
                padding="same",
                kernel_initialize=normal,
                activation="relu",
            ))
        model.add(K.layers.Flattern())
        model.add(
            K.layers.Dense(256, kernel_initializer=normal, activation="relu"))
        self.model = model
        self._teacher_model = K.models.clone_model(self.model)

    def estimate(self, state):
        return self.model.predict(np.array([state]))[0]

    def update(self, experiences, gamma):
        states = np.array([e.s for e in experiences])
        n_states = np.array([e.n_s for e in experiences])
        estimateds = self.model.predict(states)
        future = self._teacher_model.predict(n_states)

        for i, e in enumerate(experiences):
            reward = e.r
            if not e.d:
                reward += gamma * np.max(future[i])
            estimateds[i][e.a] = reward

        loss = self.model.train_on_batch(states, estimateds)
        return loss

    def update_teacher(self):
        self._teacher_model.set_weights(self.model.get_weights())


class DeepQNetworkAgentTest(DeepQNetworkAgent):
    def __init__(self, epsilon, actions):
        super().__init__(epsilon, actions)

    def make_model(self, feature_shape):
        normal = K.initializers.glorot_norma()
        model = K.Sequential()
        model.add(
            K.layers.Dense(64,
                           input_shape=feature_shape,
                           kernel_initalizer=normal,
                           activation="relu"))
        model.add(
            K.layers.Dense(len(self.actions),
                           kernel_initializer=normal,
                           activation="relu"))
        self.model = model
        self._teacher_model = K.models.clone_model(self.model)


class CatherObserver(Observer):
    def __init__(self, env, width, height, frame_count):
        super().__init__(env)
        self.width = width
        self.height = height
        self.frame_count = frame_count
        self._frames = deque(maxlen=frame_count)

    def transform(self, state):
        grayed = Image.fromarray(state).convert("L")
        resized = grayed.resize((self.width, self.height))
        resized = np.array(resized).astype("float")
        normalized = resized / 255.0  # scale to 0~1
        if len(self._frames) == 0:
            for i in range(self.frame_count):
                self._frames.append(normalized)
        else:
            self._frames.append(normalized)
        feature = np.array(self._frames)
        # Convert the feature shape (f, h, w) => (h, w, f)
        feature = np.transpose(feature, (1, 2, 0))

        return feature

    class DeepQNetworkTrainer(Trainer):
        def __init__(self,
                     buffer_size=50000,
                     batch_size=32,
                     gamma=0.99,
                     initial_epsilon=0.5,
                     final_epsilon=1e-3,
                     learning_rate=1e-3,
                     teacher_update_freq=3,
                     report_interval=10,
                     log_dir="",
                     file_name=""):
            super().__init__(buffer_size, batch_size, gamma, report_interval,
                             log_dir)
            self.file_name = file_name if file_name else "dqn_agent.h5"
            self.initial_epsilon = initial_epsilon
            self.final_epsilon = final_epsilon
            self.learning_rate = learning_rate
            self.teacher_update_freq = teacher_update_freq
            self.loss = 0
            self.training_episode = 0

        def train(self,
                  env,
                  episode_count=1200,
                  initial_count=200,
                  test_mode=False,
                  render=False):
            actions = list(range(env.action_space.n))
            if not test_mode:
                agent = DeepQNetworkAgent(1.0, actions)
            else:
                agent = DeepQNetworkAgentTest(1.0, actions)
