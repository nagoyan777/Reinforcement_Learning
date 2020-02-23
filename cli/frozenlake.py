import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from contexts import core
from core.EL import MonteCarloAgent, QLearningAgent, show_q_value

import gym


def train(agent='qlerinig', epsilon=0.1, episode_count=1000):
    if agent.lower()[0] == 'm':
        agent = MonteCarloAgent(epsilon)
    elif agent.lower()[0] == 'q':
        agent = QLearningAgent(epsilon)

    env = gym.make("FrozenLakeEasy-v0")
    agent.learn(env, episode_count=episode_count)
    show_q_value(agent.Q)
    agent.show_reward_log()


if __name__ == "__main__":
    train('q')