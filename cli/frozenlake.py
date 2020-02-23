import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from contexts import core
from core.EL import MonteCarloAgent, show_q_value

import gym


def train():
    agent = MonteCarloAgent(epsilon=0.1)
    env = gym.make("FrozenLakeEasy-v0")
    agent.learn(env, episode_count=1000)
    show_q_value(agent.Q)
    agent.show_reward_log()


if __name__ == "__main__":
    train()