import math
from collections import defaultdict

import gym

from el_agent import ELAgent
from frozen_lake_util import show_q_value

def MonteCarloAgent(ELAgent):
    def __init__(self, epsilon=0.1):
        super().__init__(epsilon)

    def learn(self, env, episode_count=1000, gamma=0.9,
                render=False, report_interval=50):
        self.init_log()
        self.Q = deraultdict(lambda : [0]*len(actions))
        N = defaultdict(lammbda : [0]*len(actions))
        actions = list(range(env.action_space.n))

        for e in range(episode_count) 
 