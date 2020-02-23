import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from contexts import core
from core.Policy_Value import CoinToss, EpsilonGreedyAgent

if __name__ == '__main__':

    def main():
        env = CoinToss([0.1, 0.5, 0.1, 0.9, 0.1])
        epsilons = [0.0, 0.1, 0.2, 0.5, 0.8]
        game_steps = list(range(10, 310, 10))
        result = {}
        for e in epsilons:
            agent = EpsilonGreedyAgent(epsilon=e)
            means = []
            for s in game_steps:
                env.max_episode_steps = s
                rewards = agent.play(env)
                means.append(np.mean(rewards))
            result[f'epsilon={e}'] = means
        result['coin toss count'] = game_steps
        result = pd.DataFrame(result)
        result.set_index('coin toss count', drop=True, inplace=True)
        result.plot.line(figsize=(10, 5))
        plt.show()

    main()