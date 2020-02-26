import random
# import numpy as np


class CoinToss():
    """ Coin toss game.
    
    Raises:
        Exception: ation > len(self.had_probs) 
    
    Returns:
        reward (float): 
        done (logical):

    Usage:
    >>> head_probs = [0.3, 0.8, 0.5]
    >>> env = CoinToss(head_probs, max_episode_steps=30)
    >>> rewards = []
    >>> for i in range(len(head_probs)):
    >>>     rewards.append(env.step(action))
    >>> rewards
    1.0, 1.0, 1.0     
    """
    def __init__(self, head_probs, max_episode_steps=30):
        self.head_probs = head_probs
        self.max_episode_steps = max_episode_steps
        self.toss_count = 0

    def __len__(self):
        return len(self.head_probs)

    def reset(self):
        self.toss_count = 0

    def step(self, action):
        final = self.max_episode_steps - 1
        if self.toss_count > final:
            raise Exception(
                'The step count exceeded maximum. Please reset env.')
        else:
            done = True if self.toss_count == final else False

        if action >= len(self.head_probs):
            raise Exception(f'The No.{action} coin does not exest.')
        else:
            head_prob = self.head_probs[action]
            if random.random() < head_prob:
                reward = 1.0
            else:
                reward = 0.0
            self.toss_count += 1
            return reward, done
