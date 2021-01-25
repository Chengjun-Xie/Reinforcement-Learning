__all__ = ['EpsilponGreedy', 'Greedy', 'Uniform']
import numpy as np
import random


class EpsilponGreedy:
    def __init__(self, max_exploration, min_exploration, rate_decay):
        self.max_exploration = max_exploration
        self.min_exploration = min_exploration
        self.rate_decay = rate_decay

    def do_exploration(self, episode):
        exploration = self.min_exploration + (self.max_exploration - self.min_exploration) * \
                      np.exp(- self.rate_decay * episode)
        sample = random.uniform(0, 1)
        return sample < exploration


class Greedy:
    def do_exploration(self, episode):
        return False


class Uniform:
    def do_exploration(self, episode):
        return True