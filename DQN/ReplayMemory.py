import random
from collections import namedtuple
import torch


class ReplayMemory:
    def __init__(self, capacity, batch_size):
        self.memory = []
        self.capacity = capacity
        self.push_count = 0
        self.batch_size = batch_size
        self.Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state'))

    def push_new_experience(self, state, action, reward, next_state):
        new_experience = self.Experience(state, action, reward, next_state)
        if len(self.memory) < self.capacity:
            self.memory.append(new_experience)
        else:
            index = self.push_count % self.capacity
            self.memory[index] = new_experience
        self.push_count += 1

    def get_batch(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        '''
        from:
            [Experience(state=1, action=1, reward=1, next_state=1),
             Experience(state=2, action=2, reward=2, next_state=2),
             Experience(state=3, action=3, reward=3, next_state=3)]
        to:
            Experience(state=(1, 2, 3), action=(1, 2, 3), reward=(1, 2, 3), next_state=(1, 2, 3))
        '''
        batch = self.Experience(*zip(*experiences))

        # by calling torch.cat() we extract all the states
        # from this batch into their own state tensor.
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        return state_batch, action_batch, reward_batch, next_state_batch

    def enough_samples(self):
        return self.batch_size < self.push_count