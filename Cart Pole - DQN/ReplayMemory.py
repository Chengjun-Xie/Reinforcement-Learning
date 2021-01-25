from collections import namedtuple
import random
import torch

Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)


class ReplayMemory:
    def __init__(self, capacity, batch_size):
        self.memory = []
        self.capacity = capacity
        self.push_count = 0
        self.batch_size = batch_size

    def push(self, new_experience):
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
            [Experience(state=1, action=1, next_state=1, reward=1),
             Experience(state=2, action=2, next_state=2, reward=2),
             Experience(state=3, action=3, next_state=3, reward=3)]
        to:
            Experience(state=(1, 2, 3), action=(1, 2, 3), next_state=(1, 2, 3), reward=(1, 2, 3))
        '''
        batch = Experience(*zip(*experiences))

        # by calling torch.cat() we extract all the states
        # from this batch into their own state tensor.
        t1 = torch.cat(batch.state)
        t2 = torch.cat(batch.action)
        t3 = torch.cat(batch.next_state)
        t4 = torch.cat(batch.reward)

        return t1, t2, t3, t4

    def enough_samples(self):
        return self.batch_size < self.push_count
