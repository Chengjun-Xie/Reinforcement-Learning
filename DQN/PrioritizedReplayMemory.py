import random
import numpy as np
from collections import namedtuple
import torch


class ReplayMemory:
    def __init__(self, capacity, batch_size, priority_scale=1.0):
        self.capacity = capacity
        self.batch_size = batch_size
        self.priority_scale = priority_scale

        self.memory = []
        self.priority = []
        self.push_count = 0
        self.Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state'))

    def push_new_experience(self, state, action, reward, next_state):
        new_experience = self.Experience(state, action, reward, next_state)

        # initialize new experience to the highest priority
        if len(self.memory) < self.capacity:
            self.memory.append(new_experience)
            self.priority.append(max(self.priority, default=1))
        else:
            index = self.push_count % self.capacity
            self.memory[index] = new_experience
            self.priority[index] = max(self.priority, default=1)
        self.push_count += 1

    def get_importance(self, probabilities):
        importance = 1 / len(self.memory) * 1 / probabilities
        importance_normalized = importance / max(importance)
        return importance_normalized

    def get_probabilities(self):
        scaled_priorities = np.array(self.priority) ** self.priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities

    def get_batch(self):
        # prioritized batch sampling
        sample_probs = self.get_probabilities()
        sample_indices = random.choices(range(len(self.memory)), k=self.batch_size, weights=sample_probs)

        experiences = []
        for index in sample_indices:
            experiences.append(self.memory[index])
        importance = self.get_importance(sample_probs[sample_indices])
        batch = self.Experience(*zip(*experiences))

        # by calling torch.cat() we extract all the states
        # from this batch into their own state tensor.
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        importance_batch = torch.tensor([importance], dtype=torch.float)
        return state_batch, action_batch, reward_batch, next_state_batch, importance_batch, sample_indices

    def enough_samples(self):
        return self.batch_size < self.push_count

    def set_priorities(self, indices, errors, offset=0.1):
        for i, e in zip(indices, errors):
            self.priority[i] = e + offset