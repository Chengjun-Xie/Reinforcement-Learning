import torch
import random


class Agent:
    def __init__(self, num_action, strategy, device):
        self.num_action = num_action
        self.strategy = strategy
        self.device = device

    def select_action(self, episode, current_state, policy_net):
        if self.strategy.do_exploration(episode):
            action = random.randrange(self.num_action)
            return torch.tensor([[action]]).to(self.device)
        else:
            with torch.no_grad():
                return policy_net(current_state).argmax(dim=1).reshape(1, 1).to(self.device)
