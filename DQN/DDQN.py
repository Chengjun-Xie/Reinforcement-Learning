from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import random
from plot_utility import moving_average
from ReplayMemory import ReplayMemory
from itertools import count
import gym


class network(nn.Module):
    def __init__(self, input_size, output_size):
        super(network, self).__init__()
        self.fc1 = nn.Linear(input_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DDQN:
    def __init__(self, env, num_episode, done_reward, reward_scale=1,
                 lr=1e-3, target_update=10,
                 memory_size=10000, batch_size=512,
                 discount_factor=1, rate_decay=5e-3):
        self.env = env
        self.num_episode = num_episode
        self.done_reward = done_reward
        self.reward_scale = reward_scale

        self.lr = lr
        self.target_update = target_update
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.rate_decay = rate_decay

        self.max_exploration = 0.99
        self.min_exploration = 1e-3
        self.nA = env.action_space.n
        self.nS = env.observation_space.shape[0]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory_buffer = ReplayMemory(memory_size, batch_size)
        self.plot = moving_average(0.98)

        self.Qnetwork_1 = network(self.nS, self.nA).to(self.device)
        self.Qnetwork_2 = network(self.nS, self.nA).to(self.device)
        self.loss_func = nn.MSELoss()
        self.optimizer_1 = optim.Adam(params=self.Qnetwork_1.parameters(), lr=lr)
        self.optimizer_2 = optim.Adam(params=self.Qnetwork_2.parameters(), lr=lr)
        self.Qnetwork_2.load_state_dict(self.Qnetwork_1.state_dict())

    def epsilon_greedy(self, episode, current_state):
        exploration = self.min_exploration + (self.max_exploration - self.min_exploration) * \
                      np.exp(-self.rate_decay * episode)

        # epsilon_greedy
        if random.uniform(0, 1) < exploration:
            action = random.randrange(self.nA)
            return torch.tensor([[action]]).to(self.device)
        else:
            with torch.no_grad():
                return (self.Qnetwork_1(current_state) + self.Qnetwork_2(current_state))\
                        .max(1)[1].reshape(1, 1).to(self.device)

    def execute_selected_action(self, state, action):
        next_state, reward, done, _ = self.env.step(action.item())
        if done:
            reward = self.done_reward

        reward = torch.tensor([reward], device=self.device, dtype=torch.float)
        next_state = torch.tensor([next_state], device=self.device, dtype=torch.float)
        self.memory_buffer.push_new_experience(state, action, reward, next_state)
        return next_state, done

    def get_loss(self, train_q1):
        state_batch, action_batch, reward_batch, next_state_batch = self.memory_buffer.get_batch()
        if train_q1:
            state_action_value = self.Qnetwork_1(state_batch).gather(1, action_batch)
            next_state_value = self.Qnetwork_2(next_state_batch).max(1)[0].detach()
            expected_state_action_values = reward_batch + self.discount_factor * next_state_value
            loss = self.loss_func(state_action_value, expected_state_action_values.unsqueeze(1))
        else:
            state_action_value = self.Qnetwork_2(state_batch).gather(1, action_batch)
            next_state_value = self.Qnetwork_1(next_state_batch).max(1)[0].detach()
            expected_state_action_values = reward_batch + self.discount_factor * next_state_value
            loss = self.loss_func(state_action_value, expected_state_action_values.unsqueeze(1))
        return loss

    def update_model(self, loss, train_q1):
        if train_q1:
            self.optimizer_1.zero_grad()
            loss.backward()
            self.optimizer_1.step()
        else:
            self.optimizer_2.zero_grad()
            loss.backward()
            self.optimizer_2.step()

    def save_model(self):
        torch.save({
            'episode': self.num_episode,
            'Q_network_1_state_dict': self.Qnetwork_1.state_dict(),
            'optimizer_1_state_dict': self.optimizer_1.state_dict(),
            'Q_network_2_state_dict': self.Qnetwork_2.state_dict(),
            'optimizer_2_state_dict': self.optimizer_2.state_dict(),
        }, "DDQN_network.tar")

    def train(self):
        self.Qnetwork_1.train()
        self.Qnetwork_2.train()
        for episode in range(self.num_episode):
            # Initialize the starting state. ==
            state = self.env.reset()
            state = torch.tensor([state], device=self.device, dtype=torch.float)

            for duration in count():
                action = self.epsilon_greedy(episode, state)
                state, done = self.execute_selected_action(state, action)

                if self.memory_buffer.enough_samples():
                    if random.uniform(0, 1) < 0.5:
                        loss = self.get_loss(True)
                        self.update_model(loss, True)
                    else:
                        loss = self.get_loss(False)
                        self.update_model(loss, False)

                if done or duration > 2500:
                    self.plot.plot_moving_average(duration)
                    break

        print('Complete')
        self.save_model()
        self.plot.save_plot()
        self.env.close()

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.Qnetwork_1.load_state_dict(checkpoint['Q_network_1_state_dict'])
        self.Qnetwork_2.load_state_dict(checkpoint['Q_network_2_state_dict'])
        self.optimizer_1.load_state_dict(checkpoint['optimizer_1_state_dict'])
        self.optimizer_2.load_state_dict(checkpoint['optimizer_2_state_dict'])
        self.Qnetwork_1.eval()
        self.Qnetwork_2.eval()
        print("loaded model: ", path)


def main():
    env = gym.make('CartPole-v0').unwrapped
    estimator = DDQN(env, 1000, -100)
    estimator.train()


if __name__ == '__main__':
    main()





