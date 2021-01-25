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


class DQN:
    def __init__(self, env, num_episode, done_reward, reward_scale=1,
                 lr=1e-3, target_update=10,
                 memory_size=50000, batch_size=1024,
                 discount_factor=1, rate_decay=5e-2):
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

        self.policy_net = network(self.nS, self.nA).to(self.device)
        self.target_net = network(self.nS, self.nA).to(self.device)
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=lr)

        # We then set the weights and biases in the target_net to be the same as those in the policy_net
        # using PyTorch’s state_dict() and load_state_dict() functions.
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # We also put the target_net into eval mode
        # which tells PyTorch that this network is not in training mode.
        self.target_net.eval()

    def epsilon_greedy(self, episode, current_state):
        exploration = self.min_exploration + (self.max_exploration - self.min_exploration) * \
                      np.exp(-self.rate_decay * episode)

        # epsilon_greedy
        if random.uniform(0, 1) < exploration:
            action = random.randrange(self.nA)
            return torch.tensor([[action]]).to(self.device)
        else:
            with torch.no_grad():
                return self.policy_net(current_state).max(1)[1].reshape(1, 1).to(self.device)

    def execute_selected_action(self, state, action):
        next_state, reward, done, _ = self.env.step(action.item())
        if done:
            reward = self.done_reward

        reward = torch.tensor([reward], device=self.device, dtype=torch.float)
        next_state = torch.tensor([next_state], device=self.device, dtype=torch.float)
        self.memory_buffer.push_new_experience(state, action, reward, next_state)
        return next_state, done

    def get_loss(self):
        state_batch, action_batch, reward_batch, next_state_batch = self.memory_buffer.get_batch()
        state_action_value = self.policy_net(state_batch).gather(1, action_batch)
        next_state_value = self.target_net(next_state_batch).max(1)[0].detach()

        expected_state_action_values = reward_batch + self.discount_factor * next_state_value
        loss = self.loss_func(state_action_value, expected_state_action_values.unsqueeze(1))
        return loss

    def update_model(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self):
        torch.save({
            'episode': self.num_episode,
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, "DQN_network.tar")

    def train(self):
        self.policy_net.train()
        for episode in range(self.num_episode):
            # Initialize the starting state. ==
            state = self.env.reset()
            state = torch.tensor([state], device=self.device, dtype=torch.float)

            for duration in count():
                '''
                For each steps:
                    Select an action.
                        Via exploration or exploitation
                    Execute selected action in an emulator.
                    Observe reward and next state.
                    Store experience in replay memory.
                    Sample random batch from replay memory.
                    Preprocess states from batch.
                    Pass batch of preprocessed states to policy network.
                    Calculate loss between output Q-values and target Q-values.
                        loss = q*(s, a) - q(s, a)
                             = E{Rt+1 + discount_factor * max(a')(s', a')} - q(s, a)
                        Requires a pass to the target network for the next state => max(a')(s', a')
                    Gradient descent updates weights in the policy network to minimize loss.
                        After time steps, weights in the target network are updated to the weights in the policy network.
                '''

                action = self.epsilon_greedy(episode, state)
                state, done = self.execute_selected_action(state, action)

                if self.memory_buffer.enough_samples():
                    loss = self.get_loss()
                    self.update_model(loss)

                if episode % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                if done or duration > 2500:
                    self.plot.plot_moving_average(duration)
                    break
        print('Complete')
        self.save_model()
        self.plot.save_plot()
        self.env.close()

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.target_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.policy_net.eval()
        self.target_net.eval()
        print("loaded model: ", path)


def main():
    env = gym.make('CartPole-v0').unwrapped
    estimator = DQN(env, 1000, -100)
    estimator.train()


if __name__ == '__main__':
    main()






