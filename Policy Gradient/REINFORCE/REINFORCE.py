from collections import namedtuple
import torch
from Agent import Agent
from plot_utility import moving_average
from itertools import count
import gym


class REINFORCE:
    def __init__(self, env, num_episode,
                 lr=1e-2, discount_factor=0.99):
        self.env = env
        self.num_episode = num_episode
        self.discount_factor = discount_factor
        self.nA = env.action_space.shape[0]
        self.nS = env.observation_space.shape[0]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'gradient'))
        self.trajectory = []
        self.plot = moving_average(0.98)

        self.agent = Agent(self.nA, self.nS, 8, lr, self.device, continuous=env.continuous)

    def compute_discounted_sum_of_return(self, reward_batch):
        current_return = 0
        discounted_sum_of_return = torch.zeros_like(reward_batch)
        for t in reversed(range(0, reward_batch.shape[0])):
            current_return = reward_batch[t] + self.discount_factor * current_return
            discounted_sum_of_return[t] = current_return
        return discounted_sum_of_return

    def push_new_experience(self, state, action, reward, next_state, gradient):
        new_experience = self.Experience(state, action, reward, next_state, gradient)
        self.trajectory.append(new_experience)

    def execute_selected_action(self, state, action, gradient):
        next_state, reward, done, _ = self.env.step(action.numpy().reshape(2))
        reward = torch.tensor([reward], device=self.device, dtype=torch.float)
        next_state = torch.tensor([next_state], device=self.device, dtype=torch.float)
        self.push_new_experience(state, action, reward, next_state, gradient)
        return next_state, done

    def compute_loss(self):
        batch = self.Experience(*zip(*self.trajectory))
        reward_batch = torch.cat(batch.reward)
        gradient_batch = torch.cat(batch.gradient)
        discounted_return_batch = self.compute_discounted_sum_of_return(reward_batch).unsqueeze(1)
        loss = (gradient_batch.T @ discounted_return_batch).mean().to(self.device)

        self.plot.plot_moving_average(reward_batch.sum().item(), loss.item())
        return loss

    def train(self, render=False):
        self.agent.set_train()
        for episode in range(self.num_episode):
            # initial environment
            current_state = self.env.reset()
            self.trajectory = []
            current_state = torch.tensor([current_state], device=self.device, dtype=torch.float)
            if render:
                env.render()

            # sample trajectory of current episode
            while True:
                action, gradient = self.agent.choose_action(current_state)
                next_state, done = self.execute_selected_action(current_state, action, gradient)
                if render:
                    env.render()

                if done:
                    # update network
                    loss = self.compute_loss()
                    self.agent.update_model(loss)
                    break
                current_state = next_state

        print('Complete')
        self.plot.save_plot()
        self.env.close()


if __name__ == "__main__":
    env = gym.make("LunarLanderContinuous-v2")
    brain = REINFORCE(env, 1000)
    brain.train()
