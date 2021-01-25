from collections import namedtuple
import torch
from Agent import Agent
from plot_utility import moving_average
from itertools import count
import gym


class vanilla_policy_gradient:
    def __init__(self, env, num_episode,
                 lr=5e-2, discount_factor=0.99):
        self.env = env
        self.num_episode = num_episode
        self.discount_factor = discount_factor
        self.nA = env.action_space.shape[0]
        self.nS = env.observation_space.shape[0]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'gradient'))
        self.trajectory = []
        self.plot = moving_average(0.98, critic=True)

        self.agent = Agent(self.nA, self.nS, 8, lr, self.device, continuous=env.continuous)

    def compute_discounted_sum_of_return(self, reward_batch):
        current_return = 0
        discounted_sum_of_return = torch.zeros_like(reward_batch)
        for t in reversed(range(0, reward_batch.shape[0])):
            current_return = reward_batch[t] + self.discount_factor * current_return
            discounted_sum_of_return[t] = current_return
        return discounted_sum_of_return.unsqueeze(1)

    def push_new_experience(self, state, action, reward, next_state, gradient):
        new_experience = self.Experience(state, action, reward, next_state, gradient)
        self.trajectory.append(new_experience)

    def execute_selected_action(self, state, action, gradient):
        next_state, reward, done, _ = self.env.step(action.numpy().reshape(2))
        reward = torch.tensor([reward], device=self.device, dtype=torch.float)
        next_state = torch.tensor([next_state], device=self.device, dtype=torch.float)
        self.push_new_experience(state, action, reward, next_state, gradient)
        return next_state, done

    def get_batch(self):
        batch = self.Experience(*zip(*self.trajectory))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        gradient_batch = torch.cat(batch.gradient)
        return state_batch, action_batch, reward_batch, next_state_batch, gradient_batch

    def compute_loss(self, gradient_batch, advantage_batch):
        loss = (gradient_batch.T @ advantage_batch).mean().to(self.device)
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
                    state_batch, action_batch, reward_batch, next_state_batch, gradient_batch = self.get_batch()
                    discounted_sum_of_return = self.compute_discounted_sum_of_return(reward_batch)
                    baseline_batch = self.agent.compute_state_value(state_batch)
                    advantage_batch = discounted_sum_of_return - baseline_batch

                    # update network
                    critic_loss = self.agent.update_critic(discounted_sum_of_return, baseline_batch)
                    actor_loss = self.compute_loss(gradient_batch, advantage_batch)
                    self.agent.update_model(actor_loss)
                    self.plot.plot_moving_average(reward_batch.sum().item(), actor_loss.item(), critic_loss.item())
                    break
                current_state = next_state

        print('Complete')
        self.plot.save_plot()
        self.env.close()


if __name__ == "__main__":
    env = gym.make("LunarLanderContinuous-v2")
    brain = vanilla_policy_gradient(env, 1000)
    brain.train()
