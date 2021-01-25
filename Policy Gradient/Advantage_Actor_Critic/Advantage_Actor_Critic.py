from collections import namedtuple
import torch
from Agent import Agent
from plot_utility import moving_average
from itertools import count
import gym


class A2C:
    def __init__(self, env, num_episode,
                 lr=1e-2, discount_factor=0.99):
        self.env = env
        self.num_episode = num_episode
        self.discount_factor = discount_factor
        self.nA = env.action_space.shape[0]
        self.nS = env.observation_space.shape[0]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'gradient'))
        self.plot = moving_average(0.98, critic=True)
        self.agent = Agent(self.nA, self.nS, 8, lr, self.device, continuous=env.continuous)

    def execute_selected_action(self, action):
        next_state, reward, done, _ = self.env.step(action.numpy().reshape(2))
        next_state = torch.tensor([next_state], device=self.device, dtype=torch.float)
        reward = torch.tensor([reward], device=self.device, dtype=torch.float)
        return next_state, reward, done

    def compute_loss(self, gradient_batch, advantage_batch):
        loss = (gradient_batch.T @ advantage_batch).mean().to(self.device)
        return loss

    def train(self, render=False):
        self.agent.set_train()
        for episode in range(self.num_episode):
            # initial environment
            current_state = self.env.reset()
            current_state = torch.tensor([current_state], device=self.device, dtype=torch.float)

            total_reward = 0
            total_actor_loss = 0
            total_critic_loss = 0

            if render:
                env.render()

            for duration in count():
                action, gradient = self.agent.choose_action(current_state)
                next_state, reward, done = self.execute_selected_action(action)

                if render:
                    env.render()

                expected_future_reward = self.agent.compute_state_value(next_state)
                predicted_state_value = self.agent.compute_state_value(current_state)
                td_0 = reward + self.discount_factor * (1 - done) * expected_future_reward
                td_error = td_0 - predicted_state_value

                # update network every step
                # TD(0)
                critic_loss = self.agent.update_critic(td_0, predicted_state_value)
                actor_loss = self.compute_loss(gradient, td_error)
                self.agent.update_model(actor_loss)

                total_reward += reward.item()
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()

                if done:
                    average_actor_loss = total_actor_loss / duration
                    average_critic_loss = total_critic_loss / duration
                    self.plot.plot_moving_average(total_reward, average_actor_loss,  average_critic_loss)
                    break
                current_state = next_state

        print('Complete')
        self.plot.save_plot()
        self.env.close()


if __name__ == "__main__":
    env = gym.make("LunarLanderContinuous-v2")
    brain = A2C(env, 1000)
    brain.train()
