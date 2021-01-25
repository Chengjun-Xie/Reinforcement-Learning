import torch.multiprocessing as mp
import torch
from collections import namedtuple
from itertools import count
from torch import optim
from torch import nn
from torch.distributions import Categorical, Normal
from Network import Actor, Critic


class Agent(mp.Process):
    def __init__(self, env, agent_id,
                 global_actor_net, global_critic_net,
                 global_actor_optim, global_critic_optim,
                 num_episode, update_iter,
                 global_episode, global_reward_queue, global_actor_loss_queue, global_critic_loss_queue,
                 hidden_size=8, discount_factor=0.99, render=False):
        super(Agent, self).__init__()
        # set global network
        self.env = env
        self.agent_id = agent_id
        self.global_actor_net = global_actor_net
        self.global_critic_net = global_critic_net
        self.global_actor_optim = global_actor_optim
        self.global_critic_optim = global_critic_optim

        # set hyper parameter
        self.num_episode = num_episode
        self.update_iter = update_iter
        self.discount_factor = discount_factor
        self.nA = env.action_space.shape[0]
        self.nS = env.observation_space.shape[0]
        self.render = render

        # set local network
        self.policy_net = Actor(self.nS, self.nA, hidden_size, env.continuous).to(self.device)
        self.critic_net = Critic(self.nS, hidden_size).to(self.device)
        self.loss = nn.MSELoss()

        # set environment
        self.continuous = self.env.continuous
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'gradient'))
        self.buffer = []

        # set global plot value
        self.global_episode = global_episode
        self.global_reward_queue = global_reward_queue
        self.global_actor_loss_queue = global_actor_loss_queue
        self.global_critic_loss_queue = global_critic_loss_queue

    def choose_action(self, current_state):
        if not self.continuous:
            probs = self.policy_net(current_state)
            m = Categorical(probs)
            action = m.sample().to(self.device)
            gradient = -m.log_prob(action).to(self.device)
        else:
            mu = self.policy_net(current_state)
            m = Normal(mu, 0.05)
            action = m.sample().to(self.device)
            gradient = -m.log_prob(action).to(self.device)
        return action, gradient

    def get_batch(self):
        batch = self.Experience(*zip(*self.buffer))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        gradient_batch = torch.cat(batch.gradient)
        return state_batch, action_batch, reward_batch, next_state_batch, gradient_batch

    def compute_discounted_sum_of_return(self, reward_batch):
        current_return = 0
        discounted_sum_of_return = torch.zeros_like(reward_batch)
        for t in reversed(range(0, reward_batch.shape[0])):
            current_return = reward_batch[t] + self.discount_factor * current_return
            discounted_sum_of_return[t] = current_return
        return discounted_sum_of_return.unsqueeze(1)

    def push_new_experience(self, state, action, reward, next_state, gradient):
        new_experience = self.Experience(state, action, reward, next_state, gradient)
        self.buffer.append(new_experience)

    def execute_selected_action(self, state, action, gradient):
        next_state, reward, done, _ = self.env.step(action.numpy().reshape(2))
        next_state = torch.tensor([next_state], device=self.device, dtype=torch.float)
        reward = torch.tensor([reward], device=self.device, dtype=torch.float)
        self.push_new_experience(state, action, reward, next_state, gradient)
        return next_state, reward.item(), done

    def compute_target_value(self, next_state, reward_batch):
        next_state_value = self.critic_net(next_state)
        target_state_value = torch.zeros_like(reward_batch)
        for t in reversed(range(0, reward_batch.shape[0])):
            next_state_value = reward_batch[t] + self.discount_factor * next_state_value
            target_state_value[t] = next_state_value
        return target_state_value

    def compute_loss(self, gradient_batch, advantage_batch):
        loss = (gradient_batch.T @ advantage_batch).mean().to(self.device)
        return loss

    def update_model(self, critic_loss, actor_loss):
        # calculate local gradients and push local parameters to global
        self.global_critic_optim.zero_grad()
        self.global_actor_optim.zero_grad()
        critic_loss.backward(retain_graph=True)
        actor_loss.backward(retain_graph=True)

        for local_params, global_params in zip(self.critic_net.parameters(), self.global_critic_net.parameters()):
            global_params._grad = local_params.grad
        self.global_critic_optim.step()

        for local_params, global_params in zip(self.policy_net.parameters(), self.global_actor_net.parameters()):
            global_params._grad = local_params.grad
        self.global_actor_optim.step()

        # pull global parameters
        self.critic_net.load_state_dict(self.global_critic_net.state_dict())
        self.policy_net.load_state_dict(self.global_actor_net.state_dict())

    def plot_result(self, total_rewards, actor_loss, critic_loss):
        with self.global_episode.get_lock():
            self.global_episode += 1
        self.global_reward_queue.put(total_rewards)
        self.global_actor_loss_queue.put(actor_loss)
        self.global_critic_loss_queue.put(critic_loss)
        print("Agent ID: {0} | Episode: {1}, \t Total Reward {2}"
              .format(self.agent_id, self.global_episode.value, total_rewards))
        print("              | Critic Loss: {0}, \t Actor Loss {1}"
              .format(actor_loss, critic_loss))

    def train(self):
        self.policy_net.train()
        self.critic_net.train()

        while self.global_episode < self.num_episode:
            # initial environment
            current_state = self.env.reset()
            current_state = torch.tensor([current_state], device=self.device, dtype=torch.float)

            self.buffer = []
            total_reward = 0
            total_actor_loss = 0
            total_critic_loss = 0

            if self.render:
                self.env.render()

            for step in count():
                action, gradient = self.choose_action(current_state)
                next_state, current_reward, done = self.execute_selected_action(current_state, action, gradient)
                total_reward += current_reward

                if self.render:
                    self.env.render()

                # update network
                if (step + 1) % self.update_iter == 0 or done:
                    state_batch, action_batch, reward_batch, next_state_batch, gradient_batch = self.get_batch()
                    target_state_value = self.compute_target_value(next_state, reward_batch)
                    predicted_state_value = self.critic_net(state_batch)
                    td_error = target_state_value - predicted_state_value

                    critic_loss = self.loss(predicted_state_value, target_state_value)
                    actor_loss = self.compute_loss(gradient, td_error)
                    self.update_model(critic_loss, actor_loss)

                    total_actor_loss += actor_loss
                    total_critic_loss += critic_loss

                if done:
                    average_actor_loss = total_actor_loss / step
                    average_critic_loss = total_critic_loss / step
                    self.plot_result(total_reward, average_actor_loss, average_critic_loss)
                    break
                current_state = next_state

        self.global_reward_queue.put(None)
        self.global_actor_loss_queue.put(None)
        self.global_critic_loss_queue.put(None)