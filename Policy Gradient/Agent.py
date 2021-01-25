from torch import optim
from torch import nn
from torch.distributions import Categorical, Normal
from Network import Actor, Critic


class Agent:
    def __init__(self, num_action, num_state,
                 hidden_size, lr, device, continuous=False):
        # environment setting
        self.num_action = num_action
        self.continuous = continuous
        self.device = device

        # network setting
        self.policy_net = Actor(num_state, num_action, hidden_size, continuous).to(self.device)
        self.critic_net = Critic(num_state, hidden_size).to(self.device)
        self.policy_net_optimizer = optim.Adam(params=self.policy_net.parameters(), lr=lr)
        self.critic_net_optimizer = optim.Adam(params=self.critic_net.parameters(), lr=lr)
        self.critic_net_loss = nn.MSELoss()

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

    def compute_state_value(self, state_batch):
        return self.critic_net(state_batch)

    def update_critic(self, expected, predicted):
        loss = self.critic_net_loss(predicted, expected)
        self.critic_net_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.critic_net_optimizer.step()
        return loss

    def update_model(self, loss):
        self.policy_net_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.policy_net_optimizer.step()

    def set_train(self):
        self.critic_net.train()
        self.policy_net.train()

    def set_eval(self):
        self.critic_net.eval()
        self.policy_net.eval()

