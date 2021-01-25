import numpy as np
import gym
import random


class monte_carlo:
    def __init__(self, env, num_episode=10000, learning_rate=0.1, discount_factor=1, exploration_rate_decay=1e-3):
        self.env = env
        self.num_episode = num_episode
        self.learning_rate = learning_rate
        self.discount = discount_factor
        self.rate_decay = exploration_rate_decay

        self.q_table = np.zeros((env.nS, env.nA))
        self.n_table = np.zeros((env.nS, env.nA))
        self.policy = np.zeros(env.nS)

        self.max_step = 100
        self.max_exploration = 1
        self.min_exploration = 0

    def _select_action(self, episode, state):
        # epsilon greedy
        exploration = self.min_exploration + (self.max_exploration - self.min_exploration) * \
                      np.exp(- self.rate_decay * episode)
        exploration_threshold = random.uniform(0, 1)
        if exploration_threshold > exploration:
            action = np.argmax(self.q_table[state])
        else:
            action = self.env.action_space.sample()
        return action

    def _sample_episode(self, episode):
        sample = []
        state = self.env.reset()

        for steps in range(self.max_step):
            action = self._select_action(episode, state)
            next_state, reward, done, _ = self.env.step(action)
            sample.append((state, action, reward))
            state = next_state
            if done:
                return sample

    def training(self, first_visit=True, initialize=True):
        if initialize:
            self.q_table = np.zeros((self.env.nS, self.env.nA))
            self.n_table = np.zeros((self.env.nS, self.env.nA))
            self.policy = np.zeros(self.env.nS)

        reward_sum = []
        for episode in range(self.num_episode):
            # reset total reward every 1000 episodes
            if episode % 1000 == 0 and episode != 0:
                success_rate = sum(reward_sum) / len(reward_sum)
                print("success rate of at {0} - {1} episodes: {2}".format(episode - 1000, episode, success_rate))
                total_reward = []

            sample = self._sample_episode(episode)
            reward_sum.append(sample[-1][2])
            visited = np.zeros((self.env.nS, self.env.nA), dtype=bool)

            tmp_total_reward = 0
            # compute total reward at time t0
            for t in range(len(sample)):
                tmp_total_reward += (self.discount ** t) * sample[t][2]
            for t in range(len(sample)):
                state, action, current_reward = sample[t]
                if first_visit:
                    """
                    first-visit monte carlo is an unbiased estimator
                    but it needs more episodes to train 
                    """
                    if not visited[state, action]:
                        self.n_table[state, action] += 1
                        self.q_table[state, action] += (1 / self.n_table[state, action]) * \
                                                       (tmp_total_reward - self.q_table[state, action])
                        # self.q_table[state, action] += self.learning_rate * \
                        #                                (tmp_total_reward - self.q_table[state, action])
                    tmp_total_reward -= current_reward
                    tmp_total_reward /= self.discount
                else:
                    """
                    every-visit monte carlo is a biased estimator
                    but it can train faster 
                    """
                    self.n_table[state, action] += 1
                    self.q_table[state, action] += (1 / self.n_table[state, action]) * \
                                                   (tmp_total_reward - self.q_table[state, action])
                    # self.q_table[state, action] += self.learning_rate * \
                    #                                (tmp_total_reward - self.q_table[state, action])
                    tmp_total_reward -= current_reward
                    tmp_total_reward /= self.discount
        success_rate = sum(reward_sum) / len(reward_sum)
        print("End training. success rate of last 1000 episodes:", success_rate)
        self.policy = np.argmax(self.q_table, axis=1)


def play():
    env_name = 'FrozenLake-v0'
    env = gym.make(env_name)
    agent = monte_carlo(env)
    agent.training()


if __name__ == "__main__":
    play()
