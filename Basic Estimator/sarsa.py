import numpy as np
import gym
import random


class sarsa:
    def __init__(self, env, num_episode=10000, learning_rate=0.1, discount_factor=0.99, exploration_rate_decay=0.001):
        self.env = env
        self.num_episode = num_episode
        self.learning_rate = learning_rate
        self.discount = discount_factor
        self.rate_decay = exploration_rate_decay

        self.q_table = np.zeros((env.nS, env.nA))
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

    def training(self, initialize=True):
        if initialize:
            self.q_table = np.zeros((self.env.nS, self.env.nA))
            self.policy = np.zeros(self.env.nS)
        total_reward = []

        for episode in range(self.num_episode):
            current_state = self.env.reset()
            current_action = self._select_action(episode, current_state)

            # reset total reward every 1000 episodes
            if episode % 1000 == 0 and episode != 0:
                success_rate = sum(total_reward) / len(total_reward)
                print("success rate of at {0} - {1} episodes: {2}".format(episode - 1000, episode, success_rate))
                total_reward = []

            for steps in range(self.max_step):
                next_state, reward, done, _ = self.env.step(current_action)
                next_action = self._select_action(episode, next_state)

                new_q = (1 - self.learning_rate) * self.q_table[current_state, current_action] + self.learning_rate * (
                        reward + self.discount * self.q_table[next_state][next_action])
                self.q_table[current_state, current_action] = new_q

                current_state = next_state
                current_action = next_action
                if done:
                    total_reward.append(reward)
                    break

        success_rate = sum(total_reward) / len(total_reward)
        print("End training. success rate of last 1000 episodes:", success_rate)
        self.policy = np.argmax(self.q_table, axis=1)


def play():
    env_name = 'FrozenLake-v0'
    env = gym.make(env_name)
    agent = sarsa(env)
    agent.training()


if __name__ == "__main__":
    play()