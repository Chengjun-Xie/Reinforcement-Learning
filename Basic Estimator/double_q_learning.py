import numpy as np
import gym
import random


class double_q_learning:
    def __init__(self, env, num_episode=10000, learning_rate=0.1, discount_factor=0.99, exploration_rate_decay=0.001):
        self.env = env
        self.num_episode = num_episode
        self.learning_rate = learning_rate
        self.discount = discount_factor
        self.rate_decay = exploration_rate_decay

        self.q_table_a = np.zeros((env.nS, env.nA))
        self.q_table_b = np.zeros((env.nS, env.nA))
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
            action = np.argmax(self.q_table_a[state] + self.q_table_b[state])
        else:
            action = self.env.action_space.sample()
        return action

    def training(self, initialize=True):
        if initialize:
            self.q_table_a = np.zeros((self.env.nS, self.env.nA))
            self.q_table_b = np.zeros((self.env.nS, self.env.nA))
            self.policy = np.zeros(self.env.nS)
        total_reward = []

        for episode in range(self.num_episode):
            state = self.env.reset()

            # reset total reward every 1000 episodes
            if episode % 1000 == 0 and episode != 0:
                success_rate = sum(total_reward) / len(total_reward)
                print("success rate of at {0} - {1} episodes: {2}".format(episode - 1000, episode, success_rate))
                total_reward = []

            for steps in range(self.max_step):
                action = self._select_action(episode, state)
                next_state, reward, done, _ = self.env.step(action)

                # update q table
                if random.uniform(0, 1) > 0.5:
                    # update q table a bootstrapping from q table b
                    new_q = (1 - self.learning_rate) * self.q_table_a[state, action] + self.learning_rate * \
                            (reward + self.discount * max(self.q_table_b[next_state]))
                    self.q_table_a[state, action] = new_q
                else:
                    # update q table b bootstrapping from q table a
                    new_q = (1 - self.learning_rate) * self.q_table_b[state, action] + self.learning_rate * \
                            (reward + self.discount * max(self.q_table_a[next_state]))
                    self.q_table_b[state, action] = new_q

                state = next_state
                if done:
                    total_reward.append(reward)
                    break

        success_rate = sum(total_reward) / len(total_reward)
        print("End training. success rate of last 1000 episodes:", success_rate)
        self.policy = np.argmax(self.q_table_a + self.q_table_b, axis=1)


def play():
    env_name = 'FrozenLake-v0'
    env = gym.make(env_name)
    agent = double_q_learning(env)
    agent.training()


if __name__ == "__main__":
    play()