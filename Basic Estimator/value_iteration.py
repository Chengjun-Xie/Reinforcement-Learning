import numpy as np
import gym


class value_iteration:
    def __init__(self, env, max_iteration=10000, discount_factor=0.99):
        self.env = env
        self.max_iter = max_iteration
        self.discount = discount_factor

        self.value = np.zeros(env.nS)
        self.policy = np.zeros(env.nS)
        self.eps = 1e-9

    def _value_improvement(self):
        for i in range(self.max_iter):
            delta = 0
            for state in range(self.env.nS):
                value_list = []
                for action in range(self.env.nA):
                    val = 0
                    for next_state in self.env.P[state][action]:
                        probability, new_state, reward, done = next_state
                        val += probability * (reward + self.discount * self.value[new_state])
                    value_list.append(val)
                cur_val = max(value_list)
                new_delta = abs(cur_val - self.value[state])
                delta = max(delta, new_delta)
                self.value[state] = cur_val

            if delta <= self.eps:
                print("value evaluation converted at iteration #%d, with delta=%s" % (i, delta))
                break

    def _policy_extraction(self):
        for state in range(self.env.nS):
            value_list = []
            for action in range(self.env.nA):
                val = 0
                for next_state in self.env.P[state][action]:
                    probability, new_state, reward, done = next_state
                    val += probability * (reward + self.discount * self.value[new_state])
                value_list.append(val)
            self.policy[state] = np.argmax(value_list)
        print("Generated new policy!")

    def training(self):
        self._value_improvement()
        self._policy_extraction()


def play():
    env_name = 'FrozenLake-v0'
    env = gym.make(env_name)
    agent = value_iteration(env)
    agent.training()


if __name__ == "__main__":
    play()