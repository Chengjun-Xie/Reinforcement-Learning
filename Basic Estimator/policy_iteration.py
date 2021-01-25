import numpy as np
import gym


class policy_iteration:
    def __init__(self, env, max_iteration=10000, discount_factor=0.99):
        self.env = env
        self.max_iter = max_iteration
        self.discount = discount_factor

        self.value = np.zeros(env.nS)
        self.policy = np.zeros(env.nS)
        self.eps = 1e-9

    def _initialization(self):
        self.policy = np.random.randint(0, 4, self.env.nS)
        print(self.policy)

    def _policy_evaluation(self):
        for i in range(self.max_iter):
            delta = 0
            for state in range(self.env.nS):
                val = 0
                action = self.policy[state]

                for next_state in self.env.P[state][action]:
                    probability, new_state, reward, done = next_state
                    val += probability * (reward + self.discount * self.value[new_state])

                new_delta = np.fabs(self.value[state] - val)
                delta = max(delta, new_delta)
                self.value[state] = val

            if delta <= self.eps:
                print("policy evaluation converted at iteration #%d, with delta=%s" % (i, delta))
                break

    def _policy_improvement(self):
        stable = True
        for state in range(self.env.nS):
            temp = np.copy(self.policy[state])
            reward_list = []
            for action in range(self.env.nA):
                val = 0
                for next_state in self.env.P[state][action]:
                    probability, new_state, reward, done = next_state
                    val += probability * (reward + self.discount * self.value[new_state])
                reward_list.append(val)
            self.policy[state] = np.argmax(reward_list)
            if temp != self.policy[state]:
                stable = False
        return stable

    def training(self):
        self._initialization()
        for n in range(self.max_iter):
            self._policy_evaluation()
            done = self._policy_improvement()
            if done:
                print("training is done at iteration #%d" % n)
                return
        print("The training process has reached the maximum iteration!")


def play():
    env_name = 'FrozenLake-v0'
    env = gym.make(env_name)
    agent = policy_iteration(env)
    agent.training()


if __name__ == "__main__":
    play()