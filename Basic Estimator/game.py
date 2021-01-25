import gym
import time
import os
from policy_iteration import policy_iteration


class FrozenLake:
    @staticmethod
    def __run_episode(env, policy):
        """
        Evaluates policy by using it to run an episode and finding its total reward.

        args:
        env: gym environment.
        policy: the policy to be used.
        gamma: discount factor.
        render: boolean to turn rendering on/off.

        returns:
        total reward: real value of the total reward recieved by agent under policy.
        """
        state = env.reset()
        step_idx = 0
        while True:
            os.system("cls")
            print("*** Game Started! ***")
            env.render()

            action = int(policy[state])
            state, current_reward, done, info = env.step(action)
            step_idx += 1
            time.sleep(0.1)
            if done:
                os.system("cls")
                env.render()
                if current_reward == 1:
                    print("****You reached the goal!****")
                    print("Total Steps:", step_idx)
                    time.sleep(3)
                else:
                    print("****You fell through a hole!****")
                    print("Total Steps:", step_idx)
                    time.sleep(3)
                break

    @staticmethod
    def play():
        env_name = 'FrozenLake-v0'
        discount_factor = 0.99
        env = gym.make(env_name)

        optimizer = policy_iteration(env, discount_factor=discount_factor)
        optimizer.training()
        time.sleep(5)
        FrozenLake.__run_episode(env, optimizer.policy)


if __name__ == "__main__":
    FrozenLake.play()