from collections import namedtuple
import torch
from torch import multiprocessing as mp
from Agent import Agent
import Network
from ShareAdam import ShareAdam
from plot_utility import moving_average
from itertools import count
import gym


def main_training_thread(num_process=1):
    # set hyper parameter
    game_name = "LunarLanderContinuous-v2"
    env = gym.make(game_name)
    if num_process >= mp.cpu_count():
        num_process = mp.cpu_count()
    num_state = env.observation_space.shape[0]
    num_action = env.action_space.shape[0]
    hidden_size = 8
    num_episode = 1000
    learning_rate = 1e-2
    continuous = env.continuous
    update_iter = 5
    render = False

    print("== Begin Main Training Thread, Number of Thread:{0} ==".format(num_process))
    print(" Environment Name: {0} \n Number of State: {1} \n Number of Action: {2} \n Continuous Action Space: {3}"
          .format(game_name, num_state, num_action, continuous))

    # set global network
    global_actor_network = Network.Actor(num_state, num_action, hidden_size, continuous)
    global_actor_network.share_memory()
    global_critic_network = Network.Critic(num_state, hidden_size)
    global_critic_network.share_memory()
    global_actor_optim = ShareAdam(params=global_actor_network.parameters(), lr=learning_rate)
    global_critic_optim = ShareAdam(params=global_actor_network.parameters(), lr=learning_rate)

    # data to plot
    global_episode = mp.Value('i', 0)
    global_reward = mp.Value('d', 0)
    reward_queue = mp.Queue()

    trainer_list = []
    for i in range(num_process):
        if i == 0 and render:
            trainer_obj = trainer(env,
                                  i,
                                  global_actor_network,
                                  global_critic_network,
                                  global_actor_optim,
                                  global_critic_optim,
                                  num_episode,
                                  update_iter,
                                  hidden_size,
                                  lr = learning_rate,
                                  render=True)
        else:
            trainer_obj = trainer(env,
                                  i,
                                  global_actor_network,
                                  global_critic_network,
                                  global_actor_optim,
                                  global_critic_optim,
                                  update_iter,
                                  num_episode,
                                  hidden_size,
                                  lr=learning_rate,
                                  render=False)
        trainer_list.append(trainer_obj)



if __name__ == "__main__":
    main_training_thread(7)
