from torch import multiprocessing as mp
from Agent import Agent
from Network import Actor, Critic
from ShareAdam import SharedAdam
from  plot_utility import moving_average
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
    num_episode = 100
    learning_rate = 1e-2
    continuous = env.continuous
    update_iter = 5
    render = False

    print("== Begin Main Training Thread, Number of Thread:{0} ==".format(num_process))
    print(" Environment Name: {0} \n Number of State: {1} \n Number of Action: {2} \n Continuous Action Space: {3}"
          .format(game_name, num_state, num_action, continuous))

    # set global network
    global_actor_network = Actor(num_state, num_action, hidden_size, continuous)
    global_actor_network.share_memory()
    global_critic_network = Critic(num_state, hidden_size)
    global_critic_network.share_memory()
    global_actor_optim = SharedAdam(params=global_actor_network.parameters(), lr=learning_rate)
    global_critic_optim = SharedAdam(params=global_critic_network.parameters(), lr=learning_rate)

    # data to plot
    plot = moving_average(0.98, critic=True)
    global_episode = mp.Value('i', 0)
    global_reward_queue = mp.Queue()
    global_actor_loss_queue = mp.Queue()
    global_critic_loss_queue = mp.Queue()

    # initialize training agent
    agent_list = []
    for i in range(num_process):
        render_option = (i == 0 and render)
        trainer_obj = Agent(env,
                            i,
                            global_actor_network,
                            global_critic_network,
                            global_actor_optim,
                            global_critic_optim,
                            num_episode,
                            update_iter,
                            global_episode,
                            global_reward_queue,
                            global_actor_loss_queue,
                            global_critic_loss_queue,
                            hidden_size,
                            render=render_option)
        agent_list.append(trainer_obj)

    # start training processes
    for agent in agent_list:
        agent.start()

    # plot data
    while True:
        reward = global_reward_queue.get()
        actor_loss = global_actor_loss_queue.get()
        critic_loss = global_critic_loss_queue.get()
        if reward and actor_loss and critic_loss:
            plot.plot_moving_average(reward, actor_loss, critic_loss)
        else:
            break

    for agent in agent_list:
        agent.join()


if __name__ == "__main__":
    main_training_thread(2)
