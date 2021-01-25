from Agent import Agent
from DQN import DQN
from ReplayMemory import ReplayMemory, Experience
from Environment_Management import environment_management
from plot_untility import moving_average
from Strategry import *
import torch as torch
from torch import optim
import torch.nn.functional as F
from torch import nn
from itertools import count


def main():
    game = 'CartPole-v0'
    max_exploration = 0.9
    min_exploration = 0.05
    rate_decay = 1e-3

    batch_size = 128
    memory_size = 10000
    discount_factor = 0.999
    target_update = 10
    lr = 5e-3
    num_episode = 50

    # == step 1: Initialize environment ==
    plot = moving_average(0.98)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_management = environment_management(game, device)
    env_management.reset()

    input_size = env_management.get_screen_size()
    output_size = env_management.get_action_size()

    epsilon_greedy = EpsilponGreedy(max_exploration, min_exploration, rate_decay)
    agent = Agent(output_size, epsilon_greedy, device)
    memory = ReplayMemory(memory_size, batch_size)

    # == step 2: Initialize policy network and target network
    policy_net = DQN(input_size, output_size).to(device)
    target_net = DQN(input_size, output_size).to(device)
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

    # We then set the weights and biases in the target_net to be the same as those in the policy_net
    # using PyTorch’s state_dict() and load_state_dict() functions.
    target_net.load_state_dict(policy_net.state_dict())

    # We also put the target_net into eval mode
    # which tells PyTorch that this network is not in training mode.
    target_net.eval()

    for episode in range(num_episode):
        # == step 3: Initialize the starting state. ==
        env_management.reset()
        state = env_management.get_state()

        for duration in count():
            '''
            For each steps:
                Select an action.
                    Via exploration or exploitation
                Execute selected action in an emulator.
                Observe reward and next state.
                Store experience in replay memory.
                Sample random batch from replay memory.
                Preprocess states from batch.
                Pass batch of preprocessed states to policy network.
                Calculate loss between output Q-values and target Q-values.
                    loss = q*(s, a) - q(s, a)
                         = E{Rt+1 + discount_factor * max(a')(s', a')} - q(s, a)
                    Requires a pass to the target network for the next state => max(a')(s', a')
                Gradient descent updates weights in the policy network to minimize loss.
                    After time steps, weights in the target network are updated to the weights in the policy network.
            '''
            action = agent.select_action(episode, state, policy_net)
            reward = env_management.take_action(action)

            next_state = env_management.get_state()
            new_experience = Experience(state, action, next_state, reward)
            memory.push(new_experience)
            state = next_state

            if memory.enough_samples():
                state_batch, action_batch, next_state_batch, reward_batch = memory.get_batch()
                state_action_value = policy_net(state_batch).gather(1, action_batch)

                # Expected values of actions for non_final_next_states are computed based
                # on the "older" target_net; selecting their best reward with argmax(dim=1).
                # Final states are represented with an all black screen.
                # Therefore, all the values within the tensor that represent that final state would be zero.
                non_final_state_location = next_state_batch.flatten(start_dim=1).sum(dim=1) != 0
                non_final_state_batch = next_state_batch[non_final_state_location]
                next_state_value = torch.zeros(batch_size, device=device)
                next_state_value[non_final_state_location] = target_net(non_final_state_batch).max(1)[0].detach()

                expected_state_action_values = reward_batch + discount_factor * next_state_value
                loss = loss_func(state_action_value, expected_state_action_values.reshape(-1, 1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if episode % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if env_management.done:
                plot.plot_moving_average(duration)
                break

    print('Complete')
    torch.save({
        'episode': num_episode,
        'model_state_dict': policy_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, "policy_network.tar")
    plot.save_plot()
    env_management.render()
    env_management.close()


if __name__ == '__main__':
    main()


