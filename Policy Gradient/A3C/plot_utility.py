import matplotlib.pyplot as plt
import random
import time


class moving_average:
    def __init__(self, beta, critic=False):
        self.critic = critic
        self.beta = beta

        self.score_average_list = []
        self.score_value_list = []
        self.loss_average_list = []
        self.loss_value_list = []

        self.pre_score = 0
        self.pre_loss = 0

        if critic:
            self.critic_average_list = []
            self.critic_value_list = []
            self.pre_critic = 0

    def _calculate_moving_average(self, score, loss, critic_loss=None):
        self.score_value_list.append(score)
        self.pre_score = self.beta * self.pre_score + (1 - self.beta) * score
        current_value = self.pre_score / (1 - self.beta ** len(self.score_value_list))
        self.score_average_list.append(current_value)

        self.loss_value_list.append(loss)
        self.pre_loss = self.beta * self.pre_loss + (1 - self.beta) * loss
        current_value = self.pre_loss / (1 - self.beta ** len(self.loss_value_list))
        self.loss_average_list.append(current_value)

        if self.critic:
            self.critic_value_list.append(critic_loss)
            self.pre_critic = self.beta * self.pre_critic + (1 - self.beta) * critic_loss
            current_value = self.pre_critic / (1 - self.beta ** len(self.critic_value_list))
            self.critic_average_list.append(current_value)

    def plot_moving_average(self, score, actor_loss, critic_loss=None):
        self._calculate_moving_average(score, actor_loss, critic_loss)

        plt.figure(2)
        plt.clf()

        plt.subplot(121)
        plt.title('Total Reward')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.plot(self.score_value_list)
        plt.plot(self.score_average_list)

        if self.critic:
            plt.subplot(222)
        else:
            plt.subplot(122)
        plt.title('Actor Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.plot(self.loss_value_list)
        plt.plot(self.loss_average_list)

        if self.critic:
            plt.subplot(224)
            plt.title('Critic Loss')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.plot(self.critic_value_list)
            plt.plot(self.critic_average_list)
        plt.pause(0.001)

    def save_plot(self):
        plt.figure(2)
        plt.clf()
        plt.title('Training...')
        plt.subplot(121)
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.plot(self.score_value_list)
        plt.plot(self.score_average_list)

        plt.subplot(122)
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.plot(self.loss_value_list)
        plt.plot(self.loss_average_list)

