import matplotlib.pyplot as plt
import random
import time


class moving_average:
    def __init__(self, beta):
        self.average_list = []
        self.value_list = []
        self.beta = beta
        self.pre_value = 0

    def _calculate_moving_average(self, new_value):
        self.value_list.append(new_value)
        self.pre_value = self.beta * self.pre_value + (1 - self.beta) * new_value
        current_value = self.pre_value / (1 - self.beta ** len(self.value_list))
        self.average_list.append(current_value)

    def plot_moving_average(self, new_value):
        plt.figure(2)
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        self._calculate_moving_average(new_value)
        plt.plot(self.value_list)
        plt.plot(self.average_list)
        plt.pause(0.001)

    def save_plot(self):
        plt.figure()
        plt.title('Training Reslut')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(self.value_list)
        plt.plot(self.average_list)
        plt.savefig("training_plot")



def plot_test():
    ma = moving_average(beta=0.9)
    for i in range(300):
        n = random.uniform(1, 1000)
        ma.plot_moving_average(n)
        time.sleep(0.1)


if __name__ == '__main__':
    plot_test()
