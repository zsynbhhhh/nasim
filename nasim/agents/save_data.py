import csv

from matplotlib import pyplot as plt

from nasim.agents.ppo import rl_utils


def save_episodes_return_list(filename, episodes_list, return_list):
    with open(filename, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Return'])
        for episode, ret in zip(episodes_list, return_list):
            writer.writerow([episode, ret])
def read_episodes_return_list(filename):
    episodes_list = []
    return_list = []
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头
        for row in reader:
            if row:  # 如果行不为空
                episodes_list.append(int(row[0]))
                return_list.append(float(row[1]))
    return episodes_list, return_list

def plot_save_to_pic_episode_return(save_to,episodes_list,return_list,rl_model,env_name):
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.subplots_adjust(left=0.15)
    # plt.ylim(-5000, 5000)
    plt.title(rl_model + ' on {}'.format(env_name))
    plt.savefig(save_to)
    plt.show()

