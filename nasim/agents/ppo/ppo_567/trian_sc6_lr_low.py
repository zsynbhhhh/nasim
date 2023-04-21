import csv

import gym
import nasim
import torch
from model import PPO
import rl_utils
import matplotlib.pyplot as plt

from nasim.agents.save_data import plot_save_to_pic_episode_return


def save_episodes_return_step_list(filename, episodes_list, return_list, step_list):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Return', 'Step'])
        for i in range(len(episodes_list)):
            writer.writerow([episodes_list[i], return_list[i], step_list[i]])


def save_target(filename, target_list):
    with open(filename, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['Target'])
        for target in target_list:
            writer.writerow([target])


def read_episodes_return_step_list(filename):
    episodes_list = []
    return_list = []
    step_list = []
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头
        for row in reader:
            if row:  # 如果行不为空
                episodes_list.append(int(row[0]))
                return_list.append(float(row[1]))
                step_list.append(int(row[2]))

    return episodes_list, return_list, step_list


actor_lr = 1e-6
critic_lr = 5e-5
num_episodes = 3000
hidden_dim = 128
gamma = 0.9
lmbda = 0.9
epochs = 10
eps = 0.2

device = torch.device("cuda") if torch.cuda.is_available() \
    else torch.device('cpu')

env_name = "sc_6"
env = nasim.make_benchmark(env_name)
# env.seed(0)
# torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
            epochs, eps, gamma, device)
return_list, step_list, action_target_count = rl_utils.train_on_policy_agent2(env, agent, num_episodes)
episodes_list = list(range(len(return_list)))
print(action_target_count)
# with open('action_target_count.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['主机', '选择次数'])  # 写入表头
#     for row in action_target_count:
#         writer.writerow(row)

save_episodes_return_step_list(r"D:\Nasim-Zsy-ppo\nasim\nasim\train_data\sc567\ppo_sc6_lr_low.csv", episodes_list,
                               return_list, step_list)

episodes_csv, return_csv, step_csv = read_episodes_return_step_list(
    r"D:\Nasim-Zsy-ppo\nasim\nasim\train_data\sc567\ppo_sc6_lr_low.csv")
plot_save_to_pic_episode_return(r"D:\Nasim-Zsy-ppo\nasim\nasim\train_data\sc567\ppo_sc6_lr_low_plot.png", episodes_csv,
                                return_csv, 'PPO', env_name)
plot_save_to_pic_episode_return(r"D:\Nasim-Zsy-ppo\nasim\nasim\train_data\sc567\ppo_sc6_lr_low_plot.png", episodes_csv,
                                step_csv, 'PPO', env_name)
