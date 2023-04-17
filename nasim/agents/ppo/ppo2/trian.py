import gym
import nasim
import torch
from model import PPO
import rl_utils
import matplotlib.pyplot as plt

from nasim.agents.save_data import save_episodes_return_list, read_episodes_return_list, \
    plot_save_to_pic_episode_return

actor_lr = 1e-4
critic_lr = 5e-3
num_episodes = 2000
hidden_dim = 128
gamma = 0.9
lmbda = 0.9
epochs = 10
eps = 0.2

device = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device('cpu')

env_name = "medium"
env = nasim.make_benchmark(env_name)
# env.seed(0)
# torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
            epochs, eps, gamma, device)
return_list = rl_utils.train_on_policy_agent2(env, agent, num_episodes)
episodes_list = list(range(len(return_list)))
save_episodes_return_list(r"D:\Nasim-Zsy-ppo\nasim\nasim\train_data\ppo\medium2\ppo2_return_save.csv",episodes_list,return_list)
episodes_csv, return_csv = read_episodes_return_list(r"D:\Nasim-Zsy-ppo\nasim\nasim\train_data\ppo\medium2\ppo2_return_save.csv")
plot_save_to_pic_episode_return(r"D:\Nasim-Zsy-ppo\nasim\nasim\train_data\ppo\medium2\ppo2_return_save.png",episodes_csv, return_csv,'PPO',env_name)

mv_return = rl_utils.moving_average(return_list, 9)
save_episodes_return_list(r"D:\Nasim-Zsy-ppo\nasim\nasim\train_data\ppo\medium2\ppo2_mv_return_save.csv",episodes_list,mv_return)
episodes_csv, mv_return_csv = read_episodes_return_list(r"D:\Nasim-Zsy-ppo\nasim\nasim\train_data\ppo\medium2\ppo2_mv_return_save.csv")
plot_save_to_pic_episode_return(r"D:\Nasim-Zsy-ppo\nasim\nasim\train_data\ppo\medium2\ppo2_mv_return_save.png",episodes_csv, return_csv,'PPO_mv_return',env_name)