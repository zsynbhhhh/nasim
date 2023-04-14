import random

import nasim
import numpy as np
import gym
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from model import DQN

import rl_utils

def dis_to_con(disrete_action, env, action_dim):
    action_lowbound = env.action_space.low[0]
    action_highbound = env.action_space.high[0]
    return action_lowbound + (disrete_action / (action_dim - 1)) * (action_highbound - action_lowbound)


def train_DQN(agent, env, num_episodes,
              replay_buffer, minimal_size,
              batch_size):
    return_list = []
    max_q_value_list = []
    max_q_value = 0
    # 玩10次游戏
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)

                    # 使用累计来表示max_q_value
                    max_q_value = agent.max_q_value(state) * 0.005 + max_q_value * 0.995
                    max_q_value_list.append(max_q_value)
                    action_continuous = dis_to_con(action, env, action_dim)
                    next_state, reward, done, _ = env.step([action_continuous])
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    # 当buffer的数量超过一定数量的时候, 进行Q网络训练
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states':b_s,
                            'actions':b_a,
                            'next_states':b_ns,
                            'rewards':b_r,
                            'dones':b_d,
                        }
                        agent.update(transition_dict)

                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / 10 * i + i_episode),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    return return_list, max_q_value_list





lr = 1e-2 # 学习率
num_episodes = 200 #迭代次数
hidden_dim = 128 #隐藏层 if torch.cuda.is_available() else torch.device('cpu')

buffer_size = 5000 # 观察数据的数量
minimal_size = 1000

env_name = "medium"
env = nasim.make_benchmark(env_name)
# env.seed(0)
# torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

gamma = 0.98
epsilon = 0.01
target_update = 50
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

random.seed(0)
np.random.seed(0)
# env.seed(0)
# torch.manual_seed(0)
replay_buffer = rl_utils.ReplayBuffer(buffer_size)
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device, 'DoubleDQN')

return_list, max_q_value_list = train_DQN(agent, env, num_episodes,
                                          replay_buffer, minimal_size,
                                          batch_size)


episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()

frames_list = list(range(len(max_q_value_list)))
plt.plot(frames_list, max_q_value_list)
plt.axhline(0, c='orange', ls='--')
plt.axhline(10, c='red', ls='--')
plt.xlabel('Frames')
plt.ylabel('Q value')
plt.title('DQN on {}'.format(env_name))
plt.show()