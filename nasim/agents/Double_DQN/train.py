import random
import gym
import nasim
import numpy as np
import collections
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import DQN
import rl_utils
from rl_utils import ReplayBuffer


lr = 1e-2 # 学习率
num_episodes = 200 #迭代次数
hidden_dim = 128 #隐藏层
gamma = 0.98
epsilon = 0.01
target_update = 50
batch_size = 64

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

buffer_size = 5000 # 观察数据的数量
minimal_size = 1000
replay_buffer = ReplayBuffer(buffer_size)
env_name = "medium"
env = nasim.make_benchmark(env_name)
state_dim = env.observation_space.shape[0] # 这里的输入是4个变量 速度, 位置, 尖端速度, 杆的角度
action_dim = env.action_space.n

def dis_to_con(disrete_action, env, action_dim):
    action_lowbound = env.action_space.low[0]
    action_upbound = env.action_space.high[0]
    return action_lowbound + (disrete_action / (action_dim - 1)) * (action_upbound - action_lowbound)

def train_DQN(agent, env, num_episodes, replay_buffer, minimal_size,
          batch_size):
    return_list = []
    max_q_value_list = []
    max_q_value = 0

    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    # 使用累计来表示max_q_value

                    max_q_value = agent.max_q_value(
                        state) * 0.005 + max_q_value * 0.995

                    max_q_value_list.append(max_q_value)
                    action_continuous = dis_to_con(action, env, agent.action_dim)  # 将角度转换为离散数据
                    next_state, reward, done, _ = env.step([action_continuous]) # 将角度输入到环境中获得下一个状态, 奖励, 是否停止
                    replay_buffer.add(state, action, reward, next_state, done) # 加入到缓冲区
                    state = next_state
                    episode_return += reward
                    # 当buffer的数量超过一定数量的时候, 进行Q网络训练
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size=batch_size) # 从缓冲区取数据
                        transition_dict = {
                            'states':b_s,
                            'actions':b_a,
                            'next_states':b_ns,
                            'rewards': b_r,
                            'dones': b_d,
                        }
                        agent.update(transition_dict) # 更新网络

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



agent = DQN(state_dim, hidden_dim, action_dim, lr,
            gamma, epsilon, target_update, device)

return_list, max_q_value_list = train_DQN(agent, env, num_episodes, replay_buffer, minimal_size,
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