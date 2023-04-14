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


class ReplayBuffer:
    """
    用于存放观察的数据
    """
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """添加观察的数据"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """取一个batch_size的数据"""
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)

lr = 2e-3 # 学习率
num_episodes = 500 #迭代次数
hidden_dim = 128 #隐藏层
gamma = 0.98
epsilon = 0.01
target_update = 10
batch_size = 64

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

buffer_size = 10000 # 观察数据的数量
minimal_size = 500
replay_buffer = ReplayBuffer(buffer_size)
env_name = "medium"
env = nasim.make_benchmark(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQN(state_dim, hidden_dim, action_dim, lr,
            gamma, epsilon, target_update, device)

return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration') as pbar:
        for i_episode in range(int(num_episodes/10)):
            episode_return = 0
            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state) # 输入到网络获得最大Q值对应的动作
                next_state, reward, done, _, _ = env.step(action) # 将action输入到环境中, 获得对应的下一个时刻的动作, 奖励值
                replay_buffer.add(state, action, reward, next_state, done) # 在缓冲区加入数据
                state = next_state
                episode_return += reward # 将奖励值进行累加
                # 当buffer的数量超过一定数量的时候, 进行Q网络训练
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size=batch_size)
                    transition_dict = {
                        'states':b_s,
                        'action':b_a,
                        'next_states':b_ns,
                        'rewards': b_r,
                        'done': b_d,
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



episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()