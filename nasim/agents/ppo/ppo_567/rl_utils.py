from nasim.envs.utils import AccessLevel
from tqdm import tqdm
import numpy as np
import torch
import collections
import random


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def get_count_hosts(env):
    _discover_host = []
    _compromised_host = []
    _root_host = []
    _user_host = []
    for i in env.network.address_space:
        state = env.current_state.copy()
        if state.get_host_discovered(i) == 1.0:
            _discover_host.append(i)
        if state.host_compromised(i) == 1.0:
            _compromised_host.append(i)
        if state.get_host_access_level(i) == AccessLevel.ROOT:
            _root_host.append(i)
        if state.get_host_access_level(i) == AccessLevel.USER:
            _user_host.append(i)
    return _discover_host, _compromised_host, _root_host, _user_host

def train_on_policy_agent567(env, agent, num_episodes):
    return_list = []
    step_list = []
    episode_num = 0
    action_target_count = {}
    for host in env.network.address_space:
        host = str(host)
        action_target_count[host] = 0
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                episode_step = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                action_valid = []
                state = env.reset()
                done = False
                # 一局放入到一块数据里面
                while not done and episode_step < 10000:
                    action = agent.take_action(state)
                    _discover_host, _compromised_host, \
                    _root_host, _user_host = get_count_hosts(env)
                    next_state, reward, done, _, _ = env.step(action)
                    specific_action = env.action_space.get_action(action)
                    # if the action are not scan actions ,and the action target is not valid action, reward will default to not increase
                    if specific_action.action_type not in ["ServiceScan", "OSScan", "SubnetScan", "ProcessScan"] and specific_action.target not in _discover_host:
                        pass
                    else:
                        # print(f"take action to {specific_action.target},action is {specific_action.action_type}")
                        episode_return += reward
                        episode_step += 1
                        action_valid.append(specific_action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['dones'].append(done)
                    transition_dict['rewards'].append(reward)
                    state = next_state
                episode_num += 1
                print(f"di {episode_num} episode returned {episode_return} step {episode_step} done = {done}")
                for action in action_valid:
                    for host in env.network.address_space:
                        if action.target == host:
                            host = str(host)
                            action_target_count[host] += 1
                            # print(f"主机{host}次数 + 1")
                return_list.append(episode_return)
                step_list.append(episode_step)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)

    return return_list, step_list, action_target_count


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                           'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta  # 时分差分算法, 表示从头到尾的影响
        advantage_list.append(advantage)
    advantage_list.reverse() # 序列反转
    return torch.tensor(advantage_list, dtype=torch.float)