"""
@Date   ：2022/11/2
@Fun: SAC算法
"""
import random
import gym
import nasim
import torch
import math
import numpy as np
from matplotlib import pyplot as plt
env_name = "medium"
env = nasim.make_benchmark(env_name)
# 智能体状态
state = env.reset()
# 动作空间（连续性问题）
actions = env.action_space
print(state, actions)

# 演员模型：接收一个状态，使用抽样方式确定动作
class ModelAction(torch.nn.Module):
    """
    继承nn.Module，必须实现__init__() 方法和forward()方法。其中__init__() 方法里创建子模块，在forward()方法里拼接子模块。
    """
    def __init__(self):
        super().__init__()
        self.fc_state = torch.nn.Sequential(torch.nn.Linear(3, 64),
                                            torch.nn.ReLU()
                                            )
        self.mu = torch.nn.Linear(64, 1)
        self.std = torch.nn.Sequential(torch.nn.Linear(64,1),
                                       torch.nn.Softplus())

    def forward(self, state):
        state = self.fc_state(state)
        mu = self.mu(state)
        std = self.std(state)
        # 根据均值和方差，定义batch size个正太分布
        dist = torch.distributions.Normal(mu, std)
        # 重采样,采样b个样本
        sample = dist.rsample()
        action = torch.tanh(sample)

        # 求熵
        log_prob = dist.log_prob(sample)
        entropy = log_prob - (1 - action.tanh() ** 2 + 1e-7).log()
        entropy = -entropy

        return action * 2, entropy
# 策略网络
actor_model = ModelAction()

class ModelValue(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_state = torch.nn.Sequential(torch.nn.Linear(4, 64),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(64, 64),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(64, 1),)

    def forward(self, state, action):
        # 接收状态和动作特征，输出价值
        state_action = torch.cat([state, action], dim=1)
        return self.fc_state(state_action)

# 评论员模型:4个价值网络模型
critic_model1 = ModelValue()
critic_model2 = ModelValue()
critic_model_next1 = ModelValue()
critic_model_next2 = ModelValue()

critic_model_next1.load_state_dict(critic_model1.state_dict())
critic_model_next2.load_state_dict(critic_model2.state_dict())

# 优化参数a,对数更稳定
alpha = torch.tensor(math.log(0.01))
alpha.requires_grad = True

# 演员模型根据状态输出确定性动作值
def get_action(state):
    state = torch.FloatTensor(state).reshape(1, 3)
    action, _ = actor_model(state)

    return action.item()

# 离线学习策略，构建Replay Buffer 样本池
datas = []
def update_data():
    state = env.reset()
    done = False

    while not done:
        action = get_action(state)
        next_state, reward, done, _ = env.step([action])
        datas.append((state, action, reward, next_state, done))
        state = next_state

    while len(datas) > 10000:
        datas.pop(0)

# 从Buffer中获取一个batch样本，迭代训练时使用
def get_samples():
    batch_size = 64
    sample = random.sample(datas, batch_size)
    states = torch.FloatTensor([i[0] for i in sample]).reshape(-1, 3)
    actions = torch.FloatTensor([i[1] for i in sample]).reshape(-1, 1)
    rewards = torch.FloatTensor([i[2] for i in sample]).reshape(-1, 1)
    next_states = torch.FloatTensor([i[3] for i in sample]).reshape(-1, 3)
    dones = torch.LongTensor([i[4] for i in sample]).reshape(-1, 1)

    return states, actions, rewards, next_states, dones


# 监督目标y的计算(时序差分)
def get_target(reward, next_state, done):
    # 计算下一状态的动作和熵
    action, entropy = actor_model(next_state)
    target1 = critic_model_next1(next_state, action)
    target2 = critic_model_next1(next_state, action)
    target = torch.min(target1, target2)
    # 加上动作熵
    target *= 0.99
    target += alpha.exp() * entropy
    target *= (1 - done)
    target += reward

    return target

# 计算策略网络（演员）的优化loss，需要借助价值网络（评论员）价值最大化特性
def get_loss_action(state):
    action, entropy = actor_model(state)
    target1 = critic_model1(state, action)
    target2 = critic_model2(state, action)
    value = torch.min(target1, target2)

    loss_action = -alpha.exp() * entropy
    loss_action -= value

    return loss_action.mean(), entropy

# 模型软更新
def soft_update(model, model_next):
    # 模型以一个微小步长参数更新
    for old, new in zip(model.parameters(), model_next.parameters()):
        value = new.data * (1 - 0.005) + old.data * 0.005
        new.data.copy_(value)

def go_test():
    state = env.reset()
    reward_sum = 0
    over = False

    while not over:
        action = get_action(state)

        state, reward, over, _ = env.step([action])
        reward_sum += reward

    return reward_sum

def train():

    optimizer_action = torch.optim.Adam(actor_model.parameters(), lr=5e-4)
    optimizer_value1 = torch.optim.Adam(critic_model1.parameters(), lr=5e-3)
    optimizer_value2 = torch.optim.Adam(critic_model2.parameters(), lr=5e-3)

    # 要学习的正则化参数，控制熵的重要程度
    optimizer_alpha = torch.optim.Adam([alpha], lr=5e-3)

    # 玩N局游戏，每局游戏玩M次
    for epoch in range(200):
        # 更新一波数据
        update_data()

        # 每次更新后，大约学习200次（一个回合的样本大概是200）
        for _ in range(200):

            states, actions, rewards, next_states, dones = get_samples()
            rewards = (rewards + 8) / 8

            # 计算target
            target = get_target(rewards, next_states, dones)
            target = target.detach()
            # 计算两个value
            value1 = critic_model1(states, actions)
            value2 = critic_model2(states, actions)
            # 计算两个loss，两个value的目标都要贴近target
            loss_value1 = torch.nn.MSELoss()(value1, target)
            loss_value2 = torch.nn.MSELoss()(value2, target)

            optimizer_value1.zero_grad()
            loss_value1.backward()
            optimizer_value1.step()

            optimizer_value2.zero_grad()
            loss_value2.backward()
            optimizer_value2.step()

            # 计算演员模型的loss
            loss_action, entropy = get_loss_action(states)
            optimizer_action.zero_grad()
            loss_action.backward()
            optimizer_action.step()

            loss_alpha = alpha.exp() * (entropy + 1).detach()
            loss_alpha = loss_alpha.mean()

            optimizer_alpha.zero_grad()
            loss_alpha.backward()
            optimizer_alpha.step()

            soft_update(critic_model1, critic_model_next1)
            soft_update(critic_model2, critic_model_next2)

        if epoch % 20 == 0:
            result = sum([go_test() for _ in range(10)]) / 10
            print(epoch, result)

train()