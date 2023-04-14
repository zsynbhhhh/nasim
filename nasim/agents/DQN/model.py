import numpy as np
import torch.nn
import torch.nn.functional as F



class Qnet(torch.nn.Module):
    '''只有一层的隐藏层的Q网络'''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class DQN:
    """DQN算法"""
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate,
                 gamma, epsilon, target_update, device, dqn_type="DoubleDQN_dim"):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        #使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update # 目标网络更新频率
        self.count = 0 #计数器,记录更新次数
        self.device = device
        self.dqn_type = dqn_type


    '''输入到模型获得动作, 使用的epsilon'''
    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item() # item表示实际的数据

        return action

    '''用来更新网络参数'''
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['action']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['done'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions) # 获得对应动作的Q值
        # 下一个状态的最大Q值
        if self.dqn_type == "DoubleDQN":
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        else:
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)


        q_target = rewards + self.gamma * max_next_q_values
        dqn_loss = torch.mean(F.mse_loss(q_values, q_target))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络

        self.count += 1

    def max_q_value(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        return self.q_net(state).max().item()