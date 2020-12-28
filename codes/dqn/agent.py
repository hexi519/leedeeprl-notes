#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-12 00:50:49
@LastEditor: John
LastEditTime: 2020-10-15 21:56:21
@Discription: 
@Environment: python 3.7.7
'''
'''off-policy
'''


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
import numpy as np
from memory import ReplayBuffer
from model import FCN
class DQN:
    def __init__(self, n_states, n_actions, gamma=0.99, epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=200, memory_capacity=10000, policy_lr=0.01, batch_size=128, device="cpu"):
        self.actions_count = 0
        self.n_actions = n_actions  # 总的动作个数
        self.device = device  # 设备，cpu或gpu等
        self.gamma = gamma
        # e-greedy策略相关参数
        self.epsilon = 0
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.policy_net = FCN(n_states, n_actions).to(self.device)
        self.target_net = FCN(n_states, n_actions).to(self.device)
        # target_net的初始模型参数完全复制policy_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 不启用 BatchNormalization 和 Dropout  # TODO BN的原理 以及 是否过气了
        # 可查parameters()与state_dict()的区别，前者require_grad=True
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.loss = 0
        self.memory = ReplayBuffer(memory_capacity)
    
    def choose_action(self, state, train=True):
        '''选择动作
        '''
        if train:
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                math.exp(-1. * self.actions_count / self.epsilon_decay) # TODO 这个decay有什么讲究么 看下别人的代码
            self.actions_count += 1
            if random.random() > self.epsilon:
                with torch.no_grad():   # TODO 这部分不会被track梯度 why？ 比如说那个更新的式子，反向传播到底如何实做？
                    # 先转为张量便于丢给神经网络,state元素数据原本为float64
                    # 注意state=torch.tensor(state).unsqueeze(0)跟state=torch.tensor([state])等价 # TODO 所以为何要unsqueeze呢 😂
                    state = torch.tensor(
                        [state], device=self.device, dtype=torch.float32)
                    # 如tensor([[-0.0798, -0.0079]], grad_fn=<AddmmBackward>)
                    q_value = self.policy_net(state)
                    # tensor.max(1)返回每行的最大值以及对应的下标，   # TODO 这里想说的是返回的下标和数值都是tensor形式的，所以下面要用item()
                    # 如torch.return_types.max(values=tensor([10.3587]),indices=tensor([0]))
                    # 所以tensor.max(1)[1]返回最大值对应的下标，即action
                    action = q_value.max(1)[1].item()  
            else:
                action = random.randrange(self.n_actions)
            return action
        else: 
            with torch.no_grad():
                    # 先转为张量便于丢给神经网络,state元素数据原本为float64
                    # 注意state=torch.tensor(state).unsqueeze(0)跟state=torch.tensor([state])等价
                    state = torch.tensor(
                        [state], device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float32)    # TODO eval一般都用CPU？
                    # 如tensor([[-0.0798, -0.0079]], grad_fn=<AddmmBackward>)
                    q_value = self.target_net(state)
                    # tensor.max(1)返回每行的最大值以及对应的下标，
                    # 如torch.return_types.max(values=tensor([10.3587]),indices=tensor([0]))
                    # 所以tensor.max(1)[1]返回最大值对应的下标，即action
                    action = q_value.max(1)[1].item() 
            return action
    
    def update(self):
        if len(self.memory) < self.batch_size:
            return
        # 从memory中随机采样transition
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            self.batch_size) 
        # 转为张量
        # 例如tensor([[-4.5543e-02, -2.3910e-01,  1.8344e-02,  2.3158e-01],...,[-1.8615e-02, -2.3921e-01, -1.1791e-02,  2.3400e-01]])
        state_batch = torch.tensor(
            state_batch, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(
            1)  # 例如tensor([[1],...,[0]])
        reward_batch = torch.tensor(
            reward_batch, device=self.device, dtype=torch.float)  # tensor([1., 1.,...,1])
        next_state_batch = torch.tensor(
            next_state_batch, device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(
            done_batch), device=self.device).unsqueeze(1)  # 将bool转为float然后转为张量    # TODO 为何就action和done需要用unsqueeze? 为何要变成二维的？如果直接用dtype不可以转换格式么

        # 计算当前(s_t,a)对应的Q(s_t, a)
        # 关于torch.gather,对于a=torch.Tensor([[1,2],[3,4]])
        # 那么a.gather(1,torch.Tensor([[0],[1]]))=torch.Tensor([[1],[3]])
        q_values = self.policy_net(state_batch).gather( # TODO 为什么这里不直接记录Qvalue？ 还是说大家都是这样（看下别的repository）还是说为了接口统一(看下别的算法)--》 别的算法是的，比如说DDQN，就解耦了action的选择和评估。 别的repository情况就不知道了。
            dim=1, index=action_batch)  # 等价于self.forward
        # 计算所有next states的V(s_{t+1})，即通过target_net中选取reward最大的对应states
        next_state_values = self.target_net(
            next_state_batch).max(1)[0].detach()  # 比如tensor([ 0.0060, -0.0171,...,])
        # 计算 expected_q_value
        # 对于终止状态，此时done_batch[0]=1, 对应的expected_q_value等于reward
        expected_q_values = reward_batch + self.gamma * \
            next_state_values * (1-done_batch[0])
        # self.loss = F.smooth_l1_loss(q_values,expected_q_values.unsqueeze(1)) # 计算 Huber loss
        self.loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))  # 计算 均方误差loss
        # 优化模型
        self.optimizer.zero_grad()  # zero_grad清除上一步所有旧的gradients from the last step
        # loss.backward()使用backpropagation计算loss相对于所有parameters(需要gradients)的微分
        self.loss.backward()
        for param in self.policy_net.parameters():  # clip防止梯度爆炸
            param.grad.data.clamp_(-1, 1)   # TODO 这个梯度大小是如何来定的？观察以后调整？   
        self.optimizer.step()  # 更新模型

    def save_target_model(self,path):
        torch.save(self.target_net.state_dict(), path)

    def load_target_model(self,path):
        self.target_net.load_state_dict(torch.load(path))  
