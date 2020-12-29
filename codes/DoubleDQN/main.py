#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-12 00:48:57
@LastEditor: John
LastEditTime: 2020-12-22 15:39:46
@Discription: 
@Environment: python 3.7.7
'''
import gym
import torch
from torch.utils.tensorboard import SummaryWriter
import os
from agent import DQN
from params import SEQUENCE,SAVED_MODEL_PATH,RESULT_PATH
from params import get_args
from utils import save_results

def train(cfg):
    print('Start to train !')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 检测gpu
    env = gym.make('CartPole-v0').unwrapped # 可google为什么unwrapped gym，此处一般不需要
    env.seed(1) # 设置env随机种子
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DQN(n_states=n_states, n_actions=n_actions, device=device, gamma=cfg.gamma, epsilon_start=cfg.epsilon_start,
                epsilon_end=cfg.epsilon_end, epsilon_decay=cfg.epsilon_decay, policy_lr=cfg.policy_lr, memory_capacity=cfg.memory_capacity, batch_size=cfg.batch_size)
    rewards = []
    moving_average_rewards = []
    ep_steps = []
    log_dir=os.path.split(os.path.abspath(__file__))[0]+"/logs/train/" + SEQUENCE
    writer = SummaryWriter(log_dir)
    for i_episode in range(1, cfg.train_eps+1):
        state = env.reset() # reset环境状态
        ep_reward = 0
        for i_step in range(1, cfg.train_steps+1):
            action = agent.choose_action(state) # 根据当前环境state选择action
            next_state, reward, done, _ = env.step(action) # 更新环境参数
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done) # 将state等这些transition存入memory
            state = next_state # 跳转到下一个状态
            agent.update() # 每步更新网络
            if done:
                break
        # 更新target network，复制DQN中的所有weights and biases
        if i_episode % cfg.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print('Episode:', i_episode, ' Reward: %i' %
              int(ep_reward), 'n_steps:', i_step, 'done: ', done,' Explore: %.2f' % agent.epsilon)
        ep_steps.append(i_step)
        rewards.append(ep_reward)
        # 计算滑动窗口的reward
        if i_episode == 1:
            moving_average_rewards.append(ep_reward)
        else:
            moving_average_rewards.append(
                0.9*moving_average_rewards[-1]+0.1*ep_reward)
<<<<<<< HEAD:codes/double_dqn/main.py
    import os
    import numpy as np
    save_path = os.path.dirname(__file__)+"/saved_model/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    agent.save_model(save_path+'checkpoint.pth')
    # 存储reward等相关结果
    output_path = os.path.dirname(__file__)+"/result/"
    # 检测是否存在文件夹
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    np.save(output_path+"rewards.npy", rewards)
    np.save(output_path+"moving_average_rewards.npy", moving_average_rewards)
    np.save(output_path+"steps.npy", ep_steps)
    print('Complete！')
    plot(rewards)
    plot(moving_average_rewards, ylabel="moving_average_rewards")
    plot(ep_steps, ylabel="steps_of_each_episode")
=======
        writer.add_scalars('rewards',{'raw':rewards[-1], 'moving_average': moving_average_rewards[-1]}, i_episode)
        writer.add_scalar('steps_of_each_episode',
                          ep_steps[-1], i_episode)
    writer.close()
    print('Complete training！')
    ''' 保存模型 '''
    if not os.path.exists(SAVED_MODEL_PATH): # 检测是否存在文件夹
        os.mkdir(SAVED_MODEL_PATH)
    agent.save_model(SAVED_MODEL_PATH+'checkpoint.pth')
    print('model saved！')
    '''存储reward等相关结果'''
    save_results(rewards,moving_average_rewards,ep_steps,tag='train',result_path=RESULT_PATH)
    
>>>>>>> dw/master:codes/DoubleDQN/main.py

def eval(cfg, saved_model_path = SAVED_MODEL_PATH):
    print('start to eval !')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 检测gpu
    env = gym.make('CartPole-v0').unwrapped # 可google为什么unwrapped gym，此处一般不需要
    env.seed(1) # 设置env随机种子
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DQN(n_states=n_states, n_actions=n_actions, device=device, gamma=cfg.gamma, epsilon_start=cfg.epsilon_start,
                epsilon_end=cfg.epsilon_end, epsilon_decay=cfg.epsilon_decay, policy_lr=cfg.policy_lr, memory_capacity=cfg.memory_capacity, batch_size=cfg.batch_size)
<<<<<<< HEAD:codes/double_dqn/main.py
    import os
    save_path = os.path.dirname(__file__)+"/saved_model/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    agent.load_model(save_path+'checkpoint.pth')
=======
    agent.load_model(saved_model_path+'checkpoint.pth')
>>>>>>> dw/master:codes/DoubleDQN/main.py
    rewards = []
    moving_average_rewards = []
    ep_steps = []
    log_dir=os.path.split(os.path.abspath(__file__))[0]+"/logs/eval/" + SEQUENCE
    writer = SummaryWriter(log_dir)
    for i_episode in range(1, cfg.eval_eps+1):
        state = env.reset()  # reset环境状态
        ep_reward = 0
        for i_step in range(1, cfg.eval_steps+1):
            action = agent.choose_action(state,train=False)  # 根据当前环境state选择action
            next_state, reward, done, _ = env.step(action)  # 更新环境参数
            ep_reward += reward
            state = next_state  # 跳转到下一个状态
            if done:
                break
        print('Episode:', i_episode, ' Reward: %i' %
<<<<<<< HEAD:codes/double_dqn/main.py
              int(ep_reward), 'end at n_steps:', i_step,' Explore: %.2f' % agent.epsilon)   # TODO 看下log，全都是0.01 消减地好厉害  # TODO 存进tensorboard去看...会不会导致events.out文件太大了...
=======
              int(ep_reward), 'n_steps:', i_step, 'done: ', done)
              
>>>>>>> dw/master:codes/DoubleDQN/main.py
        ep_steps.append(i_step)
        rewards.append(ep_reward)
        # 计算滑动窗口的reward
        if i_episode == 1:
            moving_average_rewards.append(ep_reward)
        else:
            moving_average_rewards.append(
                0.9*moving_average_rewards[-1]+0.1*ep_reward)
                
        writer.add_scalars('rewards',{'raw':rewards[-1], 'moving_average': moving_average_rewards[-1]}, i_episode)
        writer.add_scalar('steps_of_each_episode',
                          ep_steps[-1], i_episode)
    writer.close()
    '''存储reward等相关结果'''
    save_results(rewards,moving_average_rewards,ep_steps,tag='eval',result_path=RESULT_PATH)
    print('Complete evaling！')
    
if __name__ == "__main__":
    cfg = get_args()
    if cfg.train:
        train(cfg)
        eval(cfg)
    else:
        model_path = os.path.split(os.path.abspath(__file__))[0]+"/saved_model/"
        eval(cfg,saved_model_path=model_path)
