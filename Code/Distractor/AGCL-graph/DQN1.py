import argparse
import pickle
from collections import namedtuple


import os, time
import numpy as np
import matplotlib.pyplot as plt

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter

# from torch.optim.swa_utils import AveragedModel, SWALR
# from torch.optim.lr_scheduler import CosineAnnealingLR


# Hyper-parameters
seed = 1
render = False
num_episodes = 2000
torch.manual_seed(seed)

##########################################
# env = gym.make('CartPole-v0').unwrapped
# num_state = env.observation_space.shape[0]
# num_action = env.action_space.n
# env.seed(seed)
###########################################

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])

class Net(nn.Module):
    def __init__(self, num_state, num_action):
        super(Net, self).__init__()
        self.num_state = num_state
        self.num_action = num_action
        self.fc1 = nn.Linear(self.num_state, 24)
        self.fc2 = nn.Linear(24, 10)
        self.fc3 = nn.Linear(10, self.num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_value = self.fc3(x)
        return action_value

class DQN():
    capacity = 8000
    learning_rate = 1e-3
    memory_count = 0
    batch_size = 256
    gamma = 0.995
    update_count = 0

    def __init__(self, num_state, num_action):
        super(DQN, self).__init__()
        self.num_state = num_state
        self.num_action = num_action


        self.rewards = []

        self.target_net_1, self.act_net_1 = Net(self.num_state, self.num_action), Net(self.num_state, self.num_action)
        self.target_net_2, self.act_net_2 = Net(self.num_state, self.num_action), Net(self.num_state, self.num_action)
        self.memory = [None]*self.capacity
        self.optimizer = optim.Adam(list(self.act_net_1.parameters()) + list(self.act_net_2.parameters()), self.learning_rate)
        self.loss_func = nn.MSELoss()
        self.writer = SummaryWriter('./DQN/logs')


    def select_action(self,state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        value = self.act_net_1(state) + self.act_net_2(state) 
        action_max_value, index = torch.max(value, 1)
        action = index.item()
        if np.random.rand(1) >= 0.9: # epslion greedy
            action = np.random.choice(range(self.num_action), 1).item()
        return action

    def store_transition(self,transition):
        index = self.memory_count % self.capacity
        self.memory[index] = transition
        self.memory_count += 1
        return self.memory_count >= self.capacity

    def update(self):
        if self.memory_count >= self.capacity:
            state = torch.tensor([t.state for t in self.memory]).float()
            action = torch.LongTensor([t.action for t in self.memory]).view(-1,1).long()
            reward = torch.tensor([t.reward for t in self.memory]).float()
            next_state = torch.tensor([t.next_state for t in self.memory]).float()

            reward = (reward - reward.mean()) / (reward.std() + 1e-7)
            with torch.no_grad():
                target_v_1 = reward + self.gamma * self.target_net_1(next_state).max(1)[0]
                target_v_2 = reward + self.gamma * self.target_net_2(next_state).max(1)[0]
                target_v = 0.5*target_v_1 + 0.5*target_v_2

            #Update...
            for index in BatchSampler(SubsetRandomSampler(range(len(self.memory))), batch_size=self.batch_size, drop_last=False):
                v_1 = (self.act_net_1(state).gather(1, action))[index]
                v_2 = (self.act_net_2(state).gather(1, action))[index]
                v = 0.5*v_1 + 0.5*v_2
                loss = self.loss_func(target_v[index].unsqueeze(1), v)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar('loss/value_loss', loss, self.update_count)
                self.update_count +=1
                if self.update_count % 100 == 0:
                    self.target_net_1.load_state_dict(self.act_net_1.state_dict())
                    self.target_net_2.load_state_dict(self.act_net_2.state_dict())

        else:
            print("Memory Buff is too less")

    def reset(self):
        self.optimizer.zero_grad()


	# def save_model(self, curriculum_no, beam_no, env_no):

	# 	experiment_file_name = '_c' + str(curriculum_no) + '_b' + str(beam_no) + '_e' + str(env_no)
	# 	path_to_save = self.log_dir + os.sep + self.env_id + experiment_file_name + '.npz'
	# 	torch.save(self.target_net.state_dict(), layer1 = self._model['W1'], layer2 = self._model['W2'])
	# 	print("saved to: ", path_to_save)

	# def load_model(self, curriculum_no, beam_no, env_no):

	# 	experiment_file_name = '_c' + str(curriculum_no) + '_b' + str(beam_no) + '_e' + str(env_no)
	# 	path_to_load = self.log_dir + os.sep + self.env_id + experiment_file_name + '.npz'
	# 	data = np.load(path_to_load)

    # def load_model(self, curriculum_no, beam_no, env_no):
    def load_model(self):

        directory = "weights/"
        # experiment_file_name = '_c' + str(curriculum_no) + '_b' + str(beam_no) + '_e' + str(env_no)
        experiment_file_name_1 = 'tree'
        experiment_file_name_2 = 'rock'
        self.target_net_1.load_state_dict(torch.load(directory + experiment_file_name_1))
        self.act_net_1.load_state_dict(torch.load(directory + experiment_file_name_1))
        self.target_net_2.load_state_dict(torch.load(directory + experiment_file_name_2))
        self.act_net_2.load_state_dict(torch.load(directory + experiment_file_name_2))



    def save_model(self, curriculum_no, beam_no, env_no):
        directory = "weights/"
        experiment_file_name = '_c' + str(curriculum_no) + '_b' + str(beam_no) + '_e' + str(env_no)
        torch.save(self.target_net.state_dict(), directory + experiment_file_name)

    def set_rewards(self, reward):
        self.rewards.append(reward)

def main():

    env = gym.make('CartPole-v0').unwrapped
    num_state = env.observation_space.shape[0]
    num_action = env.action_space.n
    agent = DQN(num_state, num_action)
    env.seed(seed)

    for i_ep in range(num_episodes):
        state = env.reset()
        if render: env.render()
        for t in range(10000):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            if render: env.render()
            transition = Transition(state, action, reward, next_state)
            agent.store_transition(transition)
            state = next_state
            if done or t >=9999:
                agent.writer.add_scalar('live/finish_step', t+1, global_step=i_ep)
                agent.update()
                if i_ep % 10 == 0:
                    print("episodes {}, step is {} ".format(i_ep, t))
                break

if __name__ == '__main__':
    main()