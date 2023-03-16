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
import time
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

        self.fc4 = nn.Linear(self.num_state, 24)
        self.fc5 = nn.Linear(24, 10)
        self.fc6 = nn.Linear(10, self.num_action)


    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        action_value1 = self.fc3(y)
        z = F.relu(self.fc4(x))
        z = F.relu(self.fc5(z))
        action_value2 = self.fc6(z)
        action_value = 0.48*action_value1 + 0.52*action_value2       
        return action_value


class small_net_1(nn.Module):
    def __init__(self, num_state, num_action):
        super(small_net_1, self).__init__()
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


class small_net_2(nn.Module):
    def __init__(self, num_state, num_action):
        super(small_net_2, self).__init__()
        self.num_state = num_state
        self.num_action = num_action
        self.fc4 = nn.Linear(self.num_state, 24)
        self.fc5 = nn.Linear(24, 10)
        self.fc6 = nn.Linear(10, self.num_action)

    def forward(self, x):
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        action_value = self.fc6(x)
        return action_value

class small_net_3(nn.Module):
    def __init__(self, num_state, num_action):
        super(small_net_3, self).__init__()
        self.num_state = num_state
        self.num_action = num_action
        self.fc7 = nn.Linear(self.num_state, 24)
        self.fc8 = nn.Linear(24, 10)
        self.fc9 = nn.Linear(10, self.num_action)

    def forward(self, x):
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        action_value = self.fc9(x)
        return action_value        

class medium_net_1(nn.Module):
    def __init__(self, num_state, num_action):
        super(medium_net_1, self).__init__()
        self.num_state = num_state
        self.num_action = num_action
        self.fc1 = nn.Linear(self.num_state, 24)
        self.fc2 = nn.Linear(24, 10)
        self.fc3 = nn.Linear(10, self.num_action)

        self.fc4 = nn.Linear(self.num_state, 24)
        self.fc5 = nn.Linear(24, 10)
        self.fc6 = nn.Linear(10, self.num_action)


    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        action_value1 = self.fc3(y)
        z = F.relu(self.fc4(x))
        z = F.relu(self.fc5(z))
        action_value2 = self.fc6(z)
        action_value = 0.46*action_value1 + 0.54*action_value2       
        return action_value

class big_net_1(nn.Module):
    def __init__(self, num_state, num_action):
        super(big_net_1, self).__init__()
        self.num_state = num_state
        self.num_action = num_action
        self.fc1 = nn.Linear(self.num_state, 24)
        self.fc2 = nn.Linear(24, 10)
        self.fc3 = nn.Linear(10, self.num_action)

        self.fc4 = nn.Linear(self.num_state, 24)
        self.fc5 = nn.Linear(24, 10)
        self.fc6 = nn.Linear(10, self.num_action)

        self.fc7 = nn.Linear(self.num_state, 24)
        self.fc8 = nn.Linear(24, 10)
        self.fc9 = nn.Linear(10, self.num_action)


    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        action_value1 = self.fc3(y)
        z = F.relu(self.fc4(x))
        z = F.relu(self.fc5(z))
        action_value2 = self.fc6(z)
        a = F.relu(self.fc7(x))
        a = F.relu(self.fc8(a))
        action_value3 = self.fc9(a)        
        action_value = 0.28*action_value1 + 0.29*action_value2 + 0.43*action_value3       
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

        self.target_net, self.act_net = Net(self.num_state, self.num_action), Net(self.num_state, self.num_action)
        self.memory = [None]*self.capacity
        self.optimizer = optim.Adam(self.act_net.parameters(), self.learning_rate)
        self.loss_func = nn.MSELoss()
        self.writer = SummaryWriter('./DQN/logs')


    def select_action(self,state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        value = self.act_net(state)
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
                target_v = reward + self.gamma * self.target_net(next_state).max(1)[0]

            #Update...
            for index in BatchSampler(SubsetRandomSampler(range(len(self.memory))), batch_size=self.batch_size, drop_last=False):
                v = (self.act_net(state).gather(1, action))[index]
                loss = self.loss_func(target_v[index].unsqueeze(1), (self.act_net(state).gather(1, action))[index])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar('loss/value_loss', loss, self.update_count)
                self.update_count +=1
                if self.update_count % 100 == 0:
                    self.target_net.load_state_dict(self.act_net.state_dict())
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

    # def load_model(self):
    def load_model(self,node_1,node_2):
        directory = "weights/"
        if node_1 == -2 and node_2 == -2:
            self.target_net = small_net_1(self.num_state, self.num_action)
            self.act_net = small_net_1(self.num_state, self.num_action)
            self.optimizer = optim.Adam(self.act_net.parameters(), self.learning_rate)

            self.save_name = 'src_1'
            print("loaded first model")
            time.sleep(3)
        if node_1 == -1 and node_2 == -1:
            self.target_net = small_net_2(self.num_state, self.num_action)
            self.act_net = small_net_2(self.num_state, self.num_action)
            self.save_name = 'src_2'
            self.optimizer = optim.Adam(self.act_net.parameters(), self.learning_rate)

            print("loaded second model")
            time.sleep(3)            
        if node_1 == 1 and node_2 == 0:
            self.temp_net1 = small_net_1(self.num_state, self.num_action)
            experiment_file_name_1 = 'src_1'
            self.temp_net1.load_state_dict(torch.load(directory+experiment_file_name_1))
            temp_params = dict(self.temp_net1.named_parameters())
            temp_params['fc7.weight'] = temp_params.pop('fc1.weight')
            temp_params['fc7.bias'] = temp_params.pop('fc1.bias')
            temp_params['fc8.weight'] = temp_params.pop('fc2.weight')
            temp_params['fc8.bias'] = temp_params.pop('fc2.bias')
            temp_params['fc9.weight'] = temp_params.pop('fc3.weight')
            temp_params['fc9.bias'] = temp_params.pop('fc3.bias')
            params1 = self.temp_net1.named_parameters()
            self.target_net = small_net_3(self.num_state, self.num_action)
            self.act_net = small_net_3(self.num_state, self.num_action)

            final_params = self.target_net.named_parameters()
            dict_final_params = dict(final_params)

            final_params_2 = self.act_net.named_parameters()
            dict_final_params_2 = dict(final_params_2)


            for name1, param1 in params1:
                if name1 in dict_final_params:
                    dict_final_params[name1].data.copy_(param1.data)

            for name1, param1 in params1:
                if name1 in dict_final_params_2:
                    dict_final_params_2[name1].data.copy_(param1.data)


            self.target_net.load_state_dict(dict_final_params)
            self.act_net.load_state_dict(dict_final_params_2)
            self.save_name = 'src_3'
            self.optimizer = optim.Adam(self.act_net.parameters(), self.learning_rate)

            print("loaded third model")
            time.sleep(3)
        if node_1 == 1 and node_2 == 2:
            experiment_file_name_1 = 'src_1'
            experiment_file_name_2 = 'src_2'
            self.temp_net1 = small_net_1(self.num_state, self.num_action)
            self.temp_net2 = small_net_2(self.num_state, self.num_action)
            self.temp_net1.load_state_dict(torch.load(directory + experiment_file_name_1))
            self.temp_net2.load_state_dict(torch.load(directory + experiment_file_name_2))
                
            params1 = self.temp_net1.named_parameters()
            params2 = self.temp_net2.named_parameters()

            self.final_net = medium_net_1(self.num_state, self.num_action)
            self.target_net = medium_net_1(self.num_state, self.num_action)
            self.act_net = medium_net_1(self.num_state, self.num_action)

            final_params = self.final_net.named_parameters()

            dict_final_params = dict(final_params)

            for name1, param1 in params1:
                if name1 in dict_final_params:
                    dict_final_params[name1].data.copy_(param1.data)

            for name2, param2 in params2:
                if name2 in dict_final_params:
                    dict_final_params[name2].data.copy_(param2.data)


            self.target_net.load_state_dict(dict_final_params)
            self.act_net.load_state_dict(dict_final_params)
            self.save_name = 'src_4'
            self.optimizer = optim.Adam(self.act_net.parameters(), self.learning_rate)

            print("loaded fourth model")
            time.sleep(3)
        if node_1 == 3 and node_2 == 4:
            experiment_file_name_1 = 'src_3'
            experiment_file_name_2 = 'src_4'
            self.temp_net1 = small_net_3(self.num_state, self.num_action)
            self.temp_net2 = medium_net_1(self.num_state, self.num_action)
            self.temp_net1.load_state_dict(torch.load(directory + experiment_file_name_1))
            self.temp_net2.load_state_dict(torch.load(directory + experiment_file_name_2))
                
            params1 = self.temp_net1.named_parameters()
            params2 = self.temp_net2.named_parameters()

            self.final_net = big_net_1(self.num_state, self.num_action)
            final_params = self.final_net.named_parameters()

            dict_final_params = dict(final_params)

            for name1, param1 in params1:
                if name1 in dict_final_params:
                    dict_final_params[name1].data.copy_(param1.data)

            for name2, param2 in params2:
                if name2 in dict_final_params:
                    dict_final_params[name2].data.copy_(param2.data)

            self.target_net = big_net_1(self.num_state, self.num_action)
            self.act_net = big_net_1(self.num_state, self.num_action)

            self.target_net.load_state_dict(dict_final_params)
            self.act_net.load_state_dict(dict_final_params)
            self.save_name = 'src_5'
            self.optimizer = optim.Adam(self.act_net.parameters(), self.learning_rate)

            print("loaded fifth model")
            time.sleep(3)


        if node_1 == 5 and node_2 == 0:
            experiment_file_name_1 = 'src_5'
            self.target_net = big_net_1(self.num_state, self.num_action)
            self.act_net = big_net_1(self.num_state, self.num_action)            
            self.target_net.load_state_dict(torch.load(directory+experiment_file_name_1))
            self.act_net.load_state_dict(torch.load(directory+experiment_file_name_1))
            self.save_name = 'src_6'
            self.optimizer = optim.Adam(self.act_net.parameters(), self.learning_rate)

            print("loaded sixth model")
            time.sleep(3)

    # def load_model(self):
    #     directory = "weights/"
    #     experiment_file_name_1 = 'test'
    #     self.target_net.load_state_dict(torch.load(directory+experiment_file_name_1))
    #     self.act_net.load_state_dict(torch.load(directory+experiment_file_name_1))
    #     print("loaded model")
    #     time.sleep(3)



    def save_model(self, curriculum_no, beam_no, env_no):
        directory = "weights/"
        experiment_file_name = self.save_name
        # experiment_file_name = '_c' + str(curriculum_no) + '_b' + str(beam_no) + '_e' + str(env_no)
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