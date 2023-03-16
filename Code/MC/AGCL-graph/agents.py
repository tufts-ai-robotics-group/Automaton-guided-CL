import argparse
import gym
import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import random
import os
from torch.nn import init

eps = np.finfo(np.float32).eps.item()



class ActorCriticPolicy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, num_actions, input_size, hidden_layer_size, learning_rate, gamma, decay_rate,
                 epsilon):
        super(ActorCriticPolicy, self).__init__()
        # store hyper-params
        self.num_actions = num_actions
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        self.affine = nn.Linear(self.input_size, self.hidden_layer_size)

        # actor's layer
        # self.action_head = nn.Linear(self.hidden_layer_size, self.num_actions)

        self.action_head = nn.Sequential(nn.Linear(self.hidden_layer_size, self.hidden_layer_size // 2),
                                         nn.Tanh(),
                                         nn.Linear(self.hidden_layer_size // 2, self.num_actions))

        # critic's layer
        # self.value_head = nn.Linear(self.hidden_layer_size, 1)
        self.value_head = nn.Sequential(nn.Linear(self.hidden_layer_size, self.hidden_layer_size//2),
                                         nn.Tanh(),
                                         nn.Linear(self.hidden_layer_size // 2, 1))

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,
                                          weight_decay=self.decay_rate)

    def reinit(self): # critic should be saved. actor is random
        for m in self.action_head.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        """
        forward of both actor and critic
        """
        enc0 = F.relu(self.affine(x))

        # actor: chooses action to take from state s_t
        # action_prob = F.softmax(self.action_head(x), dim=-1)
        action_prob = F.softmax(self.action_head(enc0), dim=-1)

        # critic: evaluates being in the state s_t
        # state_values = self.value_head(x)
        state_values = self.value_head(enc0)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values

    def select_action(self, state):
        state = torch.from_numpy(state).float()
        probs, state_value = self.forward(state)

        # epsilon greedy
        if random.uniform(0, 1) < self.epsilon:
            probs = torch.from_numpy(np.array([1/probs.shape[0] for _ in range(probs.shape[0])])).float()

        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(probs)
        # and sample an action using the distribution
        action = m.sample()

        # save to action buffer
        self.saved_actions.append((m.log_prob(action), state_value))

        # the action to take
        return action.item()

    def set_rewards(self, reward):
        self.rewards.append(reward)

    def finish_episode(self):
        """
        Training code. Calculates actor and critic loss and performs back propagation.
        """
        the_reward = 0.0
        saved_actions = self.saved_actions
        policy_losses = [] # save actor (policy) loss
        value_losses = [] # save critic (value) loss
        returns = [] # save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.rewards[::-1]:
            # calculate the discounted value
            the_reward = r + self.gamma * the_reward
            returns.insert(0, the_reward)

        returns = torch.tensor(returns)

        # if returns.size()[0] != 1:
        # returns = (returns - returns.mean()) / (returns.std() + eps)

        print("returns", returns)
        print(returns.shape)

        for (log_prob, value), the_reward in zip(saved_actions, returns):
            advantage = the_reward - value.item()

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([the_reward])))

        # reset gradients
        self.optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()


        # perform back propagation
        loss.backward()

        #clip the gradient to avoid NaN
        # cliping_value= 1
        # torch.nn.utils.clip_grad_norm_(self.parameters(), cliping_value)

        self.optimizer.step()

        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]


    def load_model(self, curriculum_no, beam_no, env_no):
        directory = "weights/"
        experiment_file_name = '_c' + str(curriculum_no) + '_b' + str(beam_no) + '_e' + str(env_no)
        """
        # path_to_load = self.log_dir + os.sep + self.env_id + experiment_file_name + '.npz'
        # data = np.load(path_to_load)
        # self._model['W1'] = data['layer1']
        # self._model['W2'] = data['layer2'] 
        """
        self.load_state_dict(torch.load(directory + experiment_file_name))


    def save_model(self, curriculum_no, beam_no, env_no):
        directory = "weights/"
        experiment_file_name = '_c' + str(curriculum_no) + '_b' + str(beam_no) + '_e' + str(env_no)

        if not os.path.exists(directory):
            os.makedirs(directory)

        """
        # path_to_save = self.log_dir + os.sep + self.env_id + experiment_file_name + '.npz'
        # np.savez(path_to_save, layer1=self._model['W1'], layer2=self._model['W2'])
        # print("saved to: ", path_to_save)
        """
        torch.save(self.state_dict(), directory + experiment_file_name)

    def reset(self):
        self.optimizer.zero_grad()
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,
        #                                   weight_decay=self.decay_rate)


if __name__ == '__main__':
    from torch.nn import init

    # s = nn.Sequential(nn.Linear(32, 32//2),
    #                   nn.Linear(32//2, 1))
    #
    # for m in s.modules():
    #     if isinstance(m, nn.Linear):
    #         m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
    #         if m.bias is not None:
    #             m.bias.data = init.constant_(m.bias.data, 0.0)



