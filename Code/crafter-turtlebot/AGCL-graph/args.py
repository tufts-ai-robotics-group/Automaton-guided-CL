import torch
import argparse
import os

class Args:
    def __init__(self):

        self.parser = argparse.ArgumentParser()

        # training
        self.parser.add_argument('--time_limit', type=int, default=600, metavar='N',
                            help='original t_imit in test_curr')
        # TODO:
        # environment
        # there are too many arguments in env, I put them in enviroments.py instead

        # agent
        self.parser.add_argument('--num_actions', type=int, default=5, help='num_actions')
        self.parser.add_argument('--input_size', type=int, default=83, help='input_size')
        self.parser.add_argument('--hidden_size', type=int, default=16, help='hidden_size')
        self.parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning_rate')
        self.parser.add_argument('--gamma', type=float, default=0.995, help='discount factor (default: 0.995)')
        self.parser.add_argument('--decay_rate', type=float, default=0.99, help='decay_rate')
        self.parser.add_argument('--epsilon', type=float, default=0.1, help='epsilon')
        self.parser.add_argument('--seed', type=int, default=10, help='random seed (default: 543)')
        self.parser.add_argument('--render', action='store_false', help='render the environment')
        self.parser.add_argument('--log-interval', type=int, default=10, help='interval between training status logs (default: 10)')
        self.parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='cuda:[d] | cpu')

    def update_args(self):
        args = self.parser.parse_args()
        return args


