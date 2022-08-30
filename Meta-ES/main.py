from __future__ import absolute_import, division, print_function
import os
import gym
import argparse
import  time
import torch

import warnings

from env import get_env_space, get_env_info
from es_network import ESContinuous, ESDiscrete
from es_train_discrete import train_loop_es_discrete
from es_train_continuous import train_loop_es_continuous
parser = argparse.ArgumentParser(description='ES')
parser.add_argument('--env_name', default='CartPole-v1',
                    metavar='ENV', help='environment')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--lr_meta', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('--sigma', type=float, default=0.01, metavar='SD',
                    help='initial noise standard deviation')
parser.add_argument('--metasigma', type=float, default=0.01, metavar='MSD',
                    help='initial metanoise standard deviation')
parser.add_argument('--m', type=int, default=40, metavar='M',
                    help='meta batch size')
parser.add_argument('--n', type=int, default=40, metavar='N',
                    help='batch size')
parser.add_argument('--max-episode-length', type=int, default=500,
                    metavar='MEL', help='maximum length of an episode')
parser.add_argument('--alpha', type=float, default=0.1, metavar='TEM',
                    help='temperature factor')
parser.add_argument('--T', type=int, default=100,
                    metavar='T', help='maximum number of iteration')
parser.add_argument('--t', type=int, default=1,
                    metavar='T', help='iterations update mata')
parser.add_argument('--use_meta', default='False',
                     help='use meta or not')
parser.add_argument('--save', default='False',
                     help='save or not')
parser.add_argument('--save_path', default="./model",
                     help='save path')
parser.add_argument('--load', default='False',
                     help='load or not')
parser.add_argument('--load_name', default="2022-04-15-19-58-17.pt",
                     help='load filename')
parser.add_argument('--render', default='False',
                     help='if render')
if __name__ == '__main__':
    print("============================================================================================")
    # set device to cpu or cuda
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        device = torch.device('cpu')
        print("Device set to : cpu")
    print("============================================================================================")
    warnings.filterwarnings("ignore")
    args = parser.parse_args()
    assert args.n % 2 == 0
    assert args.m % 2 == 0
    args.save_path = args.save_path + "/" + args.env_name
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    env, env_continuous, num_states, num_actions = get_env_info(args.env_name)
    print("Env:{},Env_continuous:{},num_states:{},num_actions:{}".format(args.env_name,
                                                                         env_continuous,
                                                                         num_states,
                                                                         num_actions))
    if args.save == 'True':
        args.save_path = args.save_path + "/" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + ".pt"
        print('Network saved in' + args.save_path)
    if args.load == 'True':
        print('Load Network in' + args.save_path + "/" +args.load_name)

    print("Using Meta:{}".format(args.use_meta))


    if env_continuous:
        synced_model = ESContinuous(env)
        for param in synced_model.parameters():
            param.requires_grad = False
        train_loop_es_continuous(args, synced_model, env)
    else:
        synced_model = ESDiscrete(env)
        for param in synced_model.parameters():
            param.requires_grad = False
        train_loop_es_discrete(args, synced_model, env)




