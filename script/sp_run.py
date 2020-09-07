import gym, safety_envs
import numpy as np
import torch

import spinup.algos.pytorch.ddpg.ddpg as ddpg
import spinup.algos.pytorch.ddpg.core as core
from spinup.utils.test_policy import load_policy_and_env

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seed', help='Random seed', type=int, default=None)
parser.add_argument('--logdir', help='Directory for logs', type=str, default='./logs')
parser.add_argument('--env', help='Gym environment id', type=str, default='RBFMinigolf-v0')
parser.add_argument('--actor_lr', type=float, default=1e-5)
parser.add_argument('--critic_lr', type=float, default=1e-4)
parser.add_argument('--std', type=float, default=0.04)
parser.add_argument('--init', type=float, default=1.)
parser.add_argument('--horizon', type=int, default=20)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--epochs', type=int, default=700)
parser.add_argument('--steps_per_epoch', type=int, default=10000)
parser.add_argument('--test', type=int, default=100)
parser.add_argument('--expname', type=str, default='minigolf')
parser.add_argument('--policy', type=str, default=None)

args = parser.parse_args()
expname = '%s%d' % (args.expname, args.seed)

logger_kwargs = {'output_dir': args.logdir+'/%s/%d'%(args.expname, args.seed), 'exp_name': expname, 'output_fname': '%s_lddpg_%d.csv' % (args.expname, args.seed)}

if args.policy is None or args.policy=='None':
    ac = core.MLPActorCritic 
else:
    def ac(obs_space, act_space,  **kwargs):
        with open(args.policy, 'rb') as fp:
            actor_params = np.load(fp)
        ac = core.MLPActorCritic(obs_space, act_space, **kwargs)
        ac.state_dict()['pi.pi.0.weight'].copy_(torch.tensor(actor_params))
        return ac

ddpg.ddpg(lambda : gym.make(args.env), actor_critic=ac, init=args.init, pi_lr=args.actor_lr, q_lr=args.critic_lr, act_noise=args.std, max_ep_len=args.horizon, gamma=args.gamma, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs, num_test_episodes=args.test, logger_kwargs=logger_kwargs, seed=args.seed)




