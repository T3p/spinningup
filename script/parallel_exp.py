import gym, safety_envs
import numpy as np

import screenutils as su

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seeds', nargs='+', help='Random seeds', type=int)
parser.add_argument('--cores', type=str, default='44-54')
parser.add_argument('--logdir', help='Directory for logs', type=str, default='./logs')
parser.add_argument('--env', help='Gym environment id', type=str, default='RBFMinigolf-v0')
parser.add_argument('--actor_lr', type=float, default=1.17e-3)
parser.add_argument('--critic_lr', type=float, default=2.49e-4)
parser.add_argument('--std', type=float, default=0.01)
parser.add_argument('--init', type=float, default=1.)
parser.add_argument('--horizon', type=int, default=20)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--epochs', type=int, default=700)
parser.add_argument('--steps_per_epoch', type=int, default=10000)
parser.add_argument('--test', type=int, default=100)
parser.add_argument('--expname', type=str, default='minigolf')
parser.add_argument('--policy', type=str, default=None)

args = parser.parse_args()

for seed in args.seeds:    
    screen = su.Screen(args.expname + str(seed), initialize=True)
    commands = 'conda activate py36 && taskset -ca %s python sp_run.py --seed %d --logdir %s --env %s --actor_lr %f --critic_lr %f --std %f --init %f --horizon %d --gamma %f --epochs %d --steps_per_epoch %d --test %d --expname %s --policy %s' % (args.cores, seed, args.logdir, args.env, args.actor_lr, args.critic_lr, args.std, args.init, args.horizon, args.gamma, args.epochs, args.steps_per_epoch, args.test, args.expname, args.policy)
    screen.send_commands(commands)





