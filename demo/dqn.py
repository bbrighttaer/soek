# Author: bbrighttaer
# Project: soek
# Date: 8/30/19
# Time: 4:11 PM
# File: pytorch_mnist.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import collections
import copy
import os
import random
import time
from datetime import datetime as dt

import gym
import numpy as np
import torch
import torch.nn as nn
from skopt.plots import plot_evaluations, plot_objective

import soek as so
from demo.utils import wrappers as wp
from demo.utils.dqn_model import DQN
from soek import RandomSearch, BayesianOptSearch, DataNode
from soek.bopt import GPMinArgs

DEFAULT_ENV_NAME = 'PongNoFrameskip-v4'
MEAN_REWARD_BOUND = 19.5

REPLAY_SIZE = 10000
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 10 ** 5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

currentDT = dt.now()
date_label = currentDT.strftime("%Y_%m_%d__%H_%M_%S")

seed = 123
torch.manual_seed(seed)
torch.cuda.set_device(0)
np.random.seed(seed)
random.seed(seed)


def make_env(env_name):
    env = gym.make(env_name)
    env = wp.MaxAndSkipEnv(env)
    env = wp.FireResetEnv(env)
    env = wp.ProcessFrame84(env)
    env = wp.ImageToPyTorch(env)
    env = wp.BufferWrapper(env, 4)
    return wp.ScaledFloatFrame(env)


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device='cpu'):
        done_reward = None
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss(batch, net, tgt_net, gamma, device):
    states, actions, rewards, dones, next_states = batch
    # convert numpy arrays to torch tensors
    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.tensor(dones, dtype=torch.bool).to(device)

    state_action_values = net(states_v)
    state_action_values = state_action_values.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    # calculate target values aka 'lables' or 'ground truth'
    expected_state_action_values = rewards_v + gamma * next_state_values

    # calculate loss (regression)
    loss = nn.MSELoss()(state_action_values, expected_state_action_values)
    return loss


class DQNTraining(so.Trainer):

    @staticmethod
    def initialize(hparams, *args, **kwargs):
        dvc = hparams['device']
        env = make_env(hparams['env'])

        net = DQN(env.observation_space.shape, env.action_space.n).to(dvc)
        tgt_net = DQN(env.observation_space.shape, env.action_space.n).to(dvc)

        replay_buffer = ExperienceBuffer(REPLAY_SIZE)
        agent = Agent(env, exp_buffer=replay_buffer)

        # optimizer configuration
        optimizer = {
            "adadelta": torch.optim.Adadelta,
            "adagrad": torch.optim.Adagrad,
            "adam": torch.optim.Adam,
            "adamax": torch.optim.Adamax,
            "asgd": torch.optim.ASGD,
            "rmsprop": torch.optim.RMSprop,
            "Rprop": torch.optim.Rprop,
            "sgd": torch.optim.SGD,
        }.get(hparams["optimizer"].lower(), None)
        assert optimizer is not None, "{} optimizer could not be found"

        # filter optimizer arguments
        optim_kwargs = dict()
        optim_key = hparams["optimizer"]
        for k, v in hparams.items():
            if "optimizer__" in k:
                attribute_tup = k.split("__")
                if optim_key == attribute_tup[1] or attribute_tup[1] == "global":
                    optim_kwargs[attribute_tup[2]] = v
        optimizer = optimizer(net.parameters(), **optim_kwargs)

        return (net, tgt_net), agent, optimizer, hparams['batch_size'], hparams['gamma'], dvc

    @staticmethod
    def train(models, agent, optimizer, batch_size, gamma, dvc, n_iters=20000, sim_data_node=None):
        net, tgt_net = models
        buffer = agent.exp_buffer
        start = time.time()
        best_model_wts = net.state_dict()
        best_score = -10000
        best_epoch = -1

        total_rewards = []
        frame_idx = 0
        ts_frame = 0
        ts = time.time()
        best_mean_reward = None

        for _ in range(n_iters):
            frame_idx += 1
            # epsilon decay strategy
            epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

            reward = agent.play_step(net, epsilon, dvc)
            if reward is not None:
                total_rewards.append(reward)
                speed = (frame_idx - ts_frame) / (time.time() - ts)
                ts_frame = frame_idx
                ts = time.time()
                mean_reward = np.mean(total_rewards[-100:])
                print(f'{frame_idx}: done {len(total_rewards)} games, mean reward {mean_reward:.3f},'
                      f' eps {epsilon:.2f}, speed {speed:.2f}')

                if best_mean_reward is None or best_mean_reward < mean_reward:
                    if best_mean_reward is not None:
                        best_model_wts = copy.deepcopy(net.state_dict())
                        print(f'Best mean reward updated {best_mean_reward:.3f} -> {mean_reward:.3f}')
                    best_mean_reward = mean_reward
                if mean_reward > args.reward:
                    print(f'Solved in {frame_idx} frames')
                    break

            if len(buffer) < REPLAY_START_SIZE:
                continue

            if frame_idx % SYNC_TARGET_FRAMES == 0:
                tgt_net.load_state_dict(net.state_dict())

            # update weights
            optimizer.zero_grad()
            batch = buffer.sample(batch_size)
            loss_t = calc_loss(batch, net, tgt_net, gamma, dvc)
            loss_t.backward()
            optimizer.step()

        duration = time.time() - start
        print('\nModel training duration: {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))
        net.load_state_dict(best_model_wts)
        return {'model': net, 'score': best_score, 'epoch': best_epoch}

    @staticmethod
    def evaluate_model(eval_fn, *args, **kwargs):
        pass

    @staticmethod
    def save_model(model, path, name):
        os.makedirs(path, exist_ok=True)
        file = os.path.join(path, name + ".mod")
        torch.save(model.state_dict(), file)

    @staticmethod
    def load_model(path, name):
        return torch.load(os.path.join(path, name))


def get_hparam_config(flags):
    return {
        "env": so.ConstantParam(flags["env"]),
        "device": so.ConstantParam(flags["device"]),
        "gamma": so.RealParam(min=0.5, max=0.999),
        "batch_size": so.DiscreteParam(min=32, max=128),
        "optimizer": so.CategoricalParam(choices=["sgd", "adam", "adadelta", "adagrad", "adamax", "rmsprop"]),
        "optimizer__global__weight_decay": so.LogRealParam(),
        "optimizer__global__lr": so.LogRealParam(),
        "optimizer__sgd__nesterov": so.CategoricalParam(choices=[True, False]),
        "optimizer__sgd__momentum": so.LogRealParam(),
        "optimizer__adam__amsgrad": so.CategoricalParam(choices=[True, False]),
        "optimizer__adadelta__rho": so.LogRealParam(),
        "optimizer__adagrad__lr_decay": so.LogRealParam(),
        "optimizer__rmsprop__momentum": so.LogRealParam(),
        "optimizer__rmsprop__centered": so.CategoricalParam(choices=[True, False])
    }


def get_hparams(flags):
    return {
        "env": flags['env'],
        "device": flags['device'],
        "gamma": 0.99,
        "batch_size": 32,
        "optimizer__global__lr": 1e-3,
        "optimizer__global__weight_decay": 1e-5,
        "optimizer": "adam"
    }


class Flags(object):
    # enables using either object referencing or dict indexing to retrieve user passed arguments of flag objects.
    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        setattr(self, key, value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND,
                        help="Mean reward boundary for stop of training, default=%.2f" % MEAN_REWARD_BOUND)
    parser.add_argument("--hparam_search",
                        action="store_true",
                        help="If true, hyperparameter searching would be performed.")
    parser.add_argument("--hparam_search_alg",
                        type=str,
                        default="bayopt_search",
                        help="Hyperparameter search algorithm to use. One of [bayopt_search, random_search]")
    parser.add_argument('--model_dir',
                        type=str,
                        default='./model_dir',
                        help='Directory of model'
                        )
    parser.add_argument('--model_name',
                        type=str,
                        default='model-{}'.format(date_label),
                        help='Name of model'
                        )
    parser.add_argument("--eval",
                        action="store_true",
                        help="If true, a saved model is loaded and evaluated")
    parser.add_argument("--eval_model_name",
                        default=None,
                        type=str,
                        help="The filename of the model to be loaded from the directory specified in --model_dir")
    args = parser.parse_args()
    flags = Flags()
    args_dict = args.__dict__
    for arg in args_dict:
        setattr(flags, arg, args_dict[arg])
    device = torch.device("cuda:0" if flags.cuda and torch.cuda.is_available() else "cpu")
    flags['device'] = device

    # Simulation data resource tree
    sim_label = "Soek_DQN_demo"
    sim_data = DataNode(label=sim_label)

    trainer = DQNTraining()
    k = 1
    if flags.hparam_search:
        print("Hyperparameter search enabled: {}".format(flags.hparam_search_alg))

        # arguments to callables
        extra_init_args = {}
        extra_data_args = {}
        extra_train_args = {}

        hparams_conf = get_hparam_config(flags)

        search_alg = {"random_search": RandomSearch,
                      "bayopt_search": BayesianOptSearch}.get(flags.hparam_search_alg, BayesianOptSearch)
        # search_args = SearchArg(n_calls=10) # For random search.
        search_args = GPMinArgs(n_calls=10)  # For bayesian optimization.
        hparam_search = search_alg(hparam_config=hparams_conf,
                                   num_folds=k,
                                   initializer=trainer.initialize,
                                   train_fn=trainer.train,
                                   save_model_fn=trainer.save_model,
                                   alg_args=search_args,
                                   init_args=extra_init_args,
                                   data_args=extra_data_args,
                                   train_args=extra_train_args,
                                   data_node=sim_data,
                                   sim_label=sim_label,
                                   results_file="{}_{}_poc_{}".format(flags.hparam_search_alg, sim_label, date_label))
        stats = hparam_search.fit(model_dir="models", model_name=flags.model_name)
        print(stats)
        print("Best params = {}".format(stats.best()))

        if isinstance(hparam_search, BayesianOptSearch):
            res = hparam_search.results
            if res is not None:
                r = plot_evaluations(res, bins=10)
                eval_fig = np.random.choice(r.ravel()).figure
                eval_fig.savefig('hparam_evaluations.png')
                r = plot_objective(res)
                obj_func_eval_fig = np.random.choice(r.ravel()).figure
                obj_func_eval_fig.savefig('hparam_obj_func_eval.png')
    else:
        models, agent, optimizer, batch_size, gamma, dvc = trainer.initialize(get_hparams(flags))
        if flags.eval:
            pass
        else:
            model, score, epoch = trainer.train(models, agent, optimizer, batch_size, gamma, dvc,
                                                sim_data_node=sim_data)
            trainer.save_model(model, flags.model_dir,
                               "DQN_{}_{}_poc_{}_{:.5f}".format(sim_label, flags.model_name, epoch, score))
    sim_data.to_json('./')
