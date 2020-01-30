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
import os
import random
import time
from datetime import datetime as dt

import numpy as np
import torch

import soek as so
from soek import RandomSearch, BayesianOptSearch, DataNode
from soek.bopt import GPMinArgs

currentDT = dt.now()
date_label = currentDT.strftime("%Y_%m_%d__%H_%M_%S")

seed = 123
torch.manual_seed(seed)
torch.cuda.set_device(0)
np.random.seed(seed)
random.seed(seed)

CUDA = torch.cuda.is_available()

DATA_DIR = "~/.pytorch"


class DQNTraining(so.Trainer):

    @staticmethod
    def initialize(hparams, *args, **kwargs):

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
        optimizer = optimizer(model.parameters(), **optim_kwargs)

        return model, optimizer

    @staticmethod
    def train(model, optimizer, n_iters=5000, sim_data_node=None):
        start = time.time()
        best_model_wts = model.state_dict()
        best_score = -10000
        best_epoch = -1

        duration = time.time() - start
        print('\nModel training duration: {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))
        model.load_state_dict(best_model_wts)
        return {'model': model, 'score': best_score, 'epoch': best_epoch}

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
        # optimizer params
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
    return {"input_dim": 784,
            "hdims": [748, 1131, 947],
            "dprob": 0.2958835664552529,
            "batch_size": 256,
            "optimizer__global__lr": 0.0025595717862307707,
            "optimizer__global__weight_decay": 0.011761824861516251,
            "optimizer": "adamax"}


class Flags(object):
    # enables using either object referencing or dict indexing to retrieve user passed arguments of flag objects.
    def __getitem__(self, item):
        return self.__dict__[item]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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

    # Simulation data resource tree
    sim_label = "Soek_pytorch_mnist_demo"
    sim_data = DataNode(label=sim_label)

    trainer = DQNTraining()
    k = 1
    if flags.hparam_search:
        print("Hyperparameter search enabled: {}".format(flags.hparam_search_alg))

        # arguments to callables
        extra_init_args = {}
        extra_data_args = {}
        extra_train_args = {"n_iters": 1000}

        hparams_conf = get_hparam_config(flags)

        search_alg = {"random_search": RandomSearch,
                      "bayopt_search": BayesianOptSearch}.get(flags.hparam_search_alg, BayesianOptSearch)
        # search_args = SearchArg(n_calls=10) # For random search.
        search_args = GPMinArgs(n_calls=10)  # For bayesian optimization.
        hparam_search = search_alg(hparam_config=hparams_conf,
                                   num_folds=k,
                                   initializer=trainer.initialize,
                                   data_provider=trainer.data_provider,
                                   train_fn=trainer.train,
                                   save_model_fn=trainer.save_model,
                                   alg_args=search_args,
                                   init_args=extra_init_args,
                                   data_args=extra_data_args,
                                   train_args=extra_train_args,
                                   data_node=sim_data,
                                   split_label="train_val",
                                   sim_label=sim_label,
                                   dataset_label="mnist",
                                   results_file="{}_{}_poc_{}".format(flags.hparam_search_alg, sim_label, date_label))
        stats = hparam_search.fit(model_dir="models", model_name=flags.model_name)
        print(stats)
        print("Best params = {}".format(stats.best()))
    else:
        model, optimizer = trainer.initialize(hparams=get_hparams(flags))
        if flags.eval:
            pass
        else:
            model, score, epoch = trainer.train(model, optimizer, n_iters=10000, sim_data_node=sim_data)
            trainer.save_model(model, flags.model_dir,
                               "mnist_{}_{}_poc_{}_{:.5f}".format(sim_label, flags.model_name, epoch, score))
    sim_data.to_json('./')
