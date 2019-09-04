# Author: bbrighttaer
# Project: soek
# Date: 8/30/19
# Time: 4:11 PM
# File: pytorch_demo.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import copy
import os
import random
import time
from datetime import datetime as dt

import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as sch
import torch.utils.data as data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import soek as so
from soek import RandomSearchCV, BayesianOptSearchCV, DataNode

currentDT = dt.now()
date_label = currentDT.strftime("%Y_%m_%d__%H_%M_%S")

seed = 123
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

CUDA = torch.cuda.is_available()

DATA_DIR = "~/.pytorch"


class Demo(so.Trainer):

    @staticmethod
    def initialize(hparams, train_dataset, val_dataset, *args, **kwargs):
        # model construction
        in_dim = hparams["input_dim"]
        lyr_lst = []
        for out_dim in hparams["hdims"]:
            lyr_lst.append(nn.Linear(in_dim, out_dim))
            lyr_lst.append(nn.BatchNorm1d(out_dim))
            lyr_lst.append(nn.ReLU())
            lyr_lst.append(nn.Dropout(hparams["dprob"]))
            in_dim = out_dim
        model = nn.Sequential(*lyr_lst,
                              nn.Linear(in_dim, 10))
        if CUDA:
            model = model.cuda()

        # data loaders
        train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=hparams["batch_size"])
        val_loader = data.DataLoader(val_dataset, shuffle=False, batch_size=hparams["batch_size"])

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

        # metrics
        metrics = (accuracy_score,)
        return model, optimizer, {"train": train_loader,
                                  "val": val_loader}, metrics

    @staticmethod
    def data_provider(fold, *args, **kwargs):
        train_dataset = dsets.MNIST(root=DATA_DIR, train=True, transform=transforms.ToTensor(), download=True)
        val_dataset = dsets.MNIST(root=DATA_DIR, train=False, transform=transforms.ToTensor(), download=True)

        return {"train": train_dataset, "val": val_dataset}

    @staticmethod
    def evaluate(eval_dict, y, y_pred, metrics):
        for metric in metrics:
            y_pred = torch.max(y_pred, dim=1)[1].cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            eval_dict[metric.__name__] = metric(y, y_pred)
        return np.mean(list(eval_dict.values()))

    @staticmethod
    def train(eval_fn, model, optimizer, data_loaders, metrics, n_iters=5000, sim_data_node=None):
        start = time.time()
        best_model_wts = model.state_dict()
        best_score = -10000
        best_epoch = -1
        n_epochs = n_iters // len(data_loaders["train"])
        scheduler = sch.StepLR(optimizer, step_size=30, gamma=0.01)
        criterion = nn.CrossEntropyLoss()
        first_epoch_loss = None

        # sub-nodes of sim data resource
        loss_lst = []
        train_loss_node = DataNode(label="training_loss", data=loss_lst)
        metrics_dict = {}
        metrics_node = DataNode(label="validation_metrics", data=metrics_dict)
        scores_lst = []
        scores_node = DataNode(label="validation_score", data=scores_lst)

        # add sim data nodes to parent node
        if sim_data_node:
            sim_data_node.data = [train_loss_node, metrics_node, scores_node]

        # Main training loop
        for epoch in range(n_epochs):
            for phase in ["train", "val"]:
                if phase == "train":
                    print("Training....")
                    # Training mode
                    model.train()
                else:
                    print("Validation...")
                    # Evaluation mode
                    model.eval()

                epoch_losses = []
                epoch_scores = []

                # Iterate through mini-batches
                i = 0
                for X, y in tqdm(data_loaders[phase]):
                    X = X.view(X.shape[0], -1)
                    if CUDA:
                        X = X.cuda()
                        y = y.cuda()

                    optimizer.zero_grad()

                    # forward propagation
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        y_pred = model(X)
                        loss = criterion(y_pred, y.squeeze())

                    if phase == "train":
                        print(
                            "\tEpoch={}/{}, batch={}/{}, loss={:.4f}".format(epoch + 1, n_epochs, i + 1,
                                                                             len(data_loaders[phase]),
                                                                             loss.item()))
                        # for epoch stats
                        epoch_losses.append(loss.item())

                        # for sim data resource
                        loss_lst.append(loss.item())

                        # optimization ops
                        loss.backward()
                        optimizer.step()
                    else:
                        if str(loss.item()) != "nan":  # useful in hyperparameter search
                            eval_dict = {}
                            score = eval_fn(eval_dict, y, y_pred, metrics)
                            # for epoch stats
                            epoch_scores.append(score)

                            # for sim data resource
                            scores_lst.append(score)
                            for m in eval_dict:
                                if m in metrics_dict:
                                    metrics_dict[m].append(eval_dict[m])
                                else:
                                    metrics_dict[m] = [eval_dict[m]]

                            print("\nEpoch={}/{}, batch={}/{}, "
                                  "evaluation results= {}, score={}".format(epoch + 1, n_epochs, i + 1,
                                                                            len(data_loaders[phase]),
                                                                            eval_dict, score))

                    i += 1
                # End of mini=batch iterations.

                if phase == "train":
                    # Adjust the learning rate.
                    scheduler.step()

                    ep_loss = np.nanmean(epoch_losses)
                    if first_epoch_loss is None:
                        first_epoch_loss = ep_loss
                    print("\nPhase: {}, avg task loss={:.4f}, ".format(phase, ep_loss))
                else:
                    mean_score = np.mean(epoch_scores)
                    if best_score < mean_score:
                        best_score = mean_score
                        best_model_wts = copy.deepcopy(model.state_dict())
                        best_epoch = epoch

        duration = time.time() - start
        print('\nModel training duration: {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))
        model.load_state_dict(best_model_wts)
        return model, best_score, best_epoch

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
    return {"input_dim": so.ConstantParam(784),
            "hdims": so.DiscreteParam(min=50, max=2048, size=so.DiscreteParam(min=1, max=4)),
            "dprob": so.RealParam(),
            "batch_size": so.CategoricalParam(choices=[64, 128, 256, 512]),
            # optimizer params
            "optimizer": so.CategoricalParam(choices=["sgd", "adam", "adadelta", "adagrad", "adamax", "rmsprop"]),
            "optimizer__global__weight_decay": so.LogRealParam(),
            "optimizer__global__lr": so.LogRealParam(),

            # # SGD
            "optimizer__sgd__nesterov": so.CategoricalParam(choices=[True, False]),
            "optimizer__sgd__momentum": so.LogRealParam(),

            # ADAM
            "optimizer__adam__amsgrad": so.CategoricalParam(choices=[True, False]),

            # Adadelta
            "optimizer__adadelta__rho": so.LogRealParam(),

            # Adagrad
            "optimizer__adagrad__lr_decay": so.LogRealParam(),

            # RMSprop
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

    trainer = Demo()
    k = 1
    if flags.hparam_search:
        print("Hyperparameter search enabled: {}".format(flags.hparam_search_alg))

        # arguments to callables
        extra_init_args = {}
        extra_data_args = {}
        extra_train_args = {"n_iters": 5000}

        hparams_conf = get_hparam_config(flags)

        search_alg = {"random_search": RandomSearchCV,
                      "bayopt_search": BayesianOptSearchCV}.get(flags.hparam_search_alg, BayesianOptSearchCV)

        hparam_search = search_alg(hparam_config=hparams_conf,
                                   num_folds=k,
                                   initializer=trainer.initialize,
                                   data_provider=trainer.data_provider,
                                   train_fn=trainer.train,
                                   eval_fn=trainer.evaluate,
                                   save_model_fn=trainer.save_model,
                                   init_args=extra_init_args,
                                   data_args=extra_data_args,
                                   train_args=extra_train_args,
                                   data_node=sim_data,
                                   split_label="train_val",
                                   sim_label=sim_label,
                                   dataset_label="mnist")
        stats = hparam_search.fit(model_dir="models",
                                  model_name=flags.model_name, max_iter=40, seed=seed)
        print(stats)
        stats.to_csv("{}_{}_poc_{}.csv".format(flags.hparam_search_alg, sim_label, date_label))
        print("Best params = {}".format(stats.best(m="max")))
    else:
        datasets = trainer.data_provider(fold=k)
        model, optimizer, data_loaders, metrics = trainer.initialize(hparams=get_hparams(flags),
                                                                     train_dataset=datasets["train"],
                                                                     val_dataset=datasets["val"])
        if flags.eval:
            pass
        else:
            # Train the model
            model, score, epoch = trainer.train(trainer.evaluate, model, optimizer, data_loaders, metrics,
                                                n_iters=10000, sim_data_node=sim_data)
            # Save the model
            trainer.save_model(model, flags.model_dir,
                               "mnist_{}_{}_poc_{}_{:.5f}".format(sim_label, flags.model_name, epoch, score))
