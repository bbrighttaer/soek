# Author: bbrighttaer
# Project: soek
# Date: 6/28/19
# Time: 11:40 AM
# File: template.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc


class Trainer(abc.ABC):
    """A template class for model training. It's been constructed to be compatible with hyperparameter
    search, as implemented in this project.
    """

    @abc.abstractstaticmethod
    def initialize(hparams, train_dataset, val_dataset, *args, **kwargs):
        """
        Creates all model training elements.

        :param hparams: dict
            Hyperparamters for creating the model and the training algorithm elements.
        :param train_dataset: torch.utils.data
            Train dataset.
        :param val_dataset: torch.utils.data
            Validation dataset.
        :param args:
            Extra arguments to this method as desired.
        :param kwargs:
            Extra arguments to this method as desired.
        :return: tuple of model training elements.
        """
        pass

    @abc.abstractstaticmethod
    def data_provider(fold, *args, **kwargs):
        """
        Provides the datasets for each fold for training. Standard train-validation(-test) split is treated as a one
        split by the hyperparameter search algorithm.

        :param fold: int
            Current fold number.
        :param args:
            Extra arguments to this method as desired.
        :param kwargs:
            Extra arguments to this method as desired.
        :return: dict
            Datasets of the different phases of training. Valid keys are (train, val, test)
        """
        pass

    @abc.abstractstaticmethod
    def evaluate(*args, **kwargs):
        """
        Evaluation function that is called after every batch in the evaluation phase.

        :param args:
        :param kwargs:
        :return: float
            the score/performance of the model under the given set of hyperparameters.
        """
        pass

    @abc.abstractstaticmethod
    def train(*args, **kwargs):
        """
        Implements the main training loop of the mod

        :param args: tuple
            Training elements provided by the `initializer` method.
        :param kwargs: dict
            Extra arguments to this method as desired.
        :return:
        """
        pass

    @abc.abstractstaticmethod
    def evaluate_model(*args, **kwargs):
        """
        Procedures for loading and evaluating an already trained model goes here.
        :param args: tuple
            Model elements provided by the `initializer` method.
        :param kwargs: dict
            Extra arguments to this method as desired.
        :return:
        """
        pass

    @abc.abstractstaticmethod
    def save_model(model, path, name):
        """
        Saves the model parameters.

        :param model:
        :param path:
        :param name:
        :return:
        """
        pass

    @abc.abstractstaticmethod
    def load_model(path, name):
        """
        Loads the parameters of a model.

        :param path:
        :param name:
        :return: The saved state_dict.
        """
        pass
