# Author: bbrighttaer
# Project: soek
# Date: 6/28/19
# Time: 11:40 AM
# File: template.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


class Trainer(object):
    """A template class for model training. It's been constructed to be compatible with hyperparameter
    search, as implemented in this project.
    """

    @staticmethod
    def initialize(hparams, *args, **kwargs):
        """
        Creates all model training elements.

        :param hparams: dict
            Hyperparamters for creating the model and the training algorithm elements.
        :param args:
            Extra arguments to this method as desired.
        :param kwargs:
            Extra arguments to this method as desired.
        :return: tuple of model training elements.
        """
        pass

    @staticmethod
    def data_provider(*args, **kwargs):
        """
        Provides the data for training. Standard train-validation(-test) split is treated as a one
        split by the hyperparameter search algorithm.

        :param args:
            Extra arguments to this method as desired.
        :param kwargs:
            Extra arguments to this method as desired.
        :return: dict
            Datasets of the different phases of training. Valid keys are (train, val, test)
        """
        pass

    @staticmethod
    def evaluate(*args, **kwargs):
        """
        Evaluation function that is called after every batch in the evaluation phase.

        :param args:
        :param kwargs:
        :return: float
            the score/performance of the model under the given set of hyperparameters.
        """
        pass

    @staticmethod
    def train(*args, **kwargs):
        """
        Implements the main training loop of the mod

        :param args: tuple
            Training elements provided by the `initializer` method.
        :param kwargs: dict
            Extra arguments to this method as desired.
        :return: dict
            A dictionary with the following structure:
            `{'model': best_model, 'score': best_score, 'epoch': best_epoch}`
            This enables easy persistence of the model.
        """
        pass

    @staticmethod
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

    @staticmethod
    def save_model(model, path, name):
        """
        Saves the model parameters.

        :param model:
        :param path:
        :param name:
        :return:
        """
        pass

    @staticmethod
    def load_model(path, name):
        """
        Loads the parameters of a model.

        :param path:
        :param name:
        :return: The saved state_dict.
        """
        pass
