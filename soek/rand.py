# Author: bbrighttaer
# Project: soek
# Date: 6/27/19
# Time: 11:22 AM
# File: rand.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from soek import DataNode
from soek import ParamInstance, ParamSearchAlg


class RandomSearchCV(ParamSearchAlg):

    def _sample_params(self):
        hparams = {k: self.config[k].sample() for k in self.config}
        return hparams

    def fit(self, model_dir, model_name, max_iter=5, verbose=True, seed=None):
        iter_data_list = []
        if self.data_node is not None:
            self.data_node.data = iter_data_list

        # Random hyperparameter search.
        for i in range(max_iter):
            folds_data = []
            iter_data_node = DataNode(label="iteration-%d" % i, data=folds_data)
            iter_data_list.append(iter_data_node)

            # Get hyperparameters.
            hparams = self._sample_params()
            self.stats.current_param = ParamInstance(hparams)

            for fold in range(self.num_folds):
                if verbose:
                    print("\nFold {}, param search iteration {}, hparams={}".format(fold, i, hparams))

                k_node = DataNode(label="Random_search_fold-%d" % fold)
                folds_data.append(k_node)

                # Get data
                data = self.data_provider_fn(fold, **self.data_args)
                train_data = data["train"]
                val_data = data["val"]
                if "test" in data:
                    test_data = data["test"]
                    self.init_args["test_dataset"] = test_data

                # initialize model, dataloaders, and other elements.
                init_objs = self.initializer_fn(hparams, train_data, val_data, **self.init_args)

                # model training
                self.train_args["sim_data_node"] = k_node
                results = self.train_fn(*init_objs, **self.train_args)
                best_model, score, epoch = results['model'], results['score'], results['epoch']
                self.stats.current_param.add_score(score)

                # avoid nan scores in search. TODO: replace this hack with an organic approach.
                if str(score) == "nan":
                    score = -1e5

                # save model
                if model_dir is not None and model_name is not None:
                    self.save_model_fn(best_model, model_dir,
                                       "{}_{}-{}-fold{}-{}-{}-{}-{}-{:.4f}".format(self.dataset_label, self.sim,
                                                                                   self.stats.current_param.id,
                                                                                   fold, i, model_name,
                                                                                   self.split_label, epoch,
                                                                                   score))
            if verbose:
                print("Random search iter = {}: params = {}".format(i, self.stats.current_param))

            # move current hparams to records
            self.stats.update_records()
            self.stats.to_csv(self.results_file)
        return self.stats
