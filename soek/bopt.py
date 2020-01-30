# Author: bbrighttaer
# Project: soek
# Date: 7/5/19
# Time: 8:31 AM
# File: bopt.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc

from skopt import gp_minimize, gbrt_minimize, forest_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args

from soek import ConstantParam, LogRealParam, DiscreteParam, CategoricalParam, DictParam, \
    RealParam
from soek import DataNode
from soek import ParamSearchAlg, ParamInstance

size_suffix = "_size"


def _create_space(space, clazz, param_name, conf):
    def _append(low, high, name):
        kwargs = {"low": low, "high": high, "name": name}
        if isinstance(conf, LogRealParam):
            kwargs["prior"] = "log-uniform"
        space.append(clazz(**kwargs))

    if clazz == Categorical:
        size = conf.size.max if isinstance(conf.size, DiscreteParam) else conf.size
        for i in range(size):
            space.append(Categorical(categories=conf.choices,
                                     name="{}___{}".format(param_name, i)))
    else:
        # if size is also a hyperparameter
        if isinstance(conf.size, DiscreteParam):
            # first add the size as a hyperparameter
            space.append(
                Integer(low=conf.size.min,
                        high=conf.size.max,
                        name="{}{}".format(param_name, size_suffix))
            )
            size = conf.size.max
        else:
            size = conf.size

        for i in range(size):
            low = pow(10, conf.min) if isinstance(conf, LogRealParam) else conf.min
            high = pow(10, conf.max) if isinstance(conf, LogRealParam) else conf.max
            _append(low, high, "{}___{}".format(param_name, i))


def _transform_hparams_dict(params_config):
    return _to_skopt_space(params_config, [])


def _to_skopt_space(params_config, space, prefix=None):
    """
    Converts the params config, in dict, to scikit-optimize space/dimension format.

    Mapping:
    DiscreteParam --> skopt.space.Integer
    CategoricalParam --> skopt.space.Categorical
    LogRealParam --> skopt.space.Real
    RealParam --> skopt.space.Real

    Args:
    :param params_config: User-defined hyperparameter config used in creating the search algorithm.
    :param space: Used to aggregate all defined parameters transformed into skopt space objects.
    :param prefix: Used during recursion to identify sub-configurations and add transform them appropriately.
    :return: scikit-optimize compatible list of spaces to be used by `gp_minimize()`
    """
    for param, conf in zip(params_config.keys(), params_config.values()):
        clazz = None
        # Constant params are ignored in parameter search.
        if isinstance(conf, ConstantParam):
            continue

        if isinstance(conf, DiscreteParam):
            clazz = Integer
        elif isinstance(conf, CategoricalParam):
            clazz = Categorical
        elif isinstance(conf, LogRealParam) or isinstance(conf, RealParam):
            clazz = Real
        elif isinstance(conf, DictParam):
            _to_skopt_space(conf, space, prefix=param)

        if clazz:
            name = "{}__{}".format(prefix, param) if prefix else param
            _create_space(space, clazz, name, conf)
    return space


def _to_hparams_dict(bopt_params, params_config):
    return _convert_to_hparams(bopt_params, params_config, {})


def _convert_to_hparams(bopt_params, params_config, hparams, prefix=None):
    """
    Converts an skopt parameter set to a params dict usable in model training

    :param bopt_params: kwargs supplied by skopt
    :param params_config: The hyperparameter search config used in creating the search algorithm.
    :param hparams: used to collect the hyperparameters in recursion.
    :param prefix: used in recursion ops to identify sub-configs.
    :return: The hyperparameters in the same structure as `params_config` but with specific values for training.
    """
    for i, param in enumerate(params_config):
        conf = params_config[param]
        if isinstance(conf, ConstantParam):
            hparams[param] = conf.sample()
            continue
        elif isinstance(conf, DictParam):
            grp_dict = {}
            _convert_to_hparams(bopt_params, conf.p_dict, grp_dict, prefix="{}__".format(param))
            hparams[param] = grp_dict
            continue

        if isinstance(conf.size, DiscreteParam):
            size = bopt_params[param + size_suffix]
        else:
            size = conf.size

        val = []
        for j in range(size):
            prefix = prefix if prefix else ''
            val.append(bopt_params["{}{}___{}".format(prefix, param, j)])
        if not conf.is_list:
            val = val[0]
        hparams[param] = val

    return hparams


def _create_objective(alg, model_dir, model_name, verbose=True):
    count = Count()
    iter_data_list = []
    if alg.data_node is not None:
        alg.data_node.data = iter_data_list

    @parse_config(alg.config)
    def objective(**bopt_params):
        count.inc()
        folds_data = []
        iter_data_node = DataNode(label="iteration-%d" % count.i, data=folds_data)
        iter_data_list.append(iter_data_node)

        # Get hyperparameters.
        hparams = _to_hparams_dict(bopt_params=bopt_params, params_config=alg.config)
        alg.stats.current_param = ParamInstance(hparams)

        for fold in range(alg.num_folds):
            k_node = DataNode(label="BayOpt_search_fold-%d" % fold)
            folds_data.append(k_node)

            # Get data
            if alg.data_provider_fn is not None:
                data = alg.data_provider_fn(fold, **alg.data_args)
                if isinstance(data, dict):
                    data = list(data.values())
            else:
                data = {}

            if verbose:
                print("\nFold {}, param search iteration {}, hparams={}".format(fold, count.i, hparams))

            # initialize model, dataloaders, and other elements.
            init_objs = alg.initializer_fn(hparams, *data, **alg.init_args)

            # start of training with selected parameters
            alg.train_args["sim_data_node"] = k_node
            results = alg.train_fn(*init_objs, **alg.train_args)
            best_model, score, epoch = results['model'], results['score'], results['epoch']
            alg.stats.current_param.add_score(score)
            # end of training

            # save model
            if model_dir is not None and model_name is not None:
                alg.save_model_fn(best_model, model_dir,
                                  "{}_{}-{}-fold{}-{}-{}-{}-{}-{:.4f}".format(alg.dataset_label, alg.sim,
                                                                              alg.stats.current_param.id,
                                                                              fold, count.i, model_name,
                                                                              alg.split_label,
                                                                              epoch,
                                                                              score))

        if verbose:
            print("BayOpt hparams search iter = {}: params = {}".format(count.i, alg.stats.current_param))

        # get the score of this hyperparameter set
        score = alg.stats.current_param.score

        # avoid nan scores in search.
        if str(score) == "nan":
            score = -1e5

        # move current hparams to records
        alg.stats.update_records()
        alg.stats.to_csv(alg.results_file + '_' + alg.alg_args.type())

        # we want to maximize the score so negate it to invert minimization by skopt
        return -score

    return objective


def parse_config(params_config):
    bopt_space = _transform_hparams_dict(params_config)
    decorator = use_named_args(bopt_space)
    return decorator


class Count(object):
    def __init__(self):
        self.i = -1

    def inc(self):
        self.i += 1


class BayesianOptSearch(ParamSearchAlg):

    def __init__(self, *args, **kwargs):
        super(BayesianOptSearch, self).__init__(*args, **kwargs)
        assert (isinstance(self.alg_args, SkoptArgs))
        self.minimizer_func = {"gp": gp_minimize,
                               "gbrt": gbrt_minimize,
                               "rf": forest_minimize}.get(self.alg_args.type().lower(), gp_minimize)
        self.results = None

    def fit(self, model_dir, model_name, verbose=True):
        space = _transform_hparams_dict(self.config)
        print("BayOpt space dimension=%d" % len(space))
        self.results = self.minimizer_func(func=_create_objective(self, model_dir, model_name, verbose),
                                           dimensions=space, **self.alg_args.args)

        print("Best score={:.4f}".format(self.results.fun))

        return self.stats


class SearchArg:
    """Base class for organizing arguments for creating hyperparameter search algorithm objects.

    Paramters
    ---------
    :param n_calls: int
        Number of calls to func / maximum number of search iterations.
    """

    def __init__(self, n_calls):
        self.n_calls = n_calls


class SkoptArgs(SearchArg):
    """Base class for defining parameters of all Scikit-Optimize surrogate models."""

    def __init__(self, base_estimator, n_random_starts=10, n_calls=100, random_state=None, x0=None, y0=None,
                 verbose=False, callback=None, n_points=10000, acq_func='EI', xi=0.01,
                 kappa=1.96, n_jobs=1):
        super(SkoptArgs, self).__init__(n_calls=n_calls)
        self.base_estimator = base_estimator
        self.n_random_starts = n_random_starts
        self.x0 = x0
        self.y0 = y0
        self.verbose = verbose
        self.callback = callback
        self.n_points = n_points
        self.random_state = random_state
        self.acq_func = acq_func
        self.xi = xi
        self.kappa = kappa
        self.n_jobs = n_jobs

    @property
    def args(self):
        return self.__dict__

    @abc.abstractmethod
    def type(self) -> str:
        pass


class ForestMinArgs(SkoptArgs):
    """forest_minimize parameters."""

    def __init__(self, base_estimator='ET', n_calls=100, n_random_starts=10, acq_func='EI', x0=None,
                 y0=None, random_state=None, verbose=False, callback=None, n_points=10000, xi=0.01, kappa=1.96,
                 n_jobs=1, model_queue_size=None):
        super(ForestMinArgs, self).__init__(base_estimator=base_estimator,
                                            n_random_starts=n_random_starts,
                                            x0=x0,
                                            y0=y0,
                                            random_state=random_state,
                                            n_calls=n_calls,
                                            acq_func=acq_func,
                                            verbose=verbose,
                                            callback=callback,
                                            n_points=n_points,
                                            xi=xi,
                                            kappa=kappa,
                                            n_jobs=n_jobs)
        self.model_queue_size = model_queue_size

    def type(self):
        return 'rf'


class GBRTMinArgs(SkoptArgs):
    """gbrt_minimize parameters."""

    def __init__(self, base_estimator=None, n_calls=100, n_random_starts=10, acq_func='EI', x0=None, y0=None,
                 random_state=None, verbose=False, callback=None, n_points=10000, xi=0.01, kappa=1.96, n_jobs=1,
                 model_queue_size=None):
        super(GBRTMinArgs, self).__init__(base_estimator=base_estimator,
                                          n_calls=n_calls,
                                          n_random_starts=n_random_starts,
                                          acq_func=acq_func,
                                          x0=x0,
                                          y0=y0,
                                          random_state=random_state,
                                          verbose=verbose,
                                          callback=callback,
                                          n_points=n_points,
                                          xi=xi,
                                          kappa=kappa,
                                          n_jobs=n_jobs)
        self.model_queue_size = model_queue_size

    def type(self):
        return 'gbrt'


class GPMinArgs(SkoptArgs):
    """gp_minimize parameters."""

    def __init__(self, base_estimator=None, n_calls=100, n_random_starts=10, acq_func='gp_hedge',
                 acq_optimizer='auto', x0=None, y0=None, random_state=None, verbose=False, callback=None,
                 n_points=10000, n_restarts_optimizer=5, xi=0.01, kappa=1.96, noise='gaussian', n_jobs=1,
                 model_queue_size=None):
        super(GPMinArgs, self).__init__(base_estimator=base_estimator,
                                        n_calls=n_calls,
                                        n_random_starts=n_random_starts,
                                        acq_func=acq_func,
                                        x0=x0,
                                        y0=y0,
                                        random_state=random_state,
                                        verbose=verbose,
                                        callback=callback,
                                        n_points=n_points,
                                        xi=xi,
                                        kappa=kappa,
                                        n_jobs=n_jobs)
        self.noise = noise
        self.acq_optimizer = acq_optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.model_queue_size = model_queue_size

    def type(self):
        return 'gp'
