# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 16:44:59 2021

@author: rober
"""
import pickle
import sys
import time

import numpy as np
import tensorflow as tf

from experiments.Models.CVPinn import CVPinn
from experiments.Models.FEM import FEM
from experiments.Models.RescalingVariationalPinn import RescalingVariationalPinn
from experiments.Models.VanillaCurriculumLearning import VanillaCurriculumPinn
from experiments.Models.VanillaPinn import VanillaPinn
from experiments.Models.VariationalPinn import VariationalPinn
from experiments.Models.VariationalPinn_minimim_on_u import VariationalPinnMinOnU
from experiments.utils import g, n, PlusX, get_prediction_domain
from lib.IntelligentModels.NNFlow import NNFlow
from lib.utils import Bounds, NamedPartial


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.get_logger().setLevel('INFO')


def experiment(experiment_type, repetition, epsilon, n_train_r, sampler, float_precision, k, alpha, g, n,
               r_weight_proportion=0.5, test_factor=10):
    tf.set_random_seed(repetition)
    np.random.seed(repetition)

    x_bounds = Bounds(lower=0, upper=1)

    t0 = time.time()
    e = experiment_type(epsilon=epsilon, potential=PlusX(), g=g, alpha=alpha, k=k, n=n)
    results = e.experiment(
        n_samplings=1,
        n_train_r=n_train_r,
        r_weight_proportion=r_weight_proportion,
        max_samplings=1,
        n_iters_per_sampling=1000000,
        coords2predict=get_prediction_domain([x_bounds], num_per_dim2pred=n_train_r * test_factor),
        x_bounds=x_bounds,
        intelligent_model=lambda: NNFlow(hidden_layers=(10, 10),
                                         limit_zero=False,
                                         float_precision=float_precision),  # True = NN*x(1-x)
        sampler=sampler
    )[0]
    t = time.time() - t0

    return results, t


if __name__ == "__main__":
    path, experiment_name, epsilon, repetition, n_train_r, sampler, float_precision, k, alpha, r_weight_proportion, \
    test_factor = sys.argv[1:]
    filename = "{}/{}_{}_{}_{}_{}_{}.pickle".format(path, experiment_name, epsilon, repetition, n_train_r, sampler,
                                                    float_precision)

    print("Doing experiment: ", filename)
    experiment_type = {
        "FEM": FEM,
        "VanillaPinn": VanillaPinn,
        "VanillaCurriculumPinn10": NamedPartial(VanillaCurriculumPinn, num_epsilons=10),
        "VanillaCurriculumPinn20": NamedPartial(VanillaCurriculumPinn, num_epsilons=20),
        "CVPinn": CVPinn,
        "VariationalPinn": VariationalPinn,
        "VariationalPinnExpTrue": NamedPartial(VariationalPinn, exp_integration=True),
        "VariationalPinnMinOnU": VariationalPinnMinOnU,
        "VariationalRescaling": RescalingVariationalPinn
    }[experiment_name]
    epsilon = float(epsilon)
    repetition = int(repetition)
    n_train_r = int(n_train_r)
    sampler = np.linspace if "linspace" in sampler.lower() else np.random.uniform
    # float_precision = getattr(tf, "float{}".format(float_precision))
    float_precision = {16: tf.float16, 32: tf.float32, 64: tf.float64}[int(float_precision)]
    pred, t_fit = experiment(
        experiment_type=experiment_type,
        repetition=repetition,
        epsilon=epsilon,
        n_train_r=n_train_r,
        sampler=sampler,
        float_precision=float_precision,
        k=float(k),
        alpha=float(alpha),
        g=g,
        n=n,
        r_weight_proportion=float(r_weight_proportion),
        test_factor=int(test_factor)
    )

    with open(filename, "wb") as f:
        pickle.dump({"pred": pred, "t": t_fit}, f)
