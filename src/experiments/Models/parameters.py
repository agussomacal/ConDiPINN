# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 10:36:44 2022

@author: rober
"""
import numpy as np
import tensorflow as tf

from experiments.utils import get_prediction_domain, plot_predictions, \
    Potencial, PlusX, MinusX, g, n, robin_exact_solution , robin_exact_solution_cv
from lib.utils import Bounds, NamedPartial

tf.set_random_seed(42)
np.random.seed(42)

k = 1
alpha = 0.001
EPSILON = 10
potential = PlusX()

n_train_r = 100
r_weight_proportion = 0.5

x_bounds = Bounds(lower=0, upper=1) 


num_per_dim2pred = n_train_r * 10
domain_to_predict = get_prediction_domain([x_bounds], num_per_dim2pred=num_per_dim2pred)

num_repetitions = 1

x_bounds_cv = Bounds(lower=0, upper=1/EPSILON) 
domain_to_predict_cv = get_prediction_domain([x_bounds_cv], num_per_dim2pred=num_per_dim2pred)

exact_solution_function = NamedPartial(robin_exact_solution, epsilon=EPSILON,
                                       k=k, alpha=alpha)
exact_solution_function_cv = NamedPartial(robin_exact_solution_cv, epsilon=EPSILON,
                                       k=k, alpha=alpha)
