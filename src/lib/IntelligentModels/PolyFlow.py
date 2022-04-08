from collections import OrderedDict
from itertools import chain
from typing import Tuple, List, Union

import numpy as np
import tensorflow as tf

from src.lib.IntelligentModels.BaseModelFlow import BaseModelFlow, TF_DTYPE_TO_USE


def dfs_polynomial_order(input_dimension, max_polynomial_order, max_function=sum):
    """
    Given a derivative depth for each domain variable it calculates the graph of derivations that should be done
    in order to get all the partial derivatives.
    :param domain_depth_dict: ej: {"t": 2, "x":2} -> means up to order 2 in derivatives of t and x
    :return:
    """
    root_name = [0] * input_dimension
    queue = [root_name]

    for element in queue:
        if max_function(element) < max_polynomial_order:
            for i in range(input_dimension):
                new_element = element[:]
                new_element[i] += 1
                if new_element not in queue:
                    queue.append(new_element)
                    yield tuple(element), i, tuple(new_element)


def successive_multiplications(X: np.ndarray, polynomial_order, max_function=sum):
    input_dim = np.shape(X)[1]
    polynomial_vars_dict = OrderedDict([(tuple([0] * input_dim), 1)])
    for father_node, index_2_multiply, son_node in dfs_polynomial_order(input_dim, polynomial_order, max_function):
        polynomial_vars_dict[son_node] = polynomial_vars_dict[father_node] * X[:, index_2_multiply]

    return list(polynomial_vars_dict.values())


class PolyFeatures:
    def __init__(self, degree, total_dim_max_degree=True, basis=None):
        self.basis = basis
        self.degree = degree
        self.total_dim_max_degree = total_dim_max_degree
        self.__max_function = sum if self.total_dim_max_degree else max
        self.in_dim = None
        self.out_dim = 1
        self.n_output_features_ = None

    def fit(self, X, y=None):
        self.in_dim = np.shape(X)[1]
        X_out = successive_multiplications(
            X=np.zeros((1, self.in_dim)),
            polynomial_order=self.degree,
            max_function=self.__max_function
        )
        self.n_output_features_ = len(X_out)

    def transform(self, X):
        return successive_multiplications(X, polynomial_order=self.degree, max_function=self.__max_function)


class PolyFlow(BaseModelFlow):
    def __init__(self, degree, basis=None, total_dim_max_degree=True):
        super().__init__(name=f"Flow_poly_{degree}_{basis if basis is not None else ''}")
        self.poly_features = PolyFeatures(degree=degree, basis=basis, total_dim_max_degree=total_dim_max_degree)
        self.weights = None
        self.biases = None

    # ------------- network architecture and initialization ------------
    def initialize(self, input_dim, output_dim):
        self.poly_features.fit(np.zeros((1, input_dim)))
        self.weights = tf.Variable(tf.zeros([self.poly_features.n_output_features_ - 1, 1]), dtype=TF_DTYPE_TO_USE)
        self.biases = tf.Variable(tf.zeros([1, 1], dtype=TF_DTYPE_TO_USE), dtype=TF_DTYPE_TO_USE)

    @property
    def parameters(self):
        return self.weights, self.biases

    @parameters.setter
    def parameters(self, params):
        self.weights, self.biases = params

    def __call__(self, eq_diff_domain: List[tf.Tensor]):
        X = tf.concat(eq_diff_domain, 1)

        return tf.add(tf.matmul(self.poly_features.transform(X)[1:], self.weights), self.biases)
