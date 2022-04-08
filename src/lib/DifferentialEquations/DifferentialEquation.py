from functools import reduce
from operator import add
from typing import List, Callable, Dict, Tuple, Union

import numpy as np
import tensorflow as tf

from lib.utils import Bounds
from lib.DifferentialEquations.Operators import Operator, D

APPLY_OPERATOR_TO_U = None

FUNC_APPLICATION_VAR_NAME = "values"


class Condition:
    def __init__(self, operator: Operator, function: Callable,
                 sampling_strategy: List[Tuple[str, Union[Callable, List]]], n_train: int = None,
                 valid_sampling_strategy: List[Tuple[str, Callable]] = None,
                 apply_operator_to: List[int] = APPLY_OPERATOR_TO_U):

        self.apply_operator_to = np.array([0] * len(
            sampling_strategy) if apply_operator_to is APPLY_OPERATOR_TO_U else apply_operator_to)

        self.operator = operator
        self.function = function
        self.sampling_strategy = sampling_strategy
        self.valid_sampling_strategy = sampling_strategy if valid_sampling_strategy is None else valid_sampling_strategy
        self.n_train = n_train if n_train is not None else 0

    def generate_var_names(self, condition_name):
        for axis_name, _ in self.sampling_strategy:
            var_name = "{}_{}".format(condition_name, axis_name)
            yield var_name

        var_name = "{}_{}".format(condition_name, FUNC_APPLICATION_VAR_NAME)
        yield var_name

    def generate_values(self, train=True):
        domain_on_init_condition = []
        for axis_name, strategy in (self.sampling_strategy if train else self.valid_sampling_strategy):
            values = np.array(strategy(self.n_train) if isinstance(strategy, Callable) else strategy)
            domain_on_init_condition.append(values)
            yield values

        values = self.function(*domain_on_init_condition)
        yield values


class DifferentialEquation:
    def __init__(self, name, domain_limits: List[Tuple[str, Bounds]], **conditions: Condition):
        self.name = name

        self.domain_limits = domain_limits
        self.output_var_names = ["u"]

        self.conditions = conditions
        self.model_is = [0] * len(domain_limits)
        for condition in conditions.values():
            self.model_is = np.max((self.model_is, condition.apply_operator_to), axis=0)

    def __str__(self):
        return self.name

    def __getitem__(self, item):
        assert item in self.condition_names, "Key not found."
        return self.conditions[item]

    @staticmethod
    def get_tf_domain_and_values(condition_name: str, tf_dict):
        tf_true_values = []
        tf_domain = []
        for var_name, tf_placeholder in tf_dict[condition_name].items():
            if FUNC_APPLICATION_VAR_NAME in var_name:
                tf_true_values.append(tf_placeholder)
            else:
                tf_domain.append(tf_placeholder)
        return tf_domain, tf_true_values

    def get_condition_associated_tf_model(self, condition_name: str, nn, tf_dict: Dict):
        tf_domain, tf_true_values = self.get_tf_domain_and_values(condition_name, tf_dict)

        relative_derivatives = self.model_is - self.conditions[condition_name].apply_operator_to
        derive_respect_to = reduce(add, [[index] * num for index, num in enumerate(relative_derivatives)])

        # The neural network is the integral of the solution in the space domain.
        # in 1d: u = Dx NN
        # in 2d: u = Dxy NN
        # in 3d: u = Dxyz NN
        # but can be more general
        def nn_transformed(domain):
            return D(derive_respect_to=derive_respect_to)(nn, domain)
            # return tf.nn.relu(D(derive_respect_to=derive_respect_to)(nn, domain))
            # return 1-tf.nn.relu(1-tf.nn.relu(D(derive_respect_to=derive_respect_to)(nn, domain)))
            # return tf.sigmoid(D(derive_respect_to=derive_respect_to)(nn, domain)-2)

        return self.conditions[condition_name].operator(nn, tf_domain), tf.concat(tf_true_values, 1)

    @property
    def input_dim(self):
        return len(self.domain_limits)

    @property
    def output_dim(self):
        return 1

    @property
    def condition_names(self):
        return list(self.conditions.keys())
