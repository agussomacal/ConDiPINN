# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 16:44:59 2021

@author: rober
"""

from functools import partial
from pathlib import Path
from typing import Callable, List

from src.config import results_path
from experiments.BaseExperiment import BaseExperiment
from lib.IntelligentModels.NNFlow import NNFlow

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal
import tensorflow as tf

from experiments.utils import plot_predictions, Potencial, g, n
from experiments.Models.parameters import EPSILON, alpha, k, potential, n_train_r, \
    r_weight_proportion, domain_to_predict, num_repetitions, \
    exact_solution_function

from lib.DifferentialEquations.DifferentialEquation import Condition, DifferentialEquation
from lib.IntelligentModels.BaseModelFlow import BaseModelFlow
from lib.PINN_models.PinnFlow import PinnFlow
from lib.DifferentialEquations.Operators import D, Operator
from lib.utils import Bounds


class VariationalOperator(Operator):
    def __init__(self, epsilon, potential: Potencial):
        self.epsilon = epsilon
        self.potential = potential
        super(VariationalOperator, self).__init__(name="VariationalOperator")

    def call_method(self, u: Callable, domain: List[tf.Tensor]):
        assert len(domain) == 1, "This operator only works with 1d problems."
        return 0.5 * (D(derive_respect_to=[0])(u, domain) ** 2 \
                      + u(domain) ** 2 * (
                              self.potential.lap_eval(domain) / 2 / self.epsilon +
                              self.potential.grad_eval(domain) ** 2 / 4 / (self.epsilon ** 2))
                      ) \
               - u(domain) / self.epsilon * tf.exp(-self.potential.eval(domain) / 2 / self.epsilon)


class VariationalOperatorTerm1(Operator):
    def __init__(self, epsilon, potential: Potencial):
        self.epsilon = epsilon
        self.potential = potential
        super(VariationalOperatorTerm1, self).__init__(name="VariationalOperatorTerm1")

    def call_method(self, u: Callable, domain: List[tf.Tensor]):
        assert len(domain) == 1, "This operator only works with 1d problems."
        return 0.5 * D(derive_respect_to=[0])(u, domain) ** 2 \
               + 0.5 * u(domain) ** 2 * (
                       self.potential.lap_eval(domain) / 2 / self.epsilon +
                       self.potential.grad_eval(domain) ** 2 / 4 / self.epsilon ** 2)


class VariationalOperatorTerm2(Operator):
    def __init__(self, epsilon, potential: Potencial):
        self.epsilon = epsilon
        self.potential = potential
        super(VariationalOperatorTerm2, self).__init__(name="VariationalOperatorTerm2")

    def call_method(self, u: Callable, domain: List[tf.Tensor]):
        assert len(domain) == 1, "This operator only works with 1d problems."
        return - u(domain) \
            # / self.epsilon * tf.exp(-self.potential.eval(domain) / 2 / self.epsilon)


# class VariationalOperator(Operator):
#     def __init__(self, epsilon, potential: Potencial):
#         self.epsilon = epsilon
#         self.potential = potential
#         super(VariationalOperator, self).__init__(name="VariationalOperator")
#
#     def call_method(self, u: Callable, d: List[tf.Tensor]):
#         # assert len(domain) == 1, "This operator only works with 1d problems."
#         def func(x):
#             domain = [tf.constant([[x]], dtype=float)]
#             return D(derive_respect_to=[0])(u, domain) ** 2 \
#                    + u(domain) ** 2 * (
#                            self.potential.lap_eval(domain) / 2 / self.epsilon +
#                            self.potential.grad_eval(domain) ** 2 / 4 / self.epsilon ** 2) \
#                    - u(domain) / self.epsilon * tf.exp(-self.potential.eval(domain) / 2 / self.epsilon)
#
#         return quad(func, 0, 1)


class VariationalBoundaryOperator(Operator):
    def __init__(self, epsilon, potential: Potencial, g: Callable, alpha: float, k: float, n: Callable):
        self.epsilon = epsilon
        self.potential = potential
        self.g = g
        self.alpha = alpha
        self.k = k
        self.n = n
        super(VariationalBoundaryOperator, self).__init__(name="VariationalBoundaryOperator")

    def call_method(self, u: Callable, domain: List[tf.Tensor]):
        assert len(domain) == 1, "This operator only works with 1d problems."
        return (self.k / self.alpha + 1 / 2 / self.epsilon * self.potential.grad_eval(domain) * self.n(domain)) * u(
            domain) ** 2 - 2 * \
               tf.exp(-self.potential.eval(domain) / 2 / self.epsilon) / self.alpha * self.g(domain) * u(domain)


class VariationalPinn(BaseExperiment):
    def __init__(self, epsilon, potential: Potencial, g: Callable, alpha: float, k: float, n: Callable,
                 loss_metric="id", exp_integration=False):
        self.loss_metric = loss_metric
        self.epsilon = epsilon
        self.potential = potential
        self.g = g
        self.alpha = alpha
        self.k = k
        self.n = n
        self.exp_integration = exp_integration

    def __str__(self):
        return super(VariationalPinn, self).__str__() + ("_ExpTerm" if self.exp_integration else "")

    def experiment(self, n_samplings: int, n_train_r: int, r_weight_proportion: float,
                   max_samplings: int, n_iters_per_sampling: int,
                   coords2predict: np.ndarray, x_bounds: Bounds,
                   intelligent_model: Callable[[], BaseModelFlow], sampler=np.random.uniform, **kwargs) -> List[np.ndarray]:

        # --------- core experiment --------- #
        for i in range(n_samplings):
            weight_proportion = {
                "boundary_condition": 1 - r_weight_proportion,
                "residuals": r_weight_proportion
            }

            conditions = dict()
            conditions["boundary_condition"] = Condition(
                operator=VariationalBoundaryOperator(epsilon=self.epsilon, potential=self.potential, g=self.g,
                                                     alpha=self.alpha, k=self.k, n=self.n),
                function=lambda x: 0 * x,
                sampling_strategy=[
                    ("x", partial(np.linspace, x_bounds.lower, x_bounds.upper))
                ],
                valid_sampling_strategy=[
                    ("x", partial(np.linspace, x_bounds.lower, x_bounds.upper))
                ],
                n_train=2
            )

            if self.exp_integration:
                # def exp_sampler(n):
                #     x = np.random.exponential(scale=1 / 2 / self.epsilon, size=n)
                #     x = x / np.max(x)
                #     return 1 - x

                # def exp_sampler(n):
                #     x = np.random.uniform(size=n)
                #     return -np.log(1 - x * (1 - np.exp(-1 / 2 / self.epsilon))) * 2 * self.epsilon

                #
                # def sampler(n):
                #     x = np.random.uniform(x_bounds.lower, x_bounds.upper, n)
                #     return -2 * self.epsilon * np.log(1 - x * (1 - np.exp(-1 / 2 / self.epsilon)))
                def exp_sampler(n):
                    x = np.random.uniform(size=n)
                    return -np.log(1 - x * (1 - np.exp(-1 / 2 / self.epsilon))) * 2 * self.epsilon

                conditions["residuals"] = Condition(
                    operator=VariationalOperatorTerm1(epsilon=self.epsilon, potential=self.potential),
                    function=lambda x: 0 * x,
                    sampling_strategy=[
                        ("x", partial(sampler, x_bounds.lower, x_bounds.upper))
                    ],
                    valid_sampling_strategy=[
                        ("x", partial(np.random.uniform, x_bounds.lower, x_bounds.upper))
                    ],
                    n_train=n_train_r
                )

                conditions["residuals2"] = Condition(
                    operator=VariationalOperatorTerm2(epsilon=self.epsilon, potential=self.potential),
                    function=lambda x: 0 * x,
                    sampling_strategy=[
                        ("x", exp_sampler)
                    ],
                    valid_sampling_strategy=[
                        ("x", partial(np.random.uniform, x_bounds.lower, x_bounds.upper))
                    ],
                    n_train=n_train_r
                )

                weight_proportion["residuals2"] = r_weight_proportion * 2 * (1 - np.exp(-1 / 2 / self.epsilon))

            else:

                conditions["residuals"] = Condition(
                    operator=VariationalOperator(epsilon=self.epsilon, potential=self.potential),
                    function=lambda x: 0 * x,
                    sampling_strategy=[
                        ("x", partial(sampler, x_bounds.lower, x_bounds.upper))
                    ],
                    valid_sampling_strategy=[
                        ("x", partial(np.random.uniform, x_bounds.lower, x_bounds.upper))
                    ],
                    n_train=n_train_r
                )

            differential_equation = DifferentialEquation(
                name="DifferentialEquation",
                domain_limits=[("x", x_bounds)],
                **conditions
            )

            pinn = PinnFlow(
                model=intelligent_model(),
                differential_equation=differential_equation,
                loss_metric=self.loss_metric,
                n_iters_per_sampling=n_iters_per_sampling,
                max_samplings=max_samplings,
                weight_proportion=weight_proportion,
                initialize=True
            )

            pinn.fit()

            # --------- processing data to save experiment --------- #
            num_per_dim2pred = len(coords2predict)
            u_predictions = pinn.predict(domain=coords2predict, which="u").reshape((num_per_dim2pred)) * np.exp(
                self.potential.eval([coords2predict]) / 2 / self.epsilon).reshape((num_per_dim2pred))
            pinn.free_tf_session()
            del pinn
            # tf.get_default_graph().finalize()

            return [u_predictions]


if __name__ == "__main__":
    experiment_path = Path.joinpath(results_path, 'VariationalPINN')
    experiment_path.mkdir(parents=True, exist_ok=True)

    ve = VariationalPinn(epsilon=EPSILON, potential=potential, g=g, n=n, k=k,
                         alpha=alpha, loss_metric="id", exp_integration=False)

    fig, ax = plt.subplots()
    for i in range(num_repetitions):
        u_predictions = ve.experiment(
            n_samplings=1,
            n_train_r=n_train_r,
            r_weight_proportion=r_weight_proportion,
            max_samplings=1,
            n_iters_per_sampling=1000000000,
            loss_metric="id",
            coords2predict=domain_to_predict,
            x_bounds=Bounds(lower=0, upper=1),
            intelligent_model=lambda: NNFlow(hidden_layers=(10, 10),
                                             limit_zero=False),  # True = NN*x(1-x)
        )

        plot_predictions(ax, coords2predict=domain_to_predict,
                         exact_solution_function=exact_solution_function,
                         u_predictions_best=u_predictions[-1],
                         alpha=0.5 + 0.5 / num_repetitions)
    title = "epsilon=" + format(ve.epsilon) + ",iteration=" + format(len(u_predictions)) + ",lambda" + format(
        1 - r_weight_proportion)
    plt.title(title)
    plt.savefig("{}/ApproxVSTrue.png".format(experiment_path))
    plt.show()
