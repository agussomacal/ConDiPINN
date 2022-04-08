# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 16:44:59 2021

@author: rober
"""

from functools import partial
from pathlib import Path
from typing import Callable, List

from tqdm import tqdm

from src.config import results_path
from experiments.BaseExperiment import BaseExperiment
from experiments.Models.VanillaPinn import VanillaBoundaryOperator, VanillaOperator
from lib.IntelligentModels.NNFlow import NNFlow

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal

from experiments.utils import plot_predictions, Potencial, g, n
from experiments.Models.parameters import EPSILON, alpha, k, potential, n_train_r, \
    r_weight_proportion, domain_to_predict, num_repetitions, \
    exact_solution_function

from lib.DifferentialEquations.DifferentialEquation import Condition, DifferentialEquation
from lib.IntelligentModels.BaseModelFlow import BaseModelFlow
from lib.PINN_models.PinnFlow import PinnFlow
from lib.utils import Bounds


class VanillaCurriculumPinn(BaseExperiment):
    def __init__(self, num_epsilons: int, epsilon: float, potential: Potencial, g: Callable, alpha: float, k: float,
                 n: Callable, loss_metric="l2"):
        self.num_epsilons = num_epsilons
        self.loss_metric = loss_metric
        self.epsilon = epsilon
        self.potential = potential
        self.g = g
        self.alpha = alpha
        self.k = k
        self.n = n

    def experiment(self, n_samplings: int, n_train_r: int, r_weight_proportion: float,
                   max_samplings: int, n_iters_per_sampling: int,
                   coords2predict: np.ndarray, x_bounds: Bounds,
                   intelligent_model: Callable[[], BaseModelFlow], **kwargs) -> List[np.ndarray]:
        # --------- core experiment --------- #
        num_per_dim2pred = len(coords2predict)
        sampler = partial(np.random.uniform, x_bounds.lower, x_bounds.upper)
        sampler = partial(np.linspace, x_bounds.lower, x_bounds.upper)
        boundary_condition = Condition(
            operator=VanillaBoundaryOperator(g=self.g, alpha=self.alpha, k=self.k, n=self.n),
            function=lambda x: 0 * x,
            sampling_strategy=[
                ("x", partial(np.linspace, x_bounds.lower, x_bounds.upper))
            ],
            valid_sampling_strategy=[
                ("x", partial(np.linspace, x_bounds.lower, x_bounds.upper))
            ],
            n_train=2
        )

        for i, epsilon in tqdm(enumerate(np.logspace(np.log10(self.epsilon), np.log10(0.05), self.num_epsilons)[::-1]),
        # for i, epsilon in tqdm(enumerate(np.linspace(self.epsilon, 0.05, self.num_epsilons)[::-1]),

                               desc="Doing curriculum learning PINN"):
            residuals = Condition(
                operator=VanillaOperator(epsilon=epsilon, potential=self.potential),
                function=lambda x: 0 * x,
                sampling_strategy=[
                    ("x", sampler)
                ],
                valid_sampling_strategy=[
                    ("x", partial(np.random.uniform, x_bounds.lower, x_bounds.upper))
                ],
                n_train=n_train_r
            )

            differential_equation = DifferentialEquation(
                name="DifferentialEquation",
                domain_limits=[("x", x_bounds)],
                boundary_condition=boundary_condition,
                residuals=residuals
            )

            pinn = PinnFlow(
                model=intelligent_model() if i == 0 else pinn.model,
                differential_equation=differential_equation,
                loss_metric=self.loss_metric,
                n_iters_per_sampling=n_iters_per_sampling,
                max_samplings=max_samplings,
                weight_proportion={
                    "boundary_condition": 1 - r_weight_proportion,
                    "residuals": r_weight_proportion
                },
                initialize=True if i == 0 else False
            )

            pinn.fit()

        # --------- processing data to save experiment --------- #
        u_predictions = pinn.predict(domain=coords2predict, which="u").reshape((num_per_dim2pred))
        pinn.free_tf_session()
        del pinn
        # tf.get_default_graph().finalize()

        return [u_predictions]


if __name__ == "__main__":
    experiment_path = Path.joinpath(results_path, 'VanillaCurriculumPINN')
    experiment_path.mkdir(parents=True, exist_ok=True)

    ve = VanillaCurriculumPinn(num_epsilons=5, epsilon=EPSILON, potential=potential, g=g, n=n, k=k,
                     alpha=alpha, loss_metric="l2")

    fig, ax = plt.subplots()
    for i in range(num_repetitions):
        u_predictions = ve.experiment(
            n_samplings=r_weight_proportion,
            n_train_r=n_train_r,
            r_weight_proportion=r_weight_proportion,
            max_samplings=1,
            n_iters_per_sampling=10000,
            coords2predict=domain_to_predict,
            x_bounds=Bounds(lower=0, upper=1),
            intelligent_model=lambda: NNFlow(hidden_layers=(2, 2),
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
