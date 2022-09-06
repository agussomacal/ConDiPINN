# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 16:44:59 2021

@author: rober
"""
from pathlib import Path
from typing import Callable, List

import scipy.sparse.linalg
from scipy import sparse

from src.config import results_path
from experiments.BaseExperiment import BaseExperiment
from lib.IntelligentModels.NNFlow import NNFlow

import matplotlib.pyplot as plt
import numpy as np

from experiments.utils import plot_predictions, Potencial, g, n, timeit, fig_save_context
from experiments.Models.parameters import EPSILON, alpha, k, potential, n_train_r, \
    r_weight_proportion, domain_to_predict, num_repetitions, \
    exact_solution_function

from lib.IntelligentModels.BaseModelFlow import BaseModelFlow
from lib.utils import Bounds


class FEM(BaseExperiment):
    def __init__(self, epsilon, potential: Potencial, g: Callable, alpha: float, k: float, n: Callable,
                 solve_sparse=True):
        self.epsilon = epsilon
        self.potential = potential
        self.g = g
        self.alpha = alpha
        self.k = k
        self.n = n
        self.solve_sparse = solve_sparse

    def create_matrix(self, num_points):
        if self.solve_sparse:
            # sparse.lil_matrix
            return sparse.dia_matrix((num_points, num_points))
            # return sparse.lil_matrix((num_points, num_points))
            # return sparse.csr_matrix((num_points, num_points))
        else:
            return np.zeros((num_points, num_points))

    def stiffness_matrix(self, num_points: int, h: float):
        """Stiffness matrix for P1 FEM functions
           K[i,j] = \int φ'_i φ'_j
           :param
            num_points: number of discretization points
            h: discretization
        """
        K = np.array([
            -1. / h * np.ones(num_points),
            np.hstack((1. / h, 2. / h * np.ones(num_points - 2), 1. / h)),
            -1. / h * np.ones(num_points)
        ])
        return K

    def advection_matrix(self, num_points: int):
        """A[i,j] = \int φ'_j φ_i TRANSPOSE
        """
        A = np.array([
            -1. / 2 * np.ones(num_points),
            np.hstack((-1. / 2, np.zeros(num_points - 2), 1. / 2)),
            1. / 2 * np.ones(num_points)
        ])
        return A

    def robin_matrix(self, num_points: int):
        R = np.array([
            np.zeros(num_points),
            np.hstack(
                (self.epsilon * self.k / self.alpha, np.zeros(num_points - 2), self.epsilon * self.k / self.alpha)),
            np.zeros(num_points)
        ])
        return R

    def assemble_rhs(self, num_points: int, h: float, x_bounds: Bounds, f=1):
        # \int_Ω φ_i
        integral_hat_fun = h * np.ones(num_points)
        integral_hat_fun[0] = h / 2
        integral_hat_fun[-1] = h / 2

        # \int_\partialΩ φ_i = φ_i(1)-φ_i(0)
        boundary_vec = np.zeros(num_points)
        boundary_vec[0] = self.epsilon * self.g(x_bounds.lower) / self.alpha
        boundary_vec[-1] = self.epsilon * self.g(x_bounds.upper) / self.alpha
        rhs = f * integral_hat_fun + boundary_vec
        return rhs

    @staticmethod
    def predict(u_at_nodes: np.ndarray, x_coords: np.ndarray, h: float):
        ix = np.array(x_coords // h, dtype=int)
        ix[ix >= len(u_at_nodes)] = len(u_at_nodes) - 1
        ixp1 = ix + 1
        ixp1[ixp1 >= len(u_at_nodes)] = len(u_at_nodes) - 1
        return (u_at_nodes[ixp1] - u_at_nodes[ix]) * (x_coords / h - ix) + u_at_nodes[ix]

    def experiment(self, n_samplings: int, n_train_r: int, r_weight_proportion: float,
                   max_samplings: int, n_iters_per_sampling: int,
                   coords2predict: np.ndarray, x_bounds: Bounds,
                   intelligent_model: Callable[[], BaseModelFlow], **kwargs) -> List[np.ndarray]:
        # there are n_train_r collocation points but n_train_r - 1 internal spaces dx
        h = (x_bounds.upper - x_bounds.lower) / (n_train_r - 1)

        # Assemble system matrix
        K = self.stiffness_matrix(n_train_r, h)
        A = self.advection_matrix(n_train_r)
        R = self.robin_matrix(n_train_r)

        # Full matrix
        F = 1.0
        B = sparse.dia_matrix((self.epsilon * K + F * A + R, (-1, 0, 1)), shape=(n_train_r, n_train_r))

        # Assemble rhs
        rhs = self.assemble_rhs(n_train_r, h, x_bounds, f=1)

        if self.solve_sparse:
            u_at_nodes = scipy.linalg.solve_banded((1, 1), B.data, rhs)[::-1]
        else:
            u_at_nodes = np.linalg.solve(B.todense(), rhs)
        return [self.predict(u_at_nodes, coords2predict, h).reshape((len(coords2predict)))]


if __name__ == "__main__":
    import time
    import pandas as pd

    experiment_path = Path.joinpath(results_path, 'FEM')
    experiment_path.mkdir(parents=True, exist_ok=True)

    times = pd.DataFrame(0, columns=[True, False], index=np.logspace(1, 4, num=10, dtype=int))
    for solve_sparse in times.columns:
        for n_train_r in times.index:
            ve = FEM(epsilon=EPSILON, potential=potential, g=g, n=n, k=k, alpha=alpha, solve_sparse=solve_sparse)

            with fig_save_context(
                    fig_path="{}/ApproxVSTrue_n{}_sparse{}.png".format(experiment_path, n_train_r, solve_sparse)) as ax:
                t0 = time.time()
                for i in range(num_repetitions):
                    with timeit("FEM timing sparse {}".format(ve.solve_sparse)):
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
                times.loc[n_train_r, solve_sparse] = time.time() - t0

                plot_predictions(ax, coords2predict=domain_to_predict,
                                 exact_solution_function=exact_solution_function,
                                 u_predictions_best=u_predictions[-1],
                                 alpha=0.5 + 0.5 / num_repetitions)
                ax.set_title("epsilon=" + format(ve.epsilon) + ",iteration=" + format(
                    len(u_predictions)) + ",lambda" + format(
                    1 - r_weight_proportion))

    times.rename(columns={True: "Sparse", False: "Dense"}, inplace=True)
    with fig_save_context(fig_path="{}/sparse_vs_dense.png".format(experiment_path)) as ax:
        times.plot(marker=".", ax=ax)
        plt.xscale("log")
        plt.yscale("log")
