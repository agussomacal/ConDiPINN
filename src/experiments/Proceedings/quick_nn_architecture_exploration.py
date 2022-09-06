"""
Experiment to explore depth and width of NN
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import results_path
from experiments.Models.VanillaPinn import VanillaPinn
from experiments.Models.parameters import EPSILON, potential, g, n, k, alpha, n_train_r, num_repetitions, \
    r_weight_proportion, domain_to_predict, exact_solution_function
from experiments.utils import fig_save_context, plot_predictions
from lib.IntelligentModels.NNFlow import NNFlow
from lib.utils import Bounds

ve = VanillaPinn(epsilon=EPSILON, potential=potential, g=g, n=n, k=k,
                 alpha=alpha, loss_metric="l2")

experiment_path = Path.joinpath(results_path, __file__.split(".")[0].split("/")[-1])
experiment_path.mkdir(parents=True, exist_ok=True)
for activation in ["tanh", "sigmoid"]:
    depth = [1, 2, 3, 4]
    width = [2, 10, 20, 50]
    times = pd.DataFrame(0, columns=width, index=depth)
    accuracy = pd.DataFrame(0, columns=width, index=depth)

    for d in tqdm(depth):
        for w in width:
            with fig_save_context(
                    fig_path="{}/ApproxVSTrue_n{}_w{}_d{}_{}.png".format(experiment_path, n_train_r, w, d,
                                                                         activation)) as ax:
                u_predictions = []
                t = []
                for i in range(num_repetitions):
                    t0 = time.time()
                    u_predictions += ve.experiment(
                        n_samplings=r_weight_proportion,
                        n_train_r=n_train_r,
                        r_weight_proportion=r_weight_proportion,
                        max_samplings=1,
                        n_iters_per_sampling=10000,
                        coords2predict=domain_to_predict,
                        x_bounds=Bounds(lower=0, upper=1),
                        intelligent_model=lambda: NNFlow(hidden_layers=(2, 2),
                                                         limit_zero=False, activation=activation),  # True = NN*x(1-x)
                        sampler=np.linspace
                    )
                    t.append(time.time() - t0)
                errors = np.mean((np.array(u_predictions) - exact_solution_function(domain_to_predict).T) ** 2, axis=1)
                best_ix = np.argmin(errors)
                accuracy.loc[d, w] = errors[best_ix]
                times.loc[d, w] = t[best_ix]
                plot_predictions(ax, coords2predict=domain_to_predict,
                                 exact_solution_function=exact_solution_function,
                                 u_predictions_best=u_predictions[best_ix],
                                 alpha=0.5 + 0.5 / num_repetitions)
                ax.set_title(
                    "epsilon=" + format(ve.epsilon) + ",iteration=" + format(len(u_predictions)) + ",lambda" + format(
                        1 - r_weight_proportion))

    with fig_save_context(
            fig_path="{}/Heatmap_accuracy_vs_time_connect_width_{}.png".format(experiment_path, activation)) as ax:
        for d in depth:
            ax.plot(times.loc[d, :], accuracy.loc[d, :], "-.", marker="o", label="Depth {}".format(d))
        ax.set_yscale("log")
        ax.legend()
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Accuracy l2")

    with fig_save_context(
            fig_path="{}/Heatmap_accuracy_vs_time_connect_depth_{}.png".format(experiment_path, activation)) as ax:
        for w in width:
            ax.plot(times.loc[:, w], accuracy.loc[:, w], "-.", marker="o", label="Width {}".format(w))
        ax.set_yscale("log")
        ax.legend()
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Accuracy l2")
