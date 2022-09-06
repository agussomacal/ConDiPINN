"""
Experiment to explore depth and width of NN
"""
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import results_path
from experiments.Models.VanillaPinn import VanillaPinn
from experiments.Models.parameters import potential, g, n, k, alpha, n_train_r, num_repetitions, \
    r_weight_proportion, domain_to_predict
from experiments.Proceedings.parameters import epsilons2try
from experiments.utils import fig_save_context, plot_predictions, robin_exact_solution
from lib.IntelligentModels.NNFlow import NNFlow
from lib.utils import Bounds, NamedPartial

experiment_path = Path.joinpath(results_path, __file__.split(".")[0].split("/")[-1])
experiment_path.mkdir(parents=True, exist_ok=True)

training_losses = dict()
for epsilon in tqdm(epsilons2try[::2]):
    with fig_save_context(
            fig_path="{}/ApproxVSTrue_n{}_{}.png".format(experiment_path, n_train_r, epsilon)) as ax:
        u_predictions = []
        ve = VanillaPinn(epsilon=epsilon, potential=potential, g=g, n=n, k=k,
                         alpha=alpha, loss_metric="l2")
        pred, pinn = ve.experiment(
            n_samplings=r_weight_proportion,
            n_train_r=n_train_r,
            r_weight_proportion=r_weight_proportion,
            max_samplings=1,
            n_iters_per_sampling=10000,
            coords2predict=domain_to_predict,
            x_bounds=Bounds(lower=0, upper=1),
            intelligent_model=lambda: NNFlow(hidden_layers=(2, 2),
                                             limit_zero=False, activation="tanh"),  # True = NN*x(1-x)
            sampler=np.linspace,
            return_nn=True
        )
        training_losses[epsilon] = pinn.train_loss.copy()
        u_predictions += pred

        exact_solution_function = NamedPartial(robin_exact_solution, epsilon=epsilon, k=k, alpha=alpha)
        errors = np.mean((np.array(u_predictions) - exact_solution_function(domain_to_predict).T) ** 2, axis=1)
        best_ix = np.argmin(errors)
        plot_predictions(ax, coords2predict=domain_to_predict,
                         exact_solution_function=exact_solution_function,
                         u_predictions_best=u_predictions[best_ix],
                         alpha=0.5 + 0.5 / num_repetitions)
        ax.set_title(
            "epsilon=" + format(ve.epsilon) + ",iteration=" + format(len(u_predictions)) + ",lambda" + format(
                1 - r_weight_proportion))

with fig_save_context(fig_path="{}/train_loss_{}.png".format(experiment_path, n_train_r)) as ax:
    for eps, loss in training_losses.items():
        ax.plot(loss, "-", label=r"$\epsilon=$" + str(eps), alpha=0.7, linewidth=2)
    ax.legend()
    ax.set_yscale("log")
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("Training loss")
    ax.set_title("Comparison between training losses for different epsilons")
