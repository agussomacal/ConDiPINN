import os.path
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

from config import results_path
from experiments.Proceedings.parameters import alpha, k, model_names, epsilons2try, repetitions, data_experiment_name, \
    number_of_cores
from experiments.experiment_main import collect_data, do_computations, error_plots_flexible, H1, L2, \
    models_color_dict
from experiments.utils import fig_save_context

experiment_name = os.path.basename(__file__)[:-3]
print(experiment_name)
experiment_path = Path.joinpath(
    results_path,
    experiment_name
)
experiment_path.mkdir(parents=True, exist_ok=True)

# ----------------------- Iteration params ----------------------- #
# cartesian product of variables to use:
n_train = [10, 100, 1000, 10000]
sampler = ["linspace"]
float_precision = [32]
do_computations(data_experiment_name, model_names, epsilons2try, repetitions, n_train, sampler, float_precision, k=k,
                alpha=alpha, calculate=True, recalculate=False, number_of_cores=number_of_cores, test_factor=10,
                r_weight_proportion=0.5)
df, predictions, true_solutions = collect_data(data_experiment_name, k=k, alpha=alpha, n_train=n_train, sampler=sampler,
                                               float_precision=float_precision)

with fig_save_context("{}/ComparingNTrainTime.png".format(experiment_path), figsize=(8, 6)) as ax:
    filter_dict = {"epsilon": 10}
    sns.boxenplot(data=df[(df[list(filter_dict)] == pd.Series(filter_dict)).all(axis=1)], y="time", x="model_name",
                  hue="n_train", ax=ax)
    ax.set_yscale("log")

with fig_save_context("{}/ComparingNTrainError.png".format(experiment_path), figsize=(8, 6)) as ax:
    filter_dict = {"epsilon": 10}
    sns.boxenplot(data=df[(df[list(filter_dict)] == pd.Series(filter_dict)).all(axis=1)], y=H1, x="model_name",
                  hue="n_train", ax=ax)
    ax.set_yscale("log")

for model_name in pd.unique(df.model_name):
    with fig_save_context("{}/ComparingNTrain_{}.png".format(experiment_path, model_name), figsize=(8, 6)) as ax:
        error_plots_flexible(df, ax, error_name=H1, label_var="n_train", x_var="epsilon",
                             aggfunc=np.max, model_name=model_name)

for error_name in [H1, L2]:
    with fig_save_context(
            "{}/ComparingModelsBehaviour_{}.png".format(experiment_path, error_name.replace(" ", "_")),
            figsize=(8, 6)) as ax:
        error_plots_flexible(df, ax, error_name=error_name, label_var="model_name", x_var="epsilon",
                             stat_var="repetition", aggfunc=np.mean, color_dict=models_color_dict, n_train=100)

    for aggfunc in [np.min, np.median]:
        with fig_save_context(
                "{}/{}ComparingModelsBehaviour_{}.png".format(experiment_path, aggfunc.__name__,
                                                              error_name.replace(" ", "_")),
                figsize=(8, 6)) as ax:
            error_plots_flexible(df, ax, error_name=error_name, label_var="model_name", x_var="epsilon",
                                 aggfunc=aggfunc, color_dict=models_color_dict, n_train=100)
