# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 16:44:59 2021

@author: Agus

To do the experiments the code must be run in the terminal and the working directory must be placed in
.../src/experiments folder location. Otherwise the experiment_parallel_sequential.py is not found when calling it.
"""
import itertools
import os.path
import pickle
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathos.multiprocessing import Pool, cpu_count

from experiments.utils import robin_exact_solution, get_prediction_domain, fig_save_context
from lib.utils import Bounds, NamedPartial
from src.config import results_path, experiments_path, data_path

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ------------------------------------------------------ #
# -------------------- Constants ----------------------- #
# ------------------------------------------------------ #

L2 = "l2 error"
H1 = "H1 error"

CBLIND_BLUE = sns.color_palette("colorblind")[0]
CBLIND_ORANGE = sns.color_palette("colorblind")[1]
CBLIND_GREEN = sns.color_palette("colorblind")[2]
CBLIND_RED = sns.color_palette("colorblind")[3]
CBLIND_VIOLET = sns.color_palette("colorblind")[4]
CBLIND_BROWN = sns.color_palette("colorblind")[5]
CBLIND_PINK = sns.color_palette("colorblind")[6]
CBLIND_GRAY = sns.color_palette("colorblind")[7]
CBLIND_YELLOW = sns.color_palette("colorblind")[8]
CBLIND_CYAN = sns.color_palette("colorblind")[9]

names4paper_dict = {
    "FEM": "FEM",
    "VanillaPinn": "V",
    "CVPinn": "V-z",
    "VariationalPinn": "W-z",
    "VariationalPinnExpTrue": "W-z-e",
    # "VariationalPinnExactSampler": "blue",
    "VariationalPinnMinOnU": "W",
    # "VanillaCurriculumPinn10": "gray",
    # "VanillaCurriculumPinn20": "slategray",
    "VariationalRescaling": "RW-z"
}

models_color_dict = {
    "FEM": CBLIND_GREEN,
    "VanillaPinn": CBLIND_BLUE,
    "CVPinn": CBLIND_YELLOW,
    "VariationalPinn": CBLIND_GRAY,
    "VariationalPinnExpTrue": CBLIND_VIOLET,
    # "VariationalPinnExactSampler": "blue",
    "VariationalPinnMinOnU": CBLIND_RED,
    # "VanillaCurriculumPinn10": "gray",
    # "VanillaCurriculumPinn20": "slategray",
    "VariationalRescaling": CBLIND_BROWN
}
models_color_dict = {names4paper_dict[key]: val for key, val in models_color_dict.items()}


# ------------------------------------------------------------------ #
# -------------------- Computations and data ----------------------- #
# ------------------------------------------------------------------ #

def get_data_path(experiment_name):
    experiment_data_path = Path.joinpath(
        data_path,
        experiment_name
    )
    experiment_data_path.mkdir(parents=True, exist_ok=True)
    return experiment_data_path


def do_computations(experiment_name, model_names, epsilons2try, repetitions, n_train, samplers, float_precisions, k,
                    alpha, calculate=True, recalculate=False, number_of_cores=5, test_factor=10,
                    r_weight_proportion=0.5):
    # https://alexandra-zaharia.github.io/posts/run-python-script-as-subprocess-with-multiprocessing/
    experiment_data_path = get_data_path(experiment_name)
    number_of_cores = min((number_of_cores, cpu_count() - 1))

    # parallelize computations
    if calculate:
        def filter_func(args):
            filename = "{}/{}_{}_{}_{}_{}_{}.pickle".format(experiment_data_path, *args)
            return (not os.path.exists(filename)) or (os.path.getsize(filename) == 0)

        # ----------------------- parallelize ----------------------- #
        def par_func(args):
            return subprocess.call(
                ["python",
                 "experiment_parallel_sequential.py",
                 ] + [str(experiment_data_path)] + list(map(str, args)) + [str(k), str(alpha), str(r_weight_proportion),
                                                                           str(test_factor)],
                shell=False,
            )

        parameters2calculate = list(
            itertools.product(model_names, epsilons2try, repetitions, n_train, samplers, float_precisions))

        if not recalculate:  # filter already done experiments.
            parameters2calculate = list(filter(filter_func, parameters2calculate))
        # progress_bar = tqdm(total=len(parameters2calculate))
        print("\n\n\n\n\n\n", "All the calculations are: ", len(parameters2calculate), "\n\n\n\n\n\n")

        # def update_progress_bar(_):
        #     progress_bar.update()

        map_func = Pool(number_of_cores).map if number_of_cores > 1 else map
        os.chdir(experiments_path)  # put path in experiments folder
        for res in map_func(par_func, parameters2calculate):
            pass
        # for params in parameters2calculate:
        #     p.apply_async(par_func, (params,), callback=update_progress_bar)
        # print(params)


def collect_data(experiment_name, k, alpha, **kwargs):
    experiment_data_path = get_data_path(experiment_name)

    # ----------------------- Collect data ----------------------- #
    results_dict = defaultdict(list)
    predictions = defaultdict(dict)
    i = 0
    for filename in os.listdir(experiment_data_path):
        try:
            with open("{}/{}".format(experiment_data_path, filename), "rb") as f:
                results_i = pickle.load(f)
            metadata = filename[:-7].split("_")
            ntr = int(metadata[3])
            predictions[ntr][i] = results_i["pred"]
            i += 1
            results_dict["time"].append(results_i["t"])
            results_dict["model_name"].append(names4paper_dict[metadata[0]])
            results_dict["epsilon"].append(float(metadata[1]))
            results_dict["repetition"].append(int(metadata[2]))
            results_dict["n_train"].append(ntr)
            results_dict["sampler"].append(metadata[4])
            results_dict["float_precision"].append(int(metadata[5]))
        except (FileNotFoundError, FileExistsError):
            print("File open problem: {}.".format(filename))
    predictions = {ntr: pd.DataFrame.from_dict(preds) for ntr, preds in predictions.items()}

    df = pd.DataFrame.from_dict(results_dict)

    # filter for experiment specific variables
    for key, val in kwargs.items():
        df = df[df[key].isin(val)]
    for ntr, sub_df in df.groupby(["n_train"]):
        predictions[ntr] = predictions[ntr].loc[:, predictions[ntr].columns.isin(sub_df.index)].T.sort_index()
        predictions[ntr] = predictions[ntr].values  # to numpy array

    # calculate errors
    exact_solution_function = NamedPartial(robin_exact_solution, k=k, alpha=alpha)
    true_solutions = dict()
    df[L2] = 0
    df[H1] = 0
    for ntr, sub_df in df.groupby(["n_train"]):
        preds = predictions[ntr]
        num_per_dim2pred = np.shape(preds)[1]
        eps = np.sort(np.unique(sub_df.epsilon))

        true_solutions[ntr] = np.reshape(
            [exact_solution_function(
                domain=get_prediction_domain([Bounds(lower=0, upper=1)], num_per_dim2pred=num_per_dim2pred),
                epsilon=epsilon) for epsilon in eps],
            (-1, 1, num_per_dim2pred))

        for _, data in sub_df.reset_index(drop=True).groupby(["model_name", "float_precision", "sampler"]):
            # squared error, axis=2=space dimension
            sorted_index = data.sort_values(by=["epsilon", "repetition"]).index

            df.loc[sub_df.index[sorted_index], L2] = np.sqrt(np.mean(
                (np.reshape(preds[sorted_index], (len(eps), -1, num_per_dim2pred)) - true_solutions[ntr]) ** 2,
                axis=2).ravel())

            df.loc[sub_df.index[sorted_index], H1] = np.sqrt(np.mean(np.gradient(
                np.reshape(preds[sorted_index], (len(eps), -1, num_per_dim2pred)) - true_solutions[ntr], axis=2) ** 2,
                                                                     axis=2).ravel())

    return df, predictions, true_solutions


# ------------------------------------------------------- #
# -------------------- Plot utils ----------------------- #
# ------------------------------------------------------- #

def data_filter(df, label_var, color_dict=None, **filter_dict):
    df_metadata2plot = df.loc[(df[list(filter_dict)] == pd.Series(filter_dict)).all(axis=1)]
    if color_dict is None:
        palette = np.array(sns.color_palette("colorblind"))
        color_dict = {lab: palette[i] for i, lab in enumerate(np.unique(df_metadata2plot[label_var]))}
    for label, data in df_metadata2plot.groupby(label_var):
        yield label, data, color_dict


def error_plots_flexible(df, ax, error_name, label_var, x_var, stat_var=None, color_dict=None, xlog=True, ylog=True,
                         aggfunc=np.mean, **kwargs):
    for label, data, color_dict in data_filter(df, label_var, color_dict, **kwargs):
        error = pd.pivot_table(data, index=x_var, columns=stat_var, values=error_name, aggfunc=aggfunc).sort_index()
        perc_25 = np.quantile(error, q=0.25, axis=1)
        error_mean = np.quantile(error, q=0.5, axis=1)
        perc_75 = np.quantile(error, q=0.75, axis=1)
        error_min = np.min(error, axis=1)
        ax.plot(error.index, error_mean, marker=".", linestyle='--',
                label="{}: {} ".format(label_var, label) + r" $\pm25\%$",
                c=color_dict[label])
        ax.fill_between(error.index, perc_25, perc_75, color=color_dict[label],
                        alpha=0.1)
        ax.plot(error.index, error_min, marker="o", linestyle='-', c=color_dict[label])

    ax.set_xlabel(x_var)
    ax.set_ylabel(r"$\ell_2$ error") if error_name == L2 else ax.set_ylabel(error_name)
    ax.legend()
    ax.set_title(
        "PINN models error behaviour\n" + " ".join(["{}={}".format(name, val) for name, val in kwargs.items()]))
    if xlog:
        ax.set_xscale("log")
        eps2tick = np.unique(
            np.round(
                np.logspace(np.log10(np.min(data[x_var])), np.log10(np.max(data[x_var])), 10),
                decimals=3))
        ax.set_xticks(eps2tick)
        ax.set_xticklabels(eps2tick)

    if ylog:
        ax.set_yscale("log")
        ax.set_ylim((0, 1))
    # plt.tight_layout()


# prediction plots
def prediction_plot(df, predictions, true_solutions, ax, error_name, n_train, float_precision, sampler, epsilon,
                    test_factor):
    df_metadata2plot = df[df.n_train == n_train].reset_index(drop=True)
    df_metadata2plot = df_metadata2plot.loc[
                       (df_metadata2plot.n_train == n_train) & (df_metadata2plot.float_precision == float_precision) & (
                               df_metadata2plot.sampler == sampler) & (df_metadata2plot.epsilon == epsilon), :]
    df_metadata2plot = df_metadata2plot.groupby("model_name").apply(lambda x: x.index[np.argmin(x[error_name])])

    x2pred = get_prediction_domain([Bounds(0, 1)], num_per_dim2pred=n_train * test_factor)
    true_sol = np.ravel(true_solutions[n_train][np.where(np.sort(np.unique(df.epsilon)) == epsilon)[0]])
    ax.plot(x2pred, true_sol, label="True solution", c="teal", linestyle="-", linewidth=4, alpha=0.7)
    ax.set_ylim((np.min(true_sol), np.max(true_sol)))
    for ix, model_name in zip(df_metadata2plot.values, df_metadata2plot.index):
        ax.plot(x2pred, predictions[n_train][ix].T, label=model_name,
                c=models_color_dict[model_name],
                linestyle="-.", alpha=0.8, linewidth=2)

    ax.legend()
    ax.set_title(r"PINN models prediction"  # for $\epsilon$=" + str(epsilon) #+ "\n" +
                 # "alpha=" + str(alpha) + " k=" + str(k) + " n train=" + str(n_train) + "\n" +
                 # "precision=float" + str(float_precision) + " sampler=" + str(sampler)
                 )
    ax.set_xlabel("x")
    ax.set_ylabel("u")


def time_plots(df, ax, error_name=L2, **kwargs):
    for label, data, color_dict in data_filter(df, "model_name", **kwargs):
        ax.scatter(data.time, data[error_name], marker=".",
                   label=label, c=color_dict[label])
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"$\ell_2$ error") if error_name == L2 else ax.set_ylabel(error_name)
    ax.legend()
    ax.set_title("PINN models time comparison")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim((0, 1))
    # plt.tight_layout()
