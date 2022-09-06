import numpy as np

from experiments.experiment_main import names4paper_dict

data_experiment_name = "Proceedings"

number_of_cores = 50

k = 1
alpha = 0.001

model_names = list(names4paper_dict.keys())

epsilons2try = np.round(np.logspace(np.log10(0.005), np.log10(0.5), num=10), decimals=3)
epsilons2try = np.append(epsilons2try, np.round(np.logspace(np.log10(0.5), np.log10(10), num=5), decimals=3))
epsilons2try = np.sort(np.unique(epsilons2try))
repetitions = list(range(10))
