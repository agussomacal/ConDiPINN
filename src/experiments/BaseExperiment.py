from typing import List, Callable

import numpy as np

from lib.DifferentialEquations.Operators import Operator
from lib.IntelligentModels.BaseModelFlow import BaseModelFlow
from lib.utils import Bounds


class BaseExperiment:
    def __str__(self):
        return str(self.__class__.__name__)

    def experiment(self, n_samplings: int, n_train_r: int, r_weight_proportion: float,
                   coords2predict: np.ndarray, x_bounds: Bounds,
                   intelligent_model: Callable[[], BaseModelFlow],
                   equation_operator: Operator, sampler=np.random.uniform, **kwargs) -> List[np.ndarray]:
        pass
