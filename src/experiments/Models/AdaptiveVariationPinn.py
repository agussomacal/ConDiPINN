# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 11:13:00 2022

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
        return (self.k / self.alpha + 1 / 2 / self.epsilon * self.potential.grad_eval(domain) * self.n(domain))* u(domain)**2 -2* \
               tf.exp(-self.potential.eval(domain) / 2 / self.epsilon) / self.alpha * self.g(domain) * u(domain)

        
class VariationalOperator_post(Operator):
    def __init__(self, epsilon, potential: Potencial, previous_pinns: List[PinnFlow]):
        self.epsilon = epsilon
        self.potential = potential
        self.previous_pinns = previous_pinns 
        super(VariationalOperator_post, self).__init__(name="VariationalOperator_post")

    def call_method(self, u: Callable, domain: List[tf.Tensor]):
        assert len(domain) == 1, "This operator only works with 1d problems."
        x = 0.5* (D(derive_respect_to=[0])(u, domain)+sum([D(derive_respect_to=[0])(pinn.model,domain) for pinn in self.previous_pinns]))**2
        y =  0.5*(self.potential.lap_eval(domain) / 2 / self.epsilon +self.potential.grad_eval(domain) ** 2 / 4 / (self.epsilon ** 2))*(u(domain)+sum([pinn.model(domain) for pinn in self.previous_pinns]))**2
        z = (u(domain)+sum([pinn.model(domain) for pinn in self.previous_pinns])) / self.epsilon * tf.exp(-self.potential.eval(domain) / 2 / self.epsilon)
        return x+y-z
               #0.5* (D(derive_respect_to=[0])(u, domain)+sum([D(derive_respect_to=[0])(pinn.model,domain) for pinn in self.previous_pinns]))**2\
               #+ 0.5*(self.potential.lap_eval(domain) / 2 / self.epsilon +self.potential.grad_eval(domain) ** 2 / 4 / (self.epsilon ** 2))*(\
               #    u(domain)+sum([pinn.model(domain) for pinn in self.previous_pinns]))**2\
               #-(u(domain)+sum([pinn.model(domain) for pinn in self.previous_pinns])) / self.epsilon * tf.exp(-self.potential.eval(domain) / 2 / self.epsilon)
 
class VariationalBoundaryOperator_post(Operator):
    def __init__(self, epsilon, potential: Potencial, g: Callable, alpha: float, k: float, n: Callable, previous_pinns: List[PinnFlow]):
        self.epsilon = epsilon
        self.potential = potential
        self.g = g
        self.alpha = alpha
        self.k = k
        self.n = n
        self.previous_pinns = previous_pinns
        super(VariationalBoundaryOperator_post, self).__init__(name="VariationalBoundaryOperator_post")

    def call_method(self, u: Callable, domain: List[tf.Tensor]):
        assert len(domain) == 1, "This operator only works with 1d problems."
        return (self.k / self.alpha + 1 / 2 / self.epsilon * self.potential.grad_eval(domain) * self.n(domain))* (u(domain)+sum([pinn.model(domain) for pinn in self.previous_pinns]))**2-2*tf.exp(-self.potential.eval(domain) / 2 / self.epsilon) / self.alpha * self.g(domain) * (u(domain)+sum([pinn.model(domain) for pinn in self.previous_pinns]))


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

    def experiment(self, n_samplings: int, n_train_r: int, r_weight_proportion: float,
                   max_samplings: int, n_iters_per_sampling: int,
                   coords2predict: np.ndarray, x_bounds: Bounds,
                   intelligent_model: Callable[[], BaseModelFlow], **kwargs) -> List[np.ndarray]:
        
        u_predictions = []
        previous_pinns = []
        
        # --------- core experiment --------- #
        num_per_dim2pred = len(coords2predict)
        conditions = dict()
        weight_proportion = {
            "boundary_condition": 1 - r_weight_proportion,
            "residuals": r_weight_proportion
        }
        for i in range(n_samplings):
            if i == 0:
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
                    
                sampler = partial(np.random.uniform, x_bounds.lower, x_bounds.upper)
        
                conditions["residuals"] = Condition(
                    operator=VariationalOperator(epsilon=self.epsilon, potential=self.potential),
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
                previous_pinns.append(pinn)
                u_predictions.append(pinn.predict(domain=coords2predict, which="u").reshape((num_per_dim2pred)) * np.exp(
                        self.potential.eval([coords2predict]) / 2 / self.epsilon).reshape((num_per_dim2pred)))
                #pinn.free_tf_session()
                #del pinn
            else:       
                conditions["boundary_condition"] = Condition(
                    operator=VariationalBoundaryOperator_post(epsilon=self.epsilon, potential=self.potential, g=self.g,
                                                                alpha=self.alpha, k=self.k, n=self.n,previous_pinns=previous_pinns),
                    function=lambda x: 0 * x,
                    sampling_strategy=[
                        ("x", partial(np.linspace, x_bounds.lower, x_bounds.upper))
                        ],
                    valid_sampling_strategy=[
                        ("x", partial(np.linspace, x_bounds.lower, x_bounds.upper))
                        ],
                    n_train=2
                    )
                

                sampler = partial(np.random.uniform, x_bounds.lower, x_bounds.upper)
        
                conditions["residuals"] = Condition(
                    operator=VariationalOperator_post(epsilon=self.epsilon, potential=self.potential, previous_pinns=previous_pinns),
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
                    **conditions
                    )
    
                previous_pinns.append(PinnFlow(
                    model=intelligent_model(),
                    differential_equation=differential_equation,
                    loss_metric=self.loss_metric,
                    n_iters_per_sampling=n_iters_per_sampling,
                    max_samplings=max_samplings,
                    weight_proportion=weight_proportion,
                    initialize=True
                    ).fit())
                
                u_predictions.append(
                    u_predictions[-1]+\
                    previous_pinns[-1].predict(domain=coords2predict, which="u").reshape((num_per_dim2pred))\
                    * np.exp(self.potential.eval([coords2predict]) / 2 / self.epsilon).reshape((num_per_dim2pred)))
            print("Weights median: ", list(map(np.median, previous_pinns[-1].sess.run(previous_pinns[-1].model.weights)))) 
                
        return [u_predictions[-1]]


if __name__ == "__main__":
    experiment_path = Path.joinpath(results_path, 'VariationalPINN')
    experiment_path.mkdir(parents=True, exist_ok=True)

    ve = VariationalPinn(epsilon=EPSILON, potential=potential, g=g, n=n, k=k,
                         alpha=alpha, loss_metric="id", exp_integration=False)

    fig, ax = plt.subplots()
    for l in range(num_repetitions):
        u_predictions = ve.experiment(
            n_samplings=2,
            n_train_r=n_train_r,
            r_weight_proportion=r_weight_proportion,
            max_samplings=1,
            n_iters_per_sampling=10000,
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
