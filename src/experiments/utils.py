import time
from contextlib import contextmanager
from typing import List, Union, Callable

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from numpy.random import normal

from lib.utils import Bounds


@contextmanager
def timeit(msg):
    t0 = time.time()
    yield
    print('Duracion {}: {}'.format(msg, time.time() - t0))


@contextmanager
def fig_save_context(fig_path: str, figsize=(8, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    yield ax
    plt.savefig(fig_path)
    # plt.show()
    plt.close()


# ----------- Metropolis-Hastings ------------#
def pi_creator(x, R) -> Callable:
    if 0 <= x <= 1:
        y = R(x) ** 2
    else:
        y = 0
    return y


def prop(x):
    return x + normal(size=len(x))


def q(x, y):
    dist = x - y
    return np.exp(-.5 * np.dot(dist, dist))


def MH(N, pi, q, prop, x0=np.zeros(1)):
    x = x0
    trajectory = [x0]
    for i in range(1, N):
        y = prop(x)
        ratio = pi(np.array([y])) * q(x, y) / pi(np.array([x])) / q(y, x)
        a = np.min([1., ratio])
        r = np.random.rand()
        if r < a:
            x = y
        trajectory += [x]
    return np.array(trajectory)


# ------------------------------------
#          domain samplers
def sample_domain(domain_bounds: List[Bounds], num_per_dim_list: Union[int, List[int]], sampling_function: Callable):
    num_per_dim_list = num_per_dim_list if isinstance(num_per_dim_list, List) else [num_per_dim_list] * len(
        domain_bounds)
    return np.transpose(list(map(np.ravel, np.meshgrid(
        *[sampling_function(bounds.lower, bounds.upper, num_per_dim) for num_per_dim, bounds in
          zip(num_per_dim_list, domain_bounds)]))))


def get_prediction_domain(domain_bounds, num_per_dim2pred):
    return sample_domain(domain_bounds, num_per_dim_list=num_per_dim2pred, sampling_function=np.linspace)


def get_validation_domain(domain_bounds, num_per_dim2pred):
    return sample_domain(domain_bounds, num_per_dim_list=num_per_dim2pred, sampling_function=np.random.uniform)


# ------------------------------------
#              functions
def indicatrice(t, x, x0=0.1, xf=0.5, high=1):
    return np.reshape((x >= x0) & (x < xf) * high, (-1, 1))


def indicatrice_without_t(x, x0=0.1, xf=0.5, high=1):
    return np.reshape((x >= x0) & (x < xf) * high, (-1, 1))


def sin(t, x):
    return np.sin(x * 2 * np.pi).reshape((-1, 1))


# ------------------------------------
#              samplers
def beta_sampling(low, high, num=1, a=1.0, b=1.0):
    return np.random.beta(a, b, size=num) * (high - low) + low


def beta_sampling_01(low, high):
    return beta_sampling(low, high, a=0.1, b=0.1)


def beta_sampling_b1(low, high):
    return beta_sampling(low, high, a=1, b=1)


def beta_sampling_b10(low, high):
    return beta_sampling(low, high, a=1, b=10)


def beta_sampling_b100(low, high):
    return beta_sampling(low, high, a=1, b=100)


def uniform_sur1(low, high):
    return beta_sampling(low, high / 1, a=1, b=100)


def uniform_sur2(low, high):
    return beta_sampling(low, high / 2, a=1, b=100)


def uniform_sur4(low, high):
    return beta_sampling(low, high / 4, a=1, b=100)


def beta_sampling_binf(low, high):
    return beta_sampling(low, high, a=1, b=np.inf)


def beta_sampling_binfx2(low, high):
    a, b = np.random.choice([1, np.inf], replace=False, size=2)
    return beta_sampling(low, high, a=a, b=b)


def dirac_sampling(low, high, point):
    if point == "low":
        point = low
    elif point == "high":
        point = high
    assert low <= point <= high
    return point


def unif(low, high, num, percentage=1):
    return np.random.uniform(low, high * percentage, num)


# ------------------------------------
#              Exact solutions
def transport_exact_solution(domain: np.ndarray, init_condition_function, velocity):
    t = domain[:, 0]
    x = domain[:, 1]
    return init_condition_function(t, x - velocity * t).reshape((-1, 1))


def exact_solution(domain: np.ndarray, epsilon):
    x = domain[:, 0]
    alpha = -1 / epsilon / (np.exp(1 / epsilon) - 1)
    beta = -alpha * epsilon
    return (alpha * epsilon * np.exp(x / epsilon) + x + beta).reshape((-1, 1))


def robin_exact_solution(domain: np.ndarray, epsilon, k, alpha):
    c2 = (-k - 2 * alpha) / (np.exp(1 / epsilon) * (alpha / epsilon + k) - k + alpha / epsilon)
    c1 = (1 / k) * (alpha - c2 * (k - alpha / epsilon))
    return (c1 + c2 * np.exp(domain / epsilon) + domain)


def robin_exact_solution_cv(domain: np.ndarray, epsilon, k, alpha):
    c2 = (-k - 2 * alpha) / (np.exp(1 / epsilon) * (alpha / epsilon + k) - k + alpha / epsilon)
    c1 = (1 / k) * (alpha - c2 * (k - alpha / epsilon))
    return (c1 + c2 * np.exp(domain * epsilon / epsilon) + domain * epsilon) / epsilon


# -----------------------------------------
#           plot utils
def plot_predictions(ax, coords2predict, exact_solution_function, u_predictions_best, alpha=1.0):
    u_true = exact_solution_function(coords2predict)

    ax.plot(coords2predict, u_true, "--", c="blue", label="exact solution", linewidth=4)
    ax.plot(coords2predict, u_predictions_best, "-", c="red", label="predicted solution", alpha=alpha)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.legend()
    # delta = (np.nanmax(u_true) - np.nanmin(u_true)) * 0.2
    # ax.set_ylim((np.nanmin(u_true)-delta, np.nanmax(u_true)+delta))
    ax.grid(True)


class Potencial:
    def eval(self, x: Union[List[tf.Tensor], List[np.ndarray]]):
        raise Exception("Not implemented.")

    def grad_eval(self, x: Union[List[tf.Tensor], List[np.ndarray]]):
        raise Exception("Not implemented.")

    def lap_eval(self, x: Union[List[tf.Tensor], List[np.ndarray]]):
        raise Exception("Not implemented.")


class MinusX(Potencial):
    def eval(self, x: Union[List[tf.Tensor], List[np.ndarray]]):
        return -x[0]

    def grad_eval(self, x: Union[List[tf.Tensor], List[np.ndarray]]):
        return -1

    def lap_eval(self, x: Union[List[tf.Tensor], List[np.ndarray]]):
        return 0


class PlusX(Potencial):
    def eval(self, x: Union[List[tf.Tensor], List[np.ndarray]]):
        return x[0]

    def grad_eval(self, x: Union[List[tf.Tensor], List[np.ndarray]]):
        return 1

    def lap_eval(self, x: Union[List[tf.Tensor], List[np.ndarray]]):
        return 0


def g(x: List[tf.Tensor]):
    return 0


def n(x: List[tf.Tensor]):
    return x[0] * 2 - 1
