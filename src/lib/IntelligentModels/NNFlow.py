from itertools import chain
from typing import Tuple, List

import tensorflow as tf

from lib.IntelligentModels.BaseModelFlow import BaseModelFlow


class NNFlow(BaseModelFlow):
    def __init__(self, hidden_layers: Tuple, num_linear_blocks=0, limit_zero=False, random_seed=None,
                 float_precision=tf.float64, activation="tanh"):
        super().__init__(
            name="Flow_NN_{}{}".format(
                "_".join(list(map(str, hidden_layers))),
                "_lb{}".format(num_linear_blocks) if num_linear_blocks else ""
            ),
            float_precision=float_precision
        )

        self.hidden_layers = hidden_layers
        self.num_linear_blocks = num_linear_blocks
        self.layers = None
        self.limit_zero = limit_zero
        self.activation = getattr(tf, activation)

        self.weights = []
        self.biases = []

        self.random_seed = random_seed

    # ------------- network architecture and initialization ------------
    def initialize(self, input_dim, output_dim):
        tf.set_random_seed(self.random_seed)
        self.layers = [input_dim] \
                      + list(chain(*[[hl] * (1 + self.num_linear_blocks) for hl in self.hidden_layers])) \
                      + [output_dim]

        num_layers = len(self.layers)
        for l in range(num_layers - 1):
            W = self.xavier_init(size=[self.layers[l], self.layers[l + 1]])
            b = tf.Variable(tf.zeros([1, self.layers[l + 1]], dtype=self.float_precision), dtype=self.float_precision)
            self.weights.append(W)
            self.biases.append(b)

    @property
    def parameters(self):
        return self.weights, self.biases

    @parameters.setter
    def parameters(self, params):
        self.weights, self.biases = params

    def __call__(self, eq_diff_domain: List[tf.Tensor]):
        X = tf.concat(eq_diff_domain, 1)
        H = X
        for l, (W, b) in enumerate(zip(self.weights, self.biases)):
            H = tf.add(tf.matmul(H, W), b)
            if ((self.num_linear_blocks > 0 and l % self.num_linear_blocks == self.num_linear_blocks - 1) or
                self.num_linear_blocks == 0) and (l < len(self.layers) - 2):
                H = self.activation(H)
        # H = tf.log(tf.exp(H+2) + 1)
        # H = tf.exp(H)
        # H = 1 - tf.exp(1 - H)
        # H = tf.nn.leaky_relu(H)
        # H = 1 - tf.nn.leaky_relu(1 - H)
        return H * X * (X - 1) if self.limit_zero else H
