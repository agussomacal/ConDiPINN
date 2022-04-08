from typing import List

import numpy as np
import tensorflow as tf


class BaseModelFlow:
    def __init__(self, name, float_precision=tf.float64):
        self.name = name
        self.float_precision = float_precision

    def __str__(self):
        return self.name

    @property
    def parameters(self):
        raise Exception("Not implemented")

    @parameters.setter
    def parameters(self, params):
        raise Exception("Not implemented")

    def initialize(self, input_dim, output_dim):
        raise Exception("Not implemented.")

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(
            tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=self.float_precision),
            dtype=self.float_precision)

    def model(self, eq_diff_domain: List[tf.Tensor]):
        raise Exception("Not implemented.")

    def call_method(self, eq_diff_domain: List[tf.Tensor]):
        raise Exception("Not implemented.")

    def __call__(self, eq_diff_domain: List[tf.Tensor]):
        return self.call_method(eq_diff_domain)

    def __add__(self, other):
        new_model = BaseModelFlow(name="{}+{}".format(self.name, other.name))
        # other_call = self.__process_other_call(other)
        setattr(new_model, "call_method", lambda domain: self(domain) + other(domain))
        setattr(new_model, "model", lambda domain: self(domain) + other(domain))
        return new_model
