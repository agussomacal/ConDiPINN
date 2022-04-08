from typing import List, Union, Callable

import numpy as np
import tensorflow as tf

T = 0
X = 1
Y = 2
Z = 3


class Operator:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def call_method(self, u: Callable, domain: List[tf.Tensor]):
        raise Exception("Not implemented.")

    def __call__(self, u: Callable, domain: List[tf.Tensor]):
        return self.call_method(u, domain)

    @staticmethod
    def __process_other_call(other):
        if isinstance(other, Operator):
            other_call = other.call_method
        elif isinstance(other, (float, int)):
            def other_call(u: Callable, domain: List[tf.Tensor]):
                return other
        else:
            raise Exception("Only numbers and operators can have arithmetic.")

        return other_call

    def __add__(self, other):
        other_call = self.__process_other_call(other)
        new_operator = Operator(name="{}+{}".format(self, other))
        setattr(new_operator, "call_method", lambda u, domain: self.call_method(u, domain) + other_call(u, domain))
        return new_operator

    def __radd__(self, other):
        other_call = self.__process_other_call(other)
        new_operator = Operator(name="{}+{}".format(self, other))
        setattr(new_operator, "call_method", lambda u, domain: self.call_method(u, domain) + other_call(u, domain))
        return new_operator

    def __sub__(self, other):
        other_call = self.__process_other_call(other)
        new_operator = Operator(name="{}-{}".format(self, other))
        setattr(new_operator, "call_method", lambda u, domain: self.call_method(u, domain) - other_call(u, domain))
        return new_operator

    def __mul__(self, other):
        other_call = self.__process_other_call(other)
        new_operator = Operator(name="{}*{}".format(self, other))
        setattr(new_operator, "call_method", lambda u, domain: self.call_method(u, domain) * other_call(u, domain))
        return new_operator

    def __rmul__(self, other):
        other_call = self.__process_other_call(other)
        new_operator = Operator(name="{}*{}".format(self, other))
        setattr(new_operator, "call_method", lambda u, domain: self.call_method(u, domain) * other_call(u, domain))
        return new_operator

    def __truediv__(self, other):
        other_call = self.__process_other_call(other)
        new_operator = Operator(name="{}/{}".format(self, other))
        setattr(new_operator, "call_method", lambda u, domain: self.call_method(u, domain) / other_call(u, domain))
        return new_operator

    def __pow__(self, power, modulo=None):
        other_call = self.__process_other_call(power)
        new_operator = Operator(name="{}/{}".format(self, power))
        setattr(new_operator, "call_method", lambda u, domain: self.call_method(u, domain) ** other_call(u, domain))
        return new_operator


class Id(Operator):
    def __init__(self):
        super().__init__(name="U")

    def call_method(self, u: Callable, domain: List[tf.Tensor]):
        return u(domain)


class Comparator(Operator):
    def __init__(self, lower, upper):
        super().__init__(name="Comparator")
        self.lower = lower
        self.upper = upper

    def call_method(self, u: Callable, domain: List[tf.Tensor]):
        return tf.square(
            (u(domain) - (self.upper + self.lower) / 2) *  #
            tf.to_float((self.lower > u(domain)) | (u(domain) > self.upper))
        )


class Translation(Operator):
    def __init__(self, axis, length):
        super().__init__(name="Translation")
        self.axis = axis
        self.length = length

    def call_method(self, u: Callable, domain: List[tf.Tensor]):
        new_domain = [d + self.length if i == self.axis else d for i, d in enumerate(domain)]
        return u(new_domain)


class PeriodicCondition(Translation):
    def __init__(self, axis, length):
        super().__init__(axis, length)
        self.name = "PeriodicCondition"

    def call_method(self, u: Callable, domain: List[tf.Tensor]):
        return super(PeriodicCondition, self).call_method(u, domain) - u(domain)


class IntegrateSquares(Operator):
    def __init__(self, axis, n, interval):
        super().__init__(name="IntegrateSquares")
        self.axis = axis
        self.n = n
        self.interval = interval

    def call_method(self, u: Callable, domain: List[tf.Tensor]):
        u_vals = []
        for dom_val in np.linspace(0.0, self.interval, num=self.n):
            points = domain.copy()
            # for i in range(len(domain)):
            #     points[i] += domain[i]  # tf.concat([domain[i]]*self.n, axis=-1)
            points[self.axis] += dom_val
            u_vals.append(u(points))
        return tf.reduce_mean(u_vals, axis=0) * self.interval


class D(Operator):
    def __init__(self, derive_respect_to: Union[int, List[int]]):
        self.derive_respect_to = derive_respect_to if isinstance(derive_respect_to, List) else [derive_respect_to]
        super().__init__(name="D{}".format("".join(map(str, self.derive_respect_to))))

    def call_method(self, u: Callable, domain: List[tf.Tensor]):
        res = u(domain)
        for ix in self.derive_respect_to:
            res = tf.gradients(res, domain[ix])[0]
        return res


class Top(Operator):
    def __init__(self):
        super().__init__(name="TimeOperator")

    def call_method(self, u: Callable, domain: List[tf.Tensor]):
        return domain[T]


class Dirac(Operator):
    def __init__(self):
        super().__init__(name="Dirac")

    def call_method(self, u: Callable, domain: List[tf.Tensor]):
        return


class Dx(D):
    def __init__(self):
        super().__init__([X])


class Dxx(D):
    def __init__(self):
        super().__init__([X, X])


class Dt(D):
    def __init__(self):
        super().__init__([T])


class Dtt(D):
    def __init__(self):
        super().__init__([T, T])


class Dy(D):
    def __init__(self):
        super().__init__([Y])


class Coord(Operator):
    def __init__(self, axis_coord):
        super().__init__(name="Coord{}".format(axis_coord))
        self.axis_coord = axis_coord

    def call_method(self, u: Callable, domain: List[tf.Tensor]):
        return domain[self.axis_coord]


if __name__ == "__main__":
    # tf placeholders and graph
    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
        device_count={'CPU': 1}
    )
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    x = np.zeros((5, 1))
    y = np.ones((5, 1))
    tf_x = tf.placeholder(tf.float64, shape=[None, 1], name="x")
    tf_y = tf.placeholder(tf.float64, shape=[None, 1], name="y")

    domain = [tf_x, tf_y]
    axis = 0
    u_vals = []
    for dom_val in np.linspace(0.0, 9, num=10):
        points = domain.copy()
        # for i in range(len(domain)):
        #     points[i] += domain[i]  # tf.concat([domain[i]]*self.n, axis=-1)
        points[axis] += dom_val
        u_vals.append(tf.reduce_sum(points, axis=0))
    loss = tf.reduce_mean(u_vals, axis=0)
    # loss = tf.shape(u_vals)

    res = sess.run(
        feed_dict={tf_x: x, tf_y: y},
        # fetches=[tf.reduce_mean(tf.concat([tf_x, tf_y], axis=1), axis=1)],
        fetches=[loss],
    )

    print(res)
