"""
@author: Maziar Raissi
"""
from collections import defaultdict, OrderedDict
from multiprocessing import cpu_count
from typing import Dict, Union, List

import numpy as np
import tensorflow as tf

from lib.DifferentialEquations.DifferentialEquation import DifferentialEquation, APPLY_OPERATOR_TO_U, Condition
from lib.DifferentialEquations.Operators import Id
from lib.IntelligentModels.BaseModelFlow import BaseModelFlow

WEIGHT_PROPORTION_EQUAL = "equal"


class PinnFlow:
    def __init__(self, model: BaseModelFlow, differential_equation: DifferentialEquation,
                 max_samplings: int = 1, n_iters_per_sampling: int = 5000, loss_metric: str = 'l2',
                 actualize_weights=False,
                 weight_proportion: Union[
                     str, Dict[str, Union[int, float]], List[Dict[str, Union[int, float]]]] = WEIGHT_PROPORTION_EQUAL,
                 initialize=True):

        self.differential_equation = differential_equation

        # -------- weight_proportion ---------
        if weight_proportion == WEIGHT_PROPORTION_EQUAL:
            self.weight_proportion = [{k: 1.0 for k in self.differential_equation.condition_names}]
        elif isinstance(weight_proportion, Dict):
            self.weight_proportion = [weight_proportion] * max_samplings
        else:
            self.weight_proportion = weight_proportion

        for i, wp in enumerate(self.weight_proportion):
            self.check_dict_eq_diff_compatibility(wp)
            proportion_sum = sum(wp.values())
            self.weight_proportion[i] = {k: v / proportion_sum for k, v in wp.items()}

        # -------- differential equation ---------
        u_condition = Condition(
            operator=Id(),
            function=lambda *domain: 0,
            n_train=1,
            sampling_strategy=[(var_name, np.random.uniform) for var_name, _ in
                               self.differential_equation.domain_limits],
            apply_operator_to=APPLY_OPERATOR_TO_U
        )
        self.differential_equation.conditions.update({"u": u_condition})
        for wp in self.weight_proportion:
            wp["u"] = 0

        self.weight_proportion_per_iter = self.weight_proportion

        # -------- optimization ---------
        self.max_samplings = len(self.weight_proportion)
        self.n_iters_per_sampling = n_iters_per_sampling
        self.actualize_weights = actualize_weights

        self.train_loss = []
        self.valid_loss = []
        if loss_metric.lower() == 'l2':
            self.loss_metric = tf.square
        elif loss_metric.lower() == 'l1':
            self.loss_metric = tf.abs
        elif loss_metric.lower() == 'max':
            self.loss_metric = tf.reduce_max
        elif loss_metric.lower() == "id":
            self.loss_metric = lambda x: x
        else:
            raise Exception("loss_metric should be one of 'l2' or 'l1'")

        # -------- Initialize NNs ---------
        self.model = model
        if initialize:
            self.model.initialize(differential_equation.input_dim, differential_equation.output_dim)

        # tf placeholders and graph
        config = tf.ConfigProto(
            allow_soft_placement=False,
            log_device_placement=False,
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1,
            device_count={'CPU': 1}
        )
        self.sess = tf.Session(config=config)
        # self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
        #                                              log_device_placement=True))

        self.tf_dict = defaultdict(OrderedDict)
        self.pred = {k: None for k in self.differential_equation.condition_names}
        self.optimizer = None

    # ------------- properties and util functions ------------
    def free_tf_session(self):
        self.sess.close()

    def check_dict_eq_diff_compatibility(self, dictionary):
        assert set(list(dictionary.keys())) == set(self.differential_equation.condition_names), \
            "dictionary keys of {} should coincide with those on the conditions in the differential equation {}".format(
                dictionary.keys(), self.differential_equation.condition_names)

    @staticmethod
    def correct_np_shape(single_var: np.ndarray):
        return np.reshape(single_var, (-1, 1))

    def conditions_iterator(self):
        for condition_name, condition in self.differential_equation.conditions.items():
            if condition.n_train > 0:
                yield condition, condition_name

    def create_tf_dict(self):
        for condition, condition_name in self.conditions_iterator():
            for var_name in condition.generate_var_names(condition_name):
                self.tf_dict[condition_name][var_name] = tf.placeholder(self.model.float_precision, shape=[None, 1],
                                                                        name=var_name)

    def create_np_dict(self, train=True):
        np_dict = defaultdict(dict)
        for condition, condition_name in self.conditions_iterator():
            for var_name, values in zip(condition.generate_var_names(condition_name), condition.generate_values(train)):
                np_dict[condition_name][var_name] = self.correct_np_shape(values)
        return np_dict

    def define_single_losses_functions(self):
        losses = {}
        for condition_name, condition in self.differential_equation.conditions.items():
            # if condition.n_train > 0:
            self.pred[condition_name], tf_true_values = \
                self.differential_equation.get_condition_associated_tf_model(
                    condition_name,
                    self.model,
                    self.tf_dict
                )

            losses.update({condition_name: tf.reduce_mean(
                self.loss_metric(
                    self.pred[condition_name] - tf_true_values)
            )})
        return losses

    # ------------- fit functions ------------
    def fit(self):
        best_parameters = self.model.parameters

        # -------- conditions + tf variables --------
        self.create_tf_dict()
        np_dict_valid = self.create_np_dict(train=False)
        for i, wp_per_iter in enumerate(self.weight_proportion):
            # -------- conditions + values --------
            np_dict_train = self.create_np_dict()
            loss = sum(
                [wp_per_iter[condition_name] * single_loss for condition_name, single_loss in
                 self.define_single_losses_functions().items()])

            # -------- define optimizer -------- #
            self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                loss,
                method='L-BFGS-B',
                options={
                    'maxiter': self.n_iters_per_sampling,
                    'maxfun': self.n_iters_per_sampling,
                    'maxcor': 50,
                    'maxls': 50,
                    'ftol': 1.0 * np.finfo(float).eps
                }
            )

            # -------- start optimization -------- #
            init = tf.global_variables_initializer()
            self.sess.run(init)

            feed_dict = {self.tf_dict[condition][var_name]: np_dict_train[condition][var_name]
                         for condition in np_dict_train.keys() for var_name in np_dict_train[condition]}

            def loss_callback(loss_val):
                self.train_loss.append(loss_val)
                print('\rLoss: {}'.format(loss_val), end="")

            self.optimizer.minimize(
                self.sess,
                feed_dict=feed_dict,
                fetches=[loss],
                loss_callback=loss_callback
            )

            # -------- validation next iteration --------
            if self.actualize_weights:
                # only change weights if there are many samplings to be done and only one weight set is given.
                valid_condition_error = {}
                for condition_name, single_loss in self.define_single_losses_functions().items():
                    feed_dict = {self.tf_dict[condition_name][var_name]: np_dict_valid[condition_name][var_name]
                                 for var_name in np_dict_valid[condition_name]}
                    valid_condition_error.update({condition_name: self.sess.run(
                        feed_dict=feed_dict,
                        fetches=[single_loss],
                    )})
                valid_condition_error = {cond: (np.array(w) if self.weight_proportion[cond] > 0 else 0) for cond, w in
                                         valid_condition_error.items()}
                wp_next = {cond: w / np.sqrt(sum(list(valid_condition_error.values()))) for cond, w in
                           valid_condition_error.items()}
                self.weight_proportion_per_iter.append(wp_next)
                if i < self.max_samplings:
                    self.weight_proportion[i + 1] = wp_next

            # -------- validation --------
            feed_dict = {self.tf_dict[condition][var_name]: np_dict_valid[condition][var_name]
                         for condition in np_dict_valid.keys() for var_name in np_dict_valid[condition]}
            self.valid_loss.append(
                self.sess.run(
                    feed_dict=feed_dict,
                    fetches=[loss],
                )
            )

            if np.argmin(self.valid_loss) == len(self.valid_loss) - 1:
                best_parameters = self.model.parameters
            else:
                self.model.parameters = best_parameters
        return self

    # --------------- prediction functions ---------------
    def predict(self, domain: np.ndarray, which="u"):
        # assert which in ["u"] + self.differential_equation.condition_names, "if should be one of 'u' or conditions"
        assert which in self.differential_equation.condition_names, "if should be one of conditions"

        # condition_name = which
        # if which in ["u"]:
        #     for cond_name, condition in self.differential_equation.conditions.items():
        #         if condition.apply_operator_to == APPLY_OPERATOR_TO_U and self.weight_proportion[cond_name] > 0:
        #             condition_name = cond_name
        #             break
        condition_name = which

        tf_domain, _ = self.differential_equation.get_tf_domain_and_values(condition_name, self.tf_dict)
        return self.sess.run(
            self.pred[condition_name],
            {tf_domain_var: self.correct_np_shape(np_domain_var)
             for tf_domain_var, np_domain_var in zip(tf_domain, domain.T)}
        )
