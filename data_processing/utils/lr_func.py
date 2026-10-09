# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 15:55:24 2024

@author: Olive
"""
import tensorflow as tf
import numpy as np
import math 

class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule that combines warmup and cosine annealing."""

    def __init__(
        self, 
        learning_rate_base, 
        total_steps, 
        warmup_learning_rate, 
        warmup_steps,

    ):
        super().__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

        
        if self.total_steps < self.warmup_steps:
            raise ValueError("total_steps must be larger or equal to warmup_steps.")
        if self.learning_rate_base < self.warmup_learning_rate:
            raise ValueError("learning_rate_base must be larger or equal to warmup_learning_rate.")

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        cos_annealed_lr = tf.cos(
            self.pi * (step - self.warmup_steps) / float(self.total_steps - self.warmup_steps)
        )
        learning_rate = 0.5 * self.learning_rate_base * (1 + cos_annealed_lr)

        if self.warmup_steps > 0:
            slope = (self.learning_rate_base - self.warmup_learning_rate) / self.warmup_steps
            warmup_rate = slope * step + self.warmup_learning_rate
            learning_rate = tf.where(step < self.warmup_steps, warmup_rate, learning_rate)
        
        learning_rate = tf.where(step > self.total_steps, 0.0, learning_rate, name="learning_rate")
        return learning_rate

    def get_config(self):
        """Returns the configuration of the learning rate schedule."""
        return {
            'learning_rate_base': self.learning_rate_base,
            'total_steps': self.total_steps,
            'warmup_learning_rate': self.warmup_learning_rate,
            'warmup_steps': self.warmup_steps,

        }
   

class ExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A `LearningRateSchedule` that uses an exponential decay schedule.

    When training a model, it is often useful to lower the learning rate as
    the training progresses. This schedule applies an exponential decay function
    to an optimizer step, given a provided initial learning rate.

    The schedule is a 1-arg callable that produces a decayed learning
    rate when passed the current optimizer step. This can be useful for changing
    the learning rate value across different invocations of optimizer functions.
    It is computed as:

    ```python
    def decayed_learning_rate(step):
        return initial_learning_rate * decay_rate ^ (step / decay_steps)
    ```

    If the argument `staircase` is `True`, then `step / decay_steps` is
    an integer division and the decayed learning rate follows a
    staircase function.

    You can pass this schedule directly into a `keras.optimizers.Optimizer`
    as the learning rate.
    Example: When fitting a Keras model, decay every 100000 steps with a base
    of 0.96:

    ```python
    initial_learning_rate = 0.1
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr_schedule),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(data, labels, epochs=5)
    ```

    The learning rate schedule is also serializable and deserializable using
    `keras.optimizers.schedules.serialize` and
    `keras.optimizers.schedules.deserialize`.

    Args:
        initial_learning_rate: A Python float. The initial learning rate.
        decay_steps: A Python integer. Must be positive. See the decay
            computation above.
        decay_rate: A Python float. The decay rate.
        staircase: Boolean.  If `True` decay the learning rate at discrete
            intervals.
        name: String.  Optional name of the operation.  Defaults to
            `"ExponentialDecay`".

    Returns:
        A 1-arg callable learning rate schedule that takes the current optimizer
        step and outputs the decayed learning rate, a scalar tensor of the
        same type as `initial_learning_rate`.
    """

    def __init__(
        self,
        initial_learning_rate,
        decay_steps,
        decay_rate,
        staircase=False,
        name="ExponentialDecay",
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase
        self.name = name

        if self.decay_steps <= 0:
            raise ValueError(
                "Argument `decay_steps` must be > 0. "
                f"Received: decay_steps={self.decay_steps}"
            )

    def __call__(self, step):
        with tf.name_scope(self.name):
            # Convert inputs to tensors
            initial_learning_rate = tf.convert_to_tensor(self.initial_learning_rate, dtype=tf.float32)
            decay_steps = tf.cast(self.decay_steps, dtype=tf.float32)
            decay_rate = tf.cast(self.decay_rate, dtype=tf.float32)
            global_step_recomp = tf.cast(step, dtype=tf.float32)
            
            # Compute decay factor
            p = global_step_recomp / decay_steps
            if self.staircase:
                p = tf.floor(p)
            
            # Compute and return the decayed learning rate
            return initial_learning_rate * tf.pow(decay_rate, p)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "decay_rate": self.decay_rate,
            "staircase": self.staircase,
            "name": self.name,
        }