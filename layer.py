from __future__ import annotations
from typing import List, Callable, Optional
from random import random 
from neuron import Neuron
from util import dot_product

class Layer: 
    def __init__(self, previous_layer, num_neurons, learning_rate, activation_function, derivative_activation_function):
        self.previous_layer = previous_layer
        self.neurons = []
        # TODO: todo o código a seguir poderia ser uma grande list comprehension
        for i in range(num_neurons):
            if previous_layer is None: 
                random_weights = []
            else: 
                random_weights = [random() for _ in range(len(previous_layer.neurons))]
            neuron = Neuron(random_weights, learning_rate, activation_function, derivative_activation_function)
            self.neurons.append(neuron)
        self.output_cache = [0.0 for _ in range(num_neurons)]