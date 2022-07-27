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

    def outputs(self, inputs):
        if self.previous_layer is None: 
            self.output_cache = inputs
        else: 
            self.output_cache = [n.output(inputs) for n in self.neurons]
        return self.output_cache

    # deve ser chamada somente na camada de saída:
    def calculate_deltas_for_outputs_layer(self, expected):
        for n in range(len(self.neurons)):
            # delta = f'(OutputCache) * (saída esperada - saída real)
            self.neuros[n].delta = self.neurons[n].derivative_activation_function(self.neurons[n].output_cache) * (expected[n] - self.output_cache[n])

    
    # não deve ser chamado na camada de saída: 
    def calculate_deltas_for_hidden_layer(self, next_layer):
        for index, neuron in enumerate(self.neurons):
            next_weights = [n.weights[index] for n in next_layer.neurons]
            # a linha anterior não é a mesma coisa que next_weights = next_layer.neurons.weights ?
            next_deltas = [n.delta for n in next_layer.neurons]
            sum_weights_and_deltas = dot_product(next_weights, next_deltas)
            neuron.delta = neuron.derivative_activation_function(neuron.output_cache) * sum_weights_and_deltas
