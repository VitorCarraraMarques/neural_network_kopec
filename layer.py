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
            self.neurons[n].delta = self.neurons[n].derivative_activation_function(self.neurons[n].output_cache) * (expected[n] - self.output_cache[n])

    
    # não deve ser chamado na camada de saída: 
    def calculate_deltas_for_hidden_layer(self, next_layer):
        for index, neuron in enumerate(self.neurons):
            next_weights = [n.weights[index] for n in next_layer.neurons]
            # a linha anterior não é a mesma coisa que next_weights = next_layer.neurons.weights ?
            next_deltas = [n.delta for n in next_layer.neurons]
            sum_weights_and_deltas = dot_product(next_weights, next_deltas)
            neuron.delta = neuron.derivative_activation_function(neuron.output_cache) * sum_weights_and_deltas


"""
O método init de Layer inicia uma lista de neurônios, chamada neurons. Essa lista é verdadeiramente o que é chamado 
de camada aqui. 

As funções que calculam os deltas, são onde o backpropagation de fato ocorre. 
A camada de saída tem seus deltas calculado a partir da diferença entre a saída esperada e saída real para cada neurônio. 
Isto é, os deltas são calculados a partir do erro dos neurônios da camada de saída. 
Em seguida, esses deltas são propragados para as camadas anteriores, que receberão os deltas das camadas seguintes 
na forma de next_deltas, obtido a partir de next_layer.neurons. Esses deltas das próximas camadas são então, 
utilizados para calcular os deltas da camada atual. E dessa forma, os deltas da camada atual passarão a atuar como 
next_deltas,caso haja outras camadas anteriores.  
"""