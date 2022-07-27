from typing import List, Callable 
from util import dot_product

class Neuron:
    def __init__(self, weights, learning_rate, activation_function, derivative_activation_function):
        self.weights = weights 
        self.activation_function = activation_function
        self.derivative_activation_function = derivative_activation_function
        self.learning_rate = learning_rate 
        self.output_cache = 0.0
        self.delta = 0.0

    def output(self, inputs):
        self.output_cache = dot_product(inputs, self.weights)
        return self.activation_function(self.output_cache)

""" 
Um neurônio nessa rede neural é relativamente simples. Ele é iniciado com uma série de parâmetros, os quais podem 
ser alterados a qualquer momento ao longo do funcionamento da rede neural. O seu propósito é um únio: receber um 
sinal (input) para em seguida retorná-lo ativado. Isto é, calcula o produto escalar do input com os pesos (que foram 
iniciados em init) e ativa esse resultado através da função sigmoid (a nossa função de ativação).
"""