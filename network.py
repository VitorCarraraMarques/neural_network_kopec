from __future__ import annotations
from typing import List, Callable, TypeVar, Tuple 
from functools import reduce
from layer import Layer 
from util import sigmoid, derivative_sigmoid

T = TypeVar('T') # tipo de saída para interpretação da rede neural 

class Network: 
    def __init__(self, layer_structure, learning_rate, activation_function = sigmoid, derivative_activation_function = derivative_sigmoid):
        if len(layer_structure) < 3:
            raise ValueError("Error: Should be at least 3 layers (1 input, 1 hidden, 1 output")
        self.layers = [] 
        #input layer 
        input_layer = Layer(None, layer_structure[0], learning_rate, activation_function, derivative_activation_function)
        self.layers.append(input_layer)
        # hidden layers 
        for previous, num_neurons in enumerate(layer_structure[1::]):
            next_layer = Layer(self.layers[previous], num_neurons, learning_rate, activation_function, derivative_activation_function)
            self.layers.append(next_layer)

    # Pushes input data to the first layer, then output from the first
    # as input to the second, second to third, etc. 
    def outputs(self, input): 
        return reduce(lambda inputs, layer: layer.outputs(inputs), self.layers, input) 

    # Figure out each neuron's changes based on the erros of the output 
    # versus the expected outcome
    def backpropagate(self, expected):
        # calculate delta for output layer neurons 
        last_layer = len(self.layers) - 1
        self.layers[last_layer].calculate_deltas_for_outputs_layer(expected)
        # calculate delta for hidden layer in reverse order 
        for l in range(last_layer - 1, 0, -1):
            self.layers[l].calculate_deltas_for_hidden_layer(self.layers[l + 1])

    # backpropagate() doesn't actually change any weights 
    # this function uses the deltas calculated in backpropagate() to 
    # actually make changes to the weights
    def update_weights(self):
        for layer in self.layers[1:]: # skip input layer
            for neuron in layer.neurons: 
                for w in range(len(neuron.weights)):
                    neuron.weights[w] = neuron.weights[w] + (neuron.learning_rate * (layer.previous_layer.output_cache[w]) * neuron.delta)
    
    # train() uses the results of outputs() run over manu inputs and compared 
    # against expecteds to feed backpropagate() and update_weights() 
    def train(self, inputs, expecteds): 
        for location, xs in enumerate(inputs):
            ys = expecteds[location]
            outs = self.outputs(xs)
            self.backpropagate(ys)
            self.update_weights()
    
    # for generalized results that require classification this function will return 
    # the correct number of trials and the percentage correct out of the total
    def validate(self, inputs, expecteds, interpret_output):
        correct = 0
        for input, expected in zip(inputs, expecteds):
            result = interpret_output(self.outputs(input))
            if result == expected: 
                correct += 1
        percentage = correct / len(inputs)
        return correct, len(inputs), percentage

        
            