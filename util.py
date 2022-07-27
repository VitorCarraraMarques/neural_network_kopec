from typing import List 
from math import exp

# produto escalar de dois vetores 
def dot_product(xs, ys):
    # Maneira que o código está escrito no livro
    return sum(x*y for x,y in zip(xs, ys))

    # Maneira que eu encontrei para explicar: 
    # z = zip(xs, ys)
    # soma = 0 
    # for x,y in z:
    #     soma += x*y 
    # return soma

# a clássica função sigmoid
def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))

# derivada da função sigmoid
def derivative_sigmoid(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

# supõe que todas as linhas têm o mesmo tamanho 
# e faz o feature scaling de cada coluna para que esteja no intervalo de 0 a 1 
def normalize_by_feature_scaling(dataset):
    for col_num in range(len(dataset[0])):
        column = [row[col_num] for row in dataset]
        maximum = max(column)
        minimum = min(column)
        for row_num in range(len(dataset)):
            dataset[row_num][col_num] = (dataset[row_num][col_num] - minimum) / (maximum - minimum)

