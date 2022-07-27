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


