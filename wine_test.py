import csv
from typing import List
from util import normalize_by_feature_scaling
from network import Network
from random import shuffle

if __name__ == "__main__":
    wine_parameters = []
    wine_classifications = []
    wine_species = []
    with open('wine.csv', mode='r') as wine_file: 
        wines = list(csv.reader(wine_file, quoting=csv.QUOTE_NONNUMERIC))
        shuffle(wines) # deixa nossas linhas de dados em ordem aleatória
        for wine in wines: 
            parameters = [float(n) for n in wine[1:14]]
            wine_parameters.append(parameters)
            species = int(wine[0])
            if species == 1: 
                wine_classifications.append([1.0, 0.0, 0.0])
            elif species == 2: 
                wine_classifications.append([0.0, 1.0, 0.0])
            else:
                wine_classifications.append([0.0, 0.0, 1.0])
            wine_species.append(species)
    normalize_by_feature_scaling(wine_parameters)

    wine_network = Network([13, 10, 3], 0.5)

    def wine_interpret_output(output):
        if max(output) == output[0]:
            return 1
        elif max(output) == output[1]:
            return 2
        else: 
            return 3
    

    # faz o treinamento nos 150 primeiros vinhos, 10 vezes
    wine_trainers = wine_parameters[0:150]
    wine_trainers_corrects = wine_classifications[0:150]
    for _ in range(50):
        wine_network.train(wine_trainers, wine_trainers_corrects)
    

    # teste nos últimos 28 vinhos do conjunto de dados
    wine_testers = wine_parameters[150:178]
    wine_testers_corrects = wine_species[150:178]
    wine_results = wine_network.validate(wine_testers, wine_testers_corrects, wine_interpret_output)
    print(f"{wine_results[0]} correct of {wine_results[1]} = {wine_results[2] * 100}%")

             
