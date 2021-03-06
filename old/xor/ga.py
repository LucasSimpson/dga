from __future__ import division

import numpy as np

from dga.genetic_algo import GeneticAlgorithm
from dga.model import Model
from dga.phenotype import Phenotype


class NeuralNetwork:
    """Start by defining simple FC NN with layer sizes (2, 2, 2, 1). vars=4+2+4+2=12"""

    def __init__(self):
        self.l1 = np.random.rand(2, 2)
        self.b1 = np.random.rand(1, 2)

        self.l2 = np.random.rand(2, 2)
        self.b2 = np.random.rand(1, 2)

    @staticmethod
    def from_genotype(genotype):
        """Construct a neural network from the genotype."""

        nn = NeuralNetwork()
        i = 0
        for a in range(2):
            for b in range(2):
                nn.l1[a][b] = genotype[i]
                i += 1

        for a in range(2):
            nn.b1 [0][a] = genotype[i]
            i += 1

        for a in range(2):
            for b in range(2):
                nn.l2[a][b] = genotype[i]
                i += 1

        for a in range(2):
            nn.b2[0][a] = genotype[i]
            i += 1

        return nn

    @staticmethod
    def _logistic(x, L=1, k=1, x0=0):
        return 1 / (1 + np.exp(-k*(x - x0)))

    def infer(self, data_in):
        """Push data_in through network and give results."""

        l3 = np.dot(data_in, self.l1) + self.b1
        l3 = NeuralNetwork._logistic(l3)

        l3 = np.dot(l3 , self.l2) + self.b2
        l3 = NeuralNetwork._logistic(l3)

        e = np.exp(l3[0])
        s = sum(e)

        odds = e / s

        return odds


class ModelXOR(Model):
    """Simple model for XOR problem."""

    gene_size = 12

    def __init__(self, genotype=None):
        Model.__init__(self, genotype)
        self.nn = NeuralNetwork.from_genotype(self.genotype)

    def evaluate(self, data_in):
        return self.nn.infer(data_in)


class PhenotypeXOR(Phenotype):
    """Model specific for 2048 game_2048."""

    model_class = ModelXOR

    def __init__(self, *args, **kwargs):
        Phenotype.__init__(self, *args, **kwargs)

    def infer(self, data_in):
        odds = Phenotype.infer(self, data_in)
        return np.argmax(odds)

    def get_fitness(self):

        data_ins = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]
        correct = [0, 1, 1, 0]

        num_right = 0
        for i in range(4):
            ans = self.infer(data_ins[i])
            num_right += 1 if ans == correct[i] else 0

        return num_right**2


class GAXOR(GeneticAlgorithm):
    """Genetic algorithm specific for 2048."""

    model_class = PhenotypeXOR
    elitist_keep = 10


def run():
    g = GAXOR(pop_size=1000)

    gen = 0
    m = g.generation()
    while (m.fitness != 16):
        gen += 1
        m = g.generation()
        print('Gen {}: {} :: {}'.format(gen, g.fitness(), g.average()))

    print(m.infer([0, 0]))
    print(m.infer([1, 1]))
    print(m.infer([1, 0]))
    print(m.infer([0, 1]))

