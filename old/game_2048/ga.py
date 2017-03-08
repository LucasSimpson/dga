from __future__ import division

import copy
import random

import numpy as np

from dga.genetic_algo import GeneticAlgorithm
from dga.genotype import Genotype
from dga.model import Model
from dga.phenotype import Phenotype
from .board import Board


class NeuralNetwork:
    """Start by defining simple FC NN with layer sizes (16, 32, 4, 1)"""

    def __init__(self):
        self.l1 = np.random.rand(16, 32)
        self.b1 = np.random.rand(1, 32)
        self.l2 = np.random.rand(32, 4)
        self.b2 = np.random.rand(1, 4)


    @staticmethod
    def from_genotype(genotype):
        """Construct a neural network from the genotype."""

        nn = NeuralNetwork()
        i = 0
        for a in range(16):
            for b in range(32):
                nn.l1[a][b] = genotype[i]
                i += 1

        for a in range(32):
            nn.b1 [0][a] = genotype[i]
            i += 1

        for a in range(32):
            for b in range(4):
                nn.l2[a][b] = genotype[i]
                i += 1

        for a in range(4):
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


class Model2048(Model):
    gene_size = 676

    def __init__(self, genotype=None):
        Model.__init__(self, genotype)
        self.nn = NeuralNetwork.from_genotype(self.genotype)

    def evaluate(self, data_in):
        return self.nn.infer(data_in)


class Genotype2048(Genotype):
    """Genotype specific to 2048."""

    def crossover(self, other):
        # TODO new crossover based on NN layer boundaries

        borders = [0, 16*32, 16*32+32, 16*32+32+32*4, 16*32+32+32*4+4]

        x1 = copy.deepcopy(self.gene)
        x2 = copy.deepcopy(other.gene)

        assert (len(x1) == len(x2))

        size = len(x1)

        pivot = random.randint(0, size-1)
        b_id = 0
        for i in range(len(borders)-1):
            if pivot > borders[i]:
                b_id = i

        b = borders
        p1 = [
            x1[b[0]:b[1]],
            x1[b[1]:b[2]],
            x1[b[2]:b[3]],
            x1[b[3]:b[4]],
        ]
        p2 = [
            x2[b[0]:b[1]],
            x2[b[1]:b[2]],
            x2[b[2]:b[3]],
            x2[b[3]:b[4]],
        ]

        s = p1[b_id]
        p1[b_id] = p2[b_id]
        p2[b_id] = s

        return Genotype.crossover(self, other)


class Phenotype2048(Phenotype):
    """Model specific for 2048 game_2048."""

    model_class = Model2048
    genotype_class = Genotype2048

    def infer(self, data_in):
        odds = Phenotype.infer(self, data_in)
        return np.argmax(odds)

    def play_game(self):
        b = Board()
        moves = ['w', 'a', 's', 'd']

        while not b.is_stale():
            # TODO serialize board and use as input
            m = self.infer(b.state)
            b.process_move(moves[m])

        return b.score

    def get_fitness(self):

        # play 10 games, return pessimistic lowest score
        scores = list()
        for a in range(10):
            scores.append(self.play_game())

        return min(scores)**2


class GA2048(GeneticAlgorithm):
    """Genetic algorithm specific for 2048."""

    model_class = Phenotype2048
    elitist_keep = 10


def run():
    g = GA2048(pop_size=200)

    gen = 0
    m = g.generation()
    while m.fitness < 1000**2:
        gen += 1
        m = g.generation()
        print('Gen {}: {} :: {}'.format(gen, np.math.sqrt(g.fitness()), np.math.sqrt(g.average())))
