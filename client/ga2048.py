import copy
import random

import numpy as np

from base.genetic_algo import GeneticAlgorithm
from base.genotype import Genotype
from base.model import Model
from base.phenotype import Phenotype
from base.aws_lamba import FitnessCall


class Model2048(Model):
    gene_size = 340

    def evaluate(self, data_in):
        return self.nn.infer(data_in)


class Genotype2048(Genotype):
    """Genotype specific to 2048."""

    def crossover(self, other):

        borders = [
            16 * 16,
            16 * 16 + 16,
            16 * 16 + 16 + 16 * 4
        ]

        x1 = copy.deepcopy(self.gene)
        x2 = copy.deepcopy(other.gene)

        assert (len(x1) == len(x2))

        b_id = random.randint(0, len(borders) - 1)
        pivot = borders[b_id]

        g1 = x1[:pivot] + x2[pivot:]
        g2 = x2[:pivot] + x1[pivot:]

        return self.from_gene(g1), self.from_gene(g2)


class Phenotype2048(Phenotype):
    """Model specific for 2048 game_2048."""

    model_class = Model2048
    genotype_class = Genotype2048

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.call = None

    def evaluate_async(self):
        self.call = FitnessCall(self.genotype.gene)
        self.call()

    def join(self):
        self.fitness = int(self.call.get_fitness())


class GA2048(GeneticAlgorithm):
    """Genetic algorithm specific for 2048."""

    model_class = Phenotype2048
    elitist_keep = 10


def run():
    g = GA2048(pop_size=100)

    gen = 0
    m = g.generation()
    while m.fitness < 1000 ** 2:
        gen += 1
        m = g.generation()
        print('Gen {}: {} :: {}'.format(gen, np.math.sqrt(g.fitness()), np.math.sqrt(g.average())))


if __name__ == '__main__':
    run()
