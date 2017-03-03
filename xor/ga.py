import numpy as np

from dga.genetic_algo import GeneticAlgorithm
from dga.phenotype import Phenotype


class PhenotypeXOR(Phenotype):
    """Model specific for 2048 game_2048."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def infer(self, data_in):
        odds = super().infer(data_in)
        return np.argmax(odds)

    def get_fitness(self):

        data_ins = [
            [0, 0] + [0]*14,
            [0, 1] + [0]*14,
            [1, 0] + [0]*14,
            [1, 1] + [0]*14
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


def run():
    g = GAXOR(pop_size=100)

    gen = 0
    m = g.generation()
    while (m.fitness != 16):
        gen += 1
        m = g.generation()
        print(f'Gen {gen}: {g.fitness()} :: {g.average()}')


