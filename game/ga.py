import random
import numpy as np

from dga.genetic_algo import GeneticAlgorithm
from dga.phenotype import Phenotype

from .game import Board


class Model2048(Phenotype):
    """Model specific for 2048 game."""

    def __init__(self):
        super().__init__()

    def infer(self, data_in):
        odds = super().infer(data_in)
        return np.argmax(odds)

    def get_fitness(self):

        b = Board()
        moves = ['w','a','s','d']

        while not b.is_stale():

            # TODO serialize board and use as input
            m = self.infer(np.random.rand(1, 16))
            b.process_move(moves[m])

        return b.score


class GA2048(GeneticAlgorithm):
    """Genetic algorithm specific for 2048."""

    model_class = Model2048



