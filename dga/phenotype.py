import copy
import random

from dga.model import Model
from .genotype import Genotype


class Phenotype:
    """Describes a tasks model. Used for inference."""

    model_class = Model

    def __init__(self, genotype=None):
        self.fitness = None
        self.genotype = genotype if genotype else Genotype.random(self.model_class.gene_size)
        self.model = self.model_class(self.genotype)

    @classmethod
    def from_random(cls):
        """Returns a new Phenotype with random initial parameters."""

        return cls()

    @classmethod
    def from_genotype(cls, genotype):
        """Returns a new Phenotype from genotype."""

        return cls(genotype=genotype)

    def evaluate(self):
        """Evaluate the model."""

        self.fitness = self.get_fitness()

    def breed(self, other, mutation_odds=0.2):
        """Breeds two phenotypes together. Returns a tuple of the two 'children'."""

        g1 = copy.deepcopy(self.genotype)
        g2 = copy.deepcopy(other.genotype)

        assert(len(g1) == len(g2))

        size = len(g1)
        pivot = random.randint(0, size-1)

        a1 = g1[:pivot]
        a2 = g1[pivot:]
        b1 = g2[:pivot]
        b2 = g2[pivot:]

        g1 = a1 + b2
        g2 = a2 + b1

        if random.random() <= mutation_odds:
            g1[random.randint(0, len(g1)-1)] = random.random()

        if random.random() <= mutation_odds:
            g2[random.randint(0, len(g2)-1)] = random.random()

        g1 = Genotype.from_gene(g1)
        g2 = Genotype.from_gene(g2)

        return self.__class__.from_genotype(g1), self.__class__.from_genotype(g2)

    def get_fitness(self):
        raise NotImplemented()

    def infer(self, data_in):
        return self.model.evaluate(data_in)

    def __lt__(self, other):
        """For sorting by fitness descending."""

        assert(self.fitness is not None and other.fitness is not None)
        return self.fitness > other.fitness

    def __deepcopy__(self, memodict={}):
        c = self.__class__(self.genotype)
        c.fitness = self.fitness
        return c
