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

    def breed(self, other):
        """Breeds two phenotypes together. Returns a tuple of the two 'children'."""

        g1, g2 = self.genotype.crossover(other.genotype)

        return self.__class__.from_genotype(g1), self.__class__.from_genotype(g2)

    def mutate(self, mutation_odds):
        """Mutate self's genotype."""

        self.genotype.mutate(mutation_odds)

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
