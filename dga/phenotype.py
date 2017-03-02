from .nn import NeuralNetwork
from .genotype import Genotype


class Phenotype:
    """Describes a tasks model. Used for inference."""

    def __init__(self):
        self.fitness = None
        self.genotype = Genotype.random()
        self.nn = NeuralNetwork.from_genotype(self.genotype)

    @classmethod
    def from_random(kls):
        """Returns a new Model with random initial parameters."""

        return kls()

    def evaluate(self):
        """Evaluate the model."""

        self.fitness = self.get_fitness()

    def get_fitness(self):
        raise NotImplemented()

    def infer(self, data_in):
        return self.nn.infer(data_in)

    def __lt__(self, other):
        """For sorting by fitness descending."""

        assert(self.fitness is not None and other.fitness is not None)
        return self.fitness > other.fitness

    def __deepcopy__(self, memodict={}):
        c = self.__class__()
        c.fitness = self.fitness
        return c