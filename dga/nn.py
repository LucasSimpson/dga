import numpy as np


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

        # TODO implement
        return NeuralNetwork()

    def infer(self, data_in):
        """Push data_in through network and give results."""

        l3 = np.dot(data_in, self.l1) + self.b1
        l3 = np.dot(l3 , self.l2) + self.b2

        e = np.exp(l3[0])
        s = sum(e)

        odds = e / s

        return odds

