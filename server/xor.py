from pymatrix import Matrix
import math


class NeuralNetwork:
    """Start by defining simple FC NN with layer sizes (2, 2, 2, 1). vars=4+2+4+2=12"""

    size = 12

    def __init__(self):
        self.l1 = Matrix(2, 2)
        self.b1 = Matrix(1, 2)
        self.l2 = Matrix(2, 2)
        self.b2 = Matrix(1, 2)

    @staticmethod
    def from_genotype(genotype):
        """Construct a neural network from the genotype."""

        assert (len(genotype) == NeuralNetwork.size)

        nn = NeuralNetwork()
        i = 0
        for a in range(2):
            for b in range(2):
                nn.l1[a][b] = genotype[i]
                i += 1

        for a in range(2):
            nn.b1[0][a] = genotype[i]
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
    def _logistic(x):
        r = Matrix(1, 2)
        for row, col, elem in x.elements():
            r[row][col] = 1.0 / (1 + math.exp(-elem))
        return r

    def infer(self, data_in):
        """Push data_in through network and give results."""

        l3 = Matrix.from_list([data_in]) * self.l1 + self.b1
        # l3 = NeuralNetwork._logistic(l3)

        l3 = l3 * self.l2 + self.b2
        # l3 = NeuralNetwork._logistic(l3)

        e = [math.exp(elem) for elem in l3[0]]
        s = sum(e)

        odds = [elem / s for elem in e]
        return 0 if odds[0] > odds[1] else 1


class XOR:
    def __init__(self, weights):
        self.nn = NeuralNetwork.from_genotype(weights)

    def evaluate(self):
        tests = [
            ([0, 0], 0),
            ([0, 1], 1),
            ([1, 0], 1),
            ([1, 1], 0)
        ]

        score = 0
        for test in tests:
            score += self.nn.infer(test[0]) == test[1]

        return score ** 2
