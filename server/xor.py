from pymatrix import Matrix
import math

from board import Board


class NeuralNetwork:
    """Start by defining simple FC NN with layer sizes (16, 16, 4, 1). vars=16*16 + 16 + 16*4 + 4 = 340"""

    size = 340

    def __init__(self):
        self.l1 = Matrix(16, 16)
        self.b1 = Matrix(1, 16)
        self.l2 = Matrix(16, 4)
        self.b2 = Matrix(1, 4)

    @staticmethod
    def from_genotype(genotype):
        """Construct a neural network from the genotype."""

        assert (len(genotype) == NeuralNetwork.size)

        nn = NeuralNetwork()
        i = 0
        for a in range(16):
            for b in range(16):
                nn.l1[a][b] = genotype[i]
                i += 1

        for a in range(16):
            nn.b1[0][a] = genotype[i]
            i += 1

        for a in range(16):
            for b in range(4):
                nn.l2[a][b] = genotype[i]
                i += 1

        for a in range(4):
            nn.b2[0][a] = genotype[i]
            i += 1

        return nn

    @staticmethod
    def _logistic(x):
        r = Matrix(1, 2)
        for row, col, elem in x.elements():
            r[row][col] = 1.0 / (1 + math.exp(-elem))
        return r

    @staticmethod
    def argmax(x):
        index = 0
        h = x[0]
        for i in range(1, len(x)):
            if x[i] > h:
                h = x[i]
                index = i
        return index

    def infer(self, data_in):
        """Push data_in through network and give results."""

        l3 = Matrix.from_list([data_in]) * self.l1 + self.b1
        # l3 = NeuralNetwork._logistic(l3)

        l3 = l3 * self.l2 + self.b2
        # l3 = NeuralNetwork._logistic(l3)

        e = [math.exp(elem) for elem in l3[0]]
        s = sum(e)

        odds = [elem / s for elem in e]

        return self.argmax(odds)


class Game2048:
    size = NeuralNetwork.size

    def __init__(self, weights):
        self.nn = NeuralNetwork.from_genotype(weights)

    def evaluate(self):
        b = Board()

        while not b.is_stale():
            state = [0 if s == 0 else math.log(s, 2) for s in b.get_state()]
            move = self.nn.infer(state)
            b.process_move(move)

        return b.get_score() ** 2
