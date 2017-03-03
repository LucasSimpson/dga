import random

import copy


class Genotype:
    """Defines a genotype."""

    def __init__(self, size):
        self.size = size
        self.gene = [0 for a in range(self.size)]  # TODO use bytes not char

    @staticmethod
    def random(size):
        """Return a random genotype."""

        g = Genotype(size)
        for i in range(len(g.gene)):
            g.gene[i] = random.random()

        return g

    def __deepcopy__(self, memodict={}):
        genotype = Genotype(self.size)
        genotype.gene = copy.deepcopy(self.gene)
        return genotype

    def __getitem__(self, item):
        return self.gene.__getitem__(item)

    def __setitem__(self, key, value):
        return self.gene.__setitem__(key, value)

    def __delitem__(self, key):
        return self.gene.__delitem__(key)

    def __len__(self):
        return self.gene.__len__()

    def __str__(self):
        return f'G#{self.gene[:4]}...'

