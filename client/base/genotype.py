from __future__ import division

import random

import copy


class Genotype:
    """Defines a genotype."""

    def __init__(self, size):
        self.size = size
        self.gene = [0 for a in range(self.size)]

    @classmethod
    def random(cls, size):
        """Return a random genotype."""

        g = cls(size)
        for i in range(len(g.gene)):
            g.gene[i] = random.normalvariate(0, 1)  # mean=0, var=1

        return g

    @classmethod
    def from_gene(cls, gene):
        """Returns a genotype with a specific gene."""

        g = cls(len(gene))
        g.gene = gene
        return g

    def crossover(self, other):
        """Crossover two genotypes together."""

        x1 = copy.deepcopy(self.gene)
        x2 = copy.deepcopy(other.gene)

        assert (len(x1) == len(x2))

        size = len(x1)
        pivot = random.randint(0, size - 1)

        g1 = x1[:pivot] + x2[pivot:]
        g2 = x2[:pivot] + x1[pivot:]

        return self.from_gene(g1), self.from_gene(g2)

    def mutate(self, mutation_odds):
        """Mutate our genotype. Each gene is mutated with odds mutation_odds/len(self)."""

        p = mutation_odds / len(self.gene)

        for i in range(len(self.gene)):
            if random.random() <= p:
                self.gene[i] = random.normalvariate(0, 1)  # mean=0, var=1

    def __deepcopy__(self, memodict={}):
        genotype = self.__class__(self.size)
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
        return 'G#{}{}'.format(self.gene[:min(6, len(self.gene))], "..." if len(self.gene) > 6 else "")

