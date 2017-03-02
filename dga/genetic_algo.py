import copy


class Model:
    """Describes a tasks model. Used for inference."""

    def __init__(self):
        self.fitness = None

    @classmethod
    def from_random(kls):
        """Returns a new Model with random initial parameters."""

        return kls()

    def evaluate(self):
        """Evaluate the model."""

        self.fitness = 10

    def __lt__(self, other):
        assert(self.fitness is not None and other.fitness is not None)
        return self.fitness < other.fitness

    def __deepcopy__(self, memodict={}):
        return self.__class__()


class GeneticAlgorithm:
    """Decribes a high level genetic algorithm."""

    def __init__(self, pop_size=100):
        self.pop_size = pop_size
        self.pop = [Model.from_random() for i in range(self.pop_size)]
        self.eval_and_sort()

    def eval_and_sort(self):
        """Evaluates all phenotypes and sorts accordinly."""

        for pheno in self.pop:
            pheno.evaluate()

        self.pop.sort()

    def breed(self, pheno1, pheno2) -> (Model, Model):
        """Breeds two phenotypes together. Returns a tuple of the two 'children'."""

        # TODO implement
        return pheno1, pheno2

    def generation(self) -> Model:
        """Evolve a new generation. Returns the fittest model."""

        # create copy of population, clear current pop
        self.old_pop = copy.deepcopy(self.pop)
        self.pop = list()

        # breed, roulette algo, pop_size iters
        for i in range(self.pop_size):

            # TODO randomly select
            id1 = 0
            id2 = 1

            child1, child2 = self.breed(self.old_pop[id1], self.old_pop[id2])
            self.pop.append(child1)
            self.pop.append(child2)

        # eval and sort
        self.eval_and_sort()

        # return phenotype with highest fitness
        return self.pop[0]

if __name__ == '__main__':
    a = GeneticAlgorithm();
    m = a.generation()


