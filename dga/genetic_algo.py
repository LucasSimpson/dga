import copy
import numpy as np

from .phenotype import Phenotype


class GeneticAlgorithm:
    """Decribes a high level genetic algorithm."""

    model_class = Phenotype

    def __init__(self, pop_size=100):
        self.pop_size = pop_size
        self.pop = [self.model_class.from_random() for i in range(self.pop_size)]
        self.eval_and_sort()

    def eval_and_sort(self):
        """Evaluates all phenotypes and sorts accordinly."""

        for pheno in self.pop:
            pheno.evaluate()

        self.pop.sort()

        for pheno in self.pop:
            pass  # print(pheno.fitness)

    def generation(self, num_keep=10) -> Phenotype:
        """Evolve a new generation. Returns the fittest model."""

        # create copy of population, clear current pop
        self.old_pop = copy.deepcopy(self.pop)
        self.pop = list()

        for a in range(num_keep):
            self.pop.append(self.old_pop[a])



        # breed, roulette algo, pop_size iters
        total = sum([pheno.fitness for pheno in self.old_pop])
        probs = [pheno.fitness / total for pheno in self.old_pop]
        for i in range(int((self.pop_size - num_keep)/2)):

            # TODO randomly select
            id1 = np.random.choice(self.pop_size, p=probs)
            id2 = np.random.choice(self.pop_size, p=probs)

            child1, child2 = self.old_pop[id1].breed(self.old_pop[id2])
            self.pop.append(child1)
            self.pop.append(child2)

        # eval and sort
        self.eval_and_sort()

        # return phenotype with highest fitness
        return self.pop[0]

if __name__ == '__main__':
    a = GeneticAlgorithm()
    m = a.generation()
    print(m.fitness)

