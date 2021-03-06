import copy
import numpy as np

from .phenotype import Phenotype


class GeneticAlgorithm:
    """Decribes a high level genetic algorithm."""

    model_class = Phenotype
    elitist_keep = 10
    mutation_odds = 0.05

    def __init__(self, pop_size=100):
        self.pop_size = pop_size
        self.pop = [self.model_class.from_random() for i in range(self.pop_size)]
        self.eval_and_sort()

    def eval_and_sort(self):
        """Evaluates all phenotypes and sorts accordingly."""

        print('Evaluating...')
        for pheno in self.pop:
            pheno.evaluate_async()
        for pheno in self.pop:
            pheno.join()

        self.pop.sort()

        for pheno in self.pop:
            pass  # print(pheno.fitness)

    def generation(self):
        """Evolve a new generation. Returns the fittest model."""

        print('Breeding...')

        # create copy of population, clear current pop
        self.old_pop = copy.deepcopy(self.pop)
        self.pop = list()

        for a in range(self.elitist_keep):
            self.pop.append(self.old_pop[a])

        # breed, roulette algo, pop_size iters. Only mutate these ones so elites dont mutate at all
        total = sum([pheno.fitness for pheno in self.pop])
        probs = [pheno.fitness / total for pheno in self.pop]
        for i in range(int((self.pop_size - self.elitist_keep)/2)):

            id1 = np.random.choice(self.elitist_keep, p=probs)
            id2 = np.random.choice(self.elitist_keep, p=probs)

            child1, child2 = self.pop[id1].breed(self.pop[id2])

            child1.mutate(self.mutation_odds)
            child2.mutate(self.mutation_odds)

            self.pop.append(child1)
            self.pop.append(child2)

        # eval and sort
        self.eval_and_sort()

        # return phenotype with highest fitness
        return self.pop[0]

    def fitness(self):
        """Return highest fitness in population."""

        return self.pop[0].fitness

    def average(self):
        """Return average fitness of population."""

        return sum(pheno.fitness for pheno in self.pop) / self.pop_size

if __name__ == '__main__':
    a = GeneticAlgorithm()
    m = a.generation()
    print(m.fitness)

