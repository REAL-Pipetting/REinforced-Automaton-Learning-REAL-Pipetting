"""Implements Genetic Algorithm (GA)."""

from collections import OrderedDict
import random


class GA(object):
    """Genetic Algorithm agent class."""

    def __init__(self, environment, batch_size, first_generation,
                 mutation_rate=.05, n_parents=3):
        """
        Init function.

        Arguments:
            environment - Environment class. Should have a 'sample' function.
                type == class
            batch_size - HThe number of trials in a single batch.
                type == int
            first_generation - Initial generation of samples
                type == ndarray
            mutation_rate - Hyperparameter to tune how often genes will
            mutate. Should be a value [0, 1]
                default = .05
                type == float
            n_parents - Hyperparameter to tune how many parents per generation
                to select. Should be less than batch_size.
                default = 3
                type == int

        """
        self.env = environment
        self.batch_size = batch_size
        self.generation = first_generation
        self.n_parents = n_parents
        self.mutation_rate = mutation_rate

    def fitness(self):
        """Return a list of fitness values."""
        return self.env.sample(self.generation)

    def select_parents(self, fitness_scores):
        """
        Return the top n_parents of elements/samples, sorted by fitness.

        Arguments:
            fitness_scores - List of fitness scores
                type == list
        Returns:
            list of fitness tuples of top parents
                type == list
        """
        generation_fitness_tuples = zip(fitness_scores, self.generation)
        sort_by_fitness_generation_dict = OrderedDict(
            sorted(generation_fitness_tuples, reverse=True, key=lambda x: x[0])
        )
        return list(sort_by_fitness_generation_dict.values())[:self.n_parents]

    def crossover(self, parents):
        """
        Assign self.generation with the new generation.

        New generation created by crossing-over parents and mutating.

        Arguments:
            parents - List of samples to use as parents
        """
        children = []
        for i in range(self.batch_size):
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)

            child = []
            for i in range(len(parent1)):
                if random.random() > .5:
                    child.append(parent1[i])
                else:
                    child.append(parent2[i])

            children.append(child)
        self.generation = self.mutation(children)

    def mutation(self, children):
        """
        Mutate children by randomly multiplying gene by a value in [.25, .75].

        Likelihood of mutation determined by mutation rate.
        Return mutated children.

        Arguments:
            children - generation of samples to mutate
                type == list
        """
        new_children = []
        for child in children:
            new_child = []
            for feature in child:
                if random.random() < self.mutation_rate:
                    feature = (random.random() / 2 + .25) * feature
                new_child.append(feature)
            new_children.append(new_child)
        return new_children

    def learn(self):
        """Learning function. Generate new generation from previous one."""
        fitness_scores = self.fitness()
        parents = self.select_parents(fitness_scores)
        self.crossover(parents)
        return self.generation
