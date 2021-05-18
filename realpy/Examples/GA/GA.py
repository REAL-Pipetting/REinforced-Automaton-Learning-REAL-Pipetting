import random
import numpy as np
from collections import OrderedDict

class GA(object):
    def __init__(self, env, batch_size, generation, mutation_rate=.05, n_parents=3):
        self.env = env
        self.batch_size = batch_size
        self.generation = generation
        self.n_parents = n_parents
        self.mutation_rate = mutation_rate
        
        
    def fitness(self):
        fitness_scores = []
        for spectra in self.generation:
            fitness_scores.append(self.env.sample(spectra))
        return fitness_scores
        
        
    def select_parents(self, fitness_scores):
        generation_fitness_tuples = zip(fitness_scores, self.generation)
        sorted_by_fitness_generation_dict = OrderedDict(sorted(generation_fitness_tuples, reverse=True, key = lambda x: x[0]))
        return list(sorted_by_fitness_generation_dict.values())[:self.n_parents]
        

    def crossover(self, parents):
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
        max_element = np.max(children)
        new_children = []
        for child in children:
            new_child = []
            for feature in child:
                if random.random() < self.mutation_rate:
                    feature = random.random() * feature
                new_child.append(feature)
            new_children.append(new_child)
        return new_children
    
    def learn(self):
        fitness_scores = self.fitness()
        parents = self.select_parents(fitness_scores)
        self.crossover(parents)
        return self.generation
        
                
