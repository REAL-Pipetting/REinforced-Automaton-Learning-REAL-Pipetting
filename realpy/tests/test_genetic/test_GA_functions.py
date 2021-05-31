import unittest
import numpy as np
import realpy.genetic.GA_functions as GA_functions

class TestGA(unittest.TestCase):

    def test_functions(self):
        GA_functions.set_seed(1)
        spectra = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4],[0,0,0]])
        conc = np.array([[1,1],[1,1],[3,4],[5,6],[2,4]])
        desired = np.array([1,2,3]).reshape(1,-1)
        sorted_array, median_score, max_score = GA_functions.fitness(spectra, conc, desired)
        self.assertEqual(sorted_array.shape[1], 6, 'Check fitness function')
        parents = GA_functions.select_parents(sorted_array, 5)
        self.assertEqual(parents.shape[0], 5, 'Check select_parents function')
        offspring = GA_functions.crossover(parents, 5)
        self.assertEqual(offspring.shape[0], 5, 'Check crossover function')
        array = GA_functions.mutation(offspring, 0.5)
        self.assertEqual(array.shape, offspring.shape, 'Check mutation 1 function')
        array2 = GA_functions.mutation2(array, 0.5)
        self.assertEqual(array2.shape, array.shape, 'Check mutation 2 function')
        
        
 
        
   