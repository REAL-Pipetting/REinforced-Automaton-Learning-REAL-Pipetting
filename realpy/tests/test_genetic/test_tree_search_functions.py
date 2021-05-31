import unittest
import numpy as np
import realpy.genetic.GA_functions as GA_functions
import realpy.genetic.tree_search_functions as tree_search_functions

class TestGA(unittest.TestCase):

    def test_functions(self):
        GA_functions.set_seed(1)
        spectra_array = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4],[0,0,0]])
        conc_array = np.array([[1,1],[1,1],[3,4],[5,6],[2,4]])
        conc_array_actual = conc_array
        spectra_array_actual = spectra_array
        x_test = np.array([1,2,3]).reshape(1,-1)
        next_gen_conc, current_gen_spectra, median_fitness_list, max_fitness_list, iteration, mutation_rate_list, mutation_rate_list_2 = tree_search_functions.zeroth_iteration(conc_array, spectra_array, x_test)
        self.assertEqual(conc_array.shape, next_gen_conc.shape, 'Shape of conc array is not equal in zeroth iteration')
        
        Iterations = 5
        Moves_ahead = 2
        GA_iterations = 2
        n_samples = 5
        seed = 1
        mutation_rate, mutation_rate_2, mutation_rate_list, mutation_rate_list_2, best_move, best_move_turn, max_fitness, surrogate_score, next_gen_conc, best_conc_array, dictionary_of_moves = tree_search_functions.nth_iteration(Iterations, Moves_ahead, GA_iterations, n_samples, current_gen_spectra, next_gen_conc, x_test, conc_array_actual, spectra_array_actual, seed, median_fitness_list, max_fitness_list, iteration, mutation_rate_list, mutation_rate_list_2)
        self.assertEqual(conc_array.shape, next_gen_conc.shape, 'Shape of conc array is not equal in nth iteration')
        self.assertEqual(best_move.shape, (1,3), 'best move is wrong')
       
                   
        