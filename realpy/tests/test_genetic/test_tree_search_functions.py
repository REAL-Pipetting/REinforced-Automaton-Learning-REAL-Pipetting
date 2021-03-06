"""Test tree search functions."""

import unittest
import numpy as np
import realpy.genetic.GA_functions as GA_functions
import realpy.genetic.tree_search_functions as tree_search_functions


class TestGA(unittest.TestCase):
    """Test search functions."""

    def test_functions(self):
        """Test all functions."""
        GA_functions.set_seed(1)
        spectra_array = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3],
                                  [4, 4, 4], [0, 0, 0]])
        conc_array = np.array([[1, 1], [1, 1], [3, 4], [5, 6], [2, 4]])
        conc_array_actual = conc_array
        spectra_array_actual = spectra_array
        x_test = np.array([1, 2, 3]).reshape(1, -1)

        # test zeroth_iteration
        next_gen_conc, current_gen_spectra, median_fitness_list,\
            max_fitness_list, iteration, mutation_rate_list,\
            mutation_rate_list_2 = tree_search_functions.zeroth_iteration(
                conc_array, spectra_array, x_test)
        self.assertEqual(conc_array.shape, next_gen_conc.shape,
                         "Shape of conc array isn't equal in zeroth iteration")

        Iterations = 5
        Moves_ahead = 2
        GA_iterations = 2
        n_samples = 5
        seed = 1

        # test nth_iteration
        mutation_rate, mutation_rate_2, mutation_rate_list,\
            mutation_rate_list_2, best_move, best_move_turn,\
            max_fitness, surrogate_score, next_gen_conc,\
            best_conc_array,\
            dictionary_of_moves = tree_search_functions.nth_iteration(
                Iterations, Moves_ahead, GA_iterations,
                n_samples, current_gen_spectra, next_gen_conc,
                x_test, conc_array_actual, spectra_array_actual,
                seed, median_fitness_list, max_fitness_list, iteration,
                mutation_rate_list, mutation_rate_list_2)
        self.assertEqual(conc_array.shape, next_gen_conc.shape,
                         'Shape of conc array is not equal in nth iteration')
        self.assertEqual(best_move.shape, (1, 3), 'best move is wrong')

        # test MCTS
        mutation_rate, fitness_multiplier, best_move, best_move_turn, \
            max_fitness, surrogate_score, desired_1, current_gen_spectra_1, \
            best_conc_array, dictionary_of_moves = tree_search_functions.MCTS(
                Iterations, Moves_ahead,
                GA_iterations, current_gen_spectra,
                next_gen_conc, x_test, conc_array_actual,
                spectra_array_actual, seed, n_samples)
        self.assertEqual(best_move.shape, (1, 3), 'best move is wrong')

        # test perform_Surrogate_Prediction
        simulated_spectra, surrogate_score = \
            tree_search_functions.perform_Surrogate_Prediction(
                next_gen_conc,
                conc_array_actual,
                spectra_array_actual)
