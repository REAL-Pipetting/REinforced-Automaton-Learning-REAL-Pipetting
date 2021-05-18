"""Testing functions for GA class."""

# testing imports
import unittest
import unittest.mock as mock

# standard imports
import numpy as np

# our package imports
import realpy.genetic.genetic as genetic


class TestGA(unittest.TestCase):
    """Test GA class."""

    def test___init__(self):
        """Test initialization of the GA class."""
        # initialize
        mocked_env = mock.MagicMock(name='env',
                                    return_value="Batch")
        batch_size = 5
        gen1 = np.random.dirichlet((1, 1), batch_size)
        subject = genetic.GA(mocked_env, batch_size, gen1,
                             mutation_rate=.05, n_parents=3)
        # test assignment of attributes
        self.assertTrue(hasattr(subject, 'env'))
        self.assertTrue(hasattr(subject, 'batch_size'))
        self.assertTrue(hasattr(subject, 'generation'))
        self.assertTrue(hasattr(subject, 'n_parents'))
        self.assertTrue(hasattr(subject, 'mutation_rate'))
        # test values
        self.assertEqual(subject.env, mocked_env)
        self.assertEqual(subject.mutation_rate, .05)
        self.assertEqual(subject.n_parents, 3)

    def test_fitness(self):
        """Test fitness function"""
        mocked_env = mock.Mock()
        mocked_samp = mock.Mock()
        sample_list = [1, 10, 1.3, 5, 6.9, 15, -5, .1, 0]
        mocked_samp.side_effect = sample_list
        mocked_env.sample = mocked_samp

        batch_size = 5
        gen1 = np.random.dirichlet((1, 1), batch_size)
        subject = genetic.GA(mocked_env, batch_size, gen1,
                             mutation_rate=.05, n_parents=3)
        
        self.assertEqual(subject.fitness(), sample_list[:batch_size])
        
        
    def test_select_parents(self):
        """Test select_parents function"""
        
        
        
    def test_crossover(self):
        """Test crossover function"""
        
        
    def test_learn(self):
        """Test learn function"""
        
        
        
        