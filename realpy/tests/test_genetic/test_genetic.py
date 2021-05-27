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
        """Test fitness function."""
        sample_list = [1, 10, 1.3, 5, 6.9, 15, -5, .1, 0]
        mocked_env = mock.MagicMock(name='env')
        mocked_env.sample = mock.MagicMock(name='env_sample',
                                           return_value=sample_list)
        batch_size = 5
        gen1 = np.random.dirichlet((1, 1), batch_size)
        subject = genetic.GA(mocked_env, batch_size, gen1,
                             mutation_rate=.05, n_parents=3)
        self.assertEqual(subject.fitness(), sample_list)

    def test_select_parents(self):
        """Test select_parents function."""
        mocked_env = mock.Mock()

        batch_size = 5
        gen1 = [
            [1],
            [2],
            [3]
        ]
        fitness_scores = [1, 3, 2]
        subject = genetic.GA(mocked_env, batch_size, gen1,
                             mutation_rate=.05, n_parents=3)
        parents = subject.select_parents(fitness_scores)
        self.assertEqual(parents, [[2], [3], [1]])

        subject = genetic.GA(mocked_env, batch_size, gen1,
                             mutation_rate=.05, n_parents=1)
        parents = subject.select_parents(fitness_scores)
        self.assertEqual(parents, [[2]])

    def test_crossover(self):
        """Test crossover function."""
        parents = [
            [1, 2],
            [5, 4]
        ]

        mocked_env = mock.Mock()
        batch_size = 5
        gen1 = np.random.dirichlet((1, 1), batch_size)
        subject = genetic.GA(mocked_env, batch_size, gen1,
                             mutation_rate=0, n_parents=3)
        subject.crossover(parents)
        gen = subject.generation
        possible_gen = [
            [1, 2],
            [5, 4],
            [1, 4],
            [5, 2]
        ]
        for child in gen:
            self.assertIn(child, possible_gen)

    def test_learn(self):
        """Test learn function."""
        sample_list = [1, 10, 1.3, 5, 6.9, 15, -5, .1, 0]
        mocked_env = mock.MagicMock(name='env')
        mocked_env.sample = mock.MagicMock(name='env_sample',
                                           return_value=sample_list)
        batch_size = 5
        gen1 = np.random.dirichlet((1, 1), batch_size)
        subject = genetic.GA(mocked_env, batch_size, gen1,
                             mutation_rate=0.05, n_parents=3)
        # set up mocks of functions called during learn
        subject.select_parents = mock.MagicMock(name='select_parents',
                                                return_value="Parents")
        subject.crossover = mock.MagicMock(name='crossover',
                                           return_value="crossover")
        subject.learn()
