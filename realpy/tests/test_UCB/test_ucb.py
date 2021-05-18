"""Testing functions for GP-UCB class."""

# testing imports
import unittest
import unittest.mock as mock

# standard imports
import numpy as np

# our package imports
import realpy.UCB.ucb as ucb


class TestGPUCB(unittest.TestCase):
    """Test GP UCB class."""

    def test___init__(self):
        """Test initialization of the GP UCB class."""
        # initialize
        mocked_env = mock.MagicMock(name='env',
                                    return_value="Batch")
        n = 10
        coeffs = np.arange(n)
        meshgrid = np.meshgrid(coeffs, coeffs)
        subject = ucb.GPUCB(meshgrid, mocked_env, beta=5)
        # test assignment of attributes
        self.assertTrue(hasattr(subject, 'meshgrid'))
        self.assertTrue(hasattr(subject, 'environment'))
        self.assertTrue(hasattr(subject, 'beta'))
        self.assertTrue(hasattr(subject, 'input_dimension'))
        self.assertTrue(hasattr(subject, 'X_grid'))
        # attributes that get updated during learning
        self.assertTrue(hasattr(subject, 'mu'))
        self.assertTrue(hasattr(subject, 'sigma'))
        self.assertTrue(hasattr(subject, 'X'))
        self.assertTrue(hasattr(subject, 'Y'))
        self.assertTrue(hasattr(subject, 'T'))
        # test appropraite shapes
        self.assertEqual(subject.input_dimension, 2)
        self.assertEqual(subject.X_grid.shape, (n**2, 2))
        self.assertEqual(subject.mu.shape, (n**2,))
        self.assertEqual(subject.sigma.shape, (n**2,))
        # test values
        self.assertEqual(subject.environment, mocked_env)
        self.assertEqual(subject.beta, 5)

    def test_argmax_ucb(self):
        """Test getting the argmax of the UCB."""
        # initialize
        mocked_env = mock.MagicMock(name='env',
                                    return_value="Batch")
        n = 10
        coeffs = np.arange(n)
        meshgrid = np.meshgrid(coeffs, coeffs)
        subject = ucb.GPUCB(meshgrid, mocked_env, beta=1)
        # making the max occur at index 5 by adjucting the mean
        subject.mu[5] = 1
        return_val = subject.argmax_ucb()
        self.assertEqual(return_val, 5)
        # making the max occur at index 10 by adjusting the std
        subject.sigma[10] = 2.
        return_val = subject.argmax_ucb()
        self.assertEqual(return_val, 10)

    @unittest.mock.patch('sklearn.gaussian_process.GaussianProcessRegressor')
    def test_learn(self, mocked_gp):
        """Test the learning function."""
        # initialize
        mocked_env = mock.MagicMock(name='env',
                                    return_value="Batch")
        n = 5
        coeffs = np.arange(n)
        meshgrid = np.meshgrid(coeffs, coeffs)
        subject = ucb.GPUCB(meshgrid, mocked_env, beta=1)

        # set up mocked functions
        subject.sample = mock.MagicMock(name='sample')
        mocked_gp.return_value = mock.MagicMock(name='Mocked GP')
        mocked_gp.return_value.predict.return_value = ('Mu', 'Sigma')

        # making the max occur at index 5 by adjucting the mean
        subject.mu[5] = 1
        subject.learn()

        # test appropriate sampling
        subject.sample.called_with(subject.X_grid[5])
        # test GP correctly called
        mocked_gp.return_value.fit.assert_called_with(subject.X, subject.Y)
        self.assertEqual(subject.mu, 'Mu')
        self.assertEqual(subject.sigma, 'Sigma')
        # check time step increase
        self.assertEqual(subject.T, 1)

    def test_sample(self):
        """Test the environment sampling."""
        # initialize
        mocked_env = mock.MagicMock(name='env')
        mocked_env.sample = mock.MagicMock(name='env_sample',
                                           return_value="Batch")
        n = 3
        coeffs = np.arange(n)
        meshgrid = np.meshgrid(coeffs, coeffs)
        subject = ucb.GPUCB(meshgrid, mocked_env, beta=1)
        subject.sample(2)
        subject.environment.sample.assert_called_with(2)
        self.assertEqual(subject.Y[0], "Batch")


class TestBatchGPUCB(unittest.TestCase):
    """Test Batch GP UCB class."""

    def test_inheritence(self):
        """Ensure the subclass class inherits from parent class."""
        self.assertTrue(issubclass(ucb.BatchGPUCB, ucb.GPUCB))

    def test___init__(self):
        """Test initialization of the Batch GP UCB class."""
        # initialize
        mocked_env = mock.MagicMock(name='env')
        n = 5
        batch_size = 3
        coeffs = np.arange(n)
        meshgrid = np.meshgrid(coeffs, coeffs)
        subject = ucb.BatchGPUCB(batch_size, meshgrid, mocked_env, beta=1)
        # test assignment of additional attribute
        self.assertTrue(hasattr(subject, 'batch_size'))
        self.assertEqual(subject.batch_size, batch_size)

    def test_argsort_ucb(self):
        """Test getting the argsort of the UCB."""
        # initialize
        mocked_env = mock.MagicMock(name='env',
                                    return_value="Batch")
        n = 5
        batch_size = 3
        coeffs = np.arange(n)
        meshgrid = np.meshgrid(coeffs, coeffs)
        subject = ucb.BatchGPUCB(batch_size, meshgrid, mocked_env, beta=1)
        # making the max occur at starting at index 2 for an entire batch
        subject.mu[2:2 + batch_size] = 1
        return_val = subject.argsort_ucb()
        expected_val = np.arange(2, 2 + batch_size)
        np.testing.assert_array_equal(return_val, expected_val)

    @unittest.mock.patch('sklearn.gaussian_process.GaussianProcessRegressor')
    def test_learn(self, mocked_gp):
        """Test the learning function."""
        # initialize
        mocked_env = mock.MagicMock(name='env',
                                    return_value="Batch")
        n = 5
        batch_size = 3
        coeffs = np.arange(n)
        meshgrid = np.meshgrid(coeffs, coeffs)
        subject = ucb.BatchGPUCB(batch_size, meshgrid, mocked_env, beta=1)

        # set up mocked functions
        subject.batch_sample = mock.MagicMock(name='batch sample')
        mocked_gp.return_value = mock.MagicMock(name='Mocked GP')
        mocked_gp.return_value.predict.return_value = ('Mu', 'Sigma')

        subject.learn()

        # test GP correctly called
        self.assertTrue(mocked_gp.return_value.fit.called)
        self.assertEqual(subject.mu, 'Mu')
        self.assertEqual(subject.sigma, 'Sigma')
        # check time step increase
        self.assertEqual(subject.T, 1)

    def test_batch_sample(self):
        """Test the environment sampling."""
        # initialize
        mocked_env = mock.MagicMock(name='env')
        mocked_env.sample = mock.MagicMock(name='env_sample',
                                           return_value="Batch")
        n = 5
        batch_size = 3
        coeffs = np.arange(n)
        meshgrid = np.meshgrid(coeffs, coeffs)
        subject = ucb.BatchGPUCB(batch_size, meshgrid, mocked_env, beta=1)

        indices = [2, 4, 5]
        subject.batch_sample(indices)
        self.assertEqual(subject.Y[0], ["Batch", "Batch", "Batch"])

    @unittest.mock.patch('smt.sampling_methods.LHS')
    def test_london_hypercube_sample(self, mocked_lhs):
        mocked_env = mock.MagicMock(name='env')
        mocked_env.sample = mock.MagicMock(name='env_sample',
                                           return_value="Batch")

        mocked_lhs.return_value = mock.MagicMock(name='LH_sample',
                                                 return_value=[[3.0, 2.1],
                                                               [0.4, 1.1],
                                                               [0.0, 1.9]])

        n = 5
        batch_size = 3
        coeffs = np.arange(n)
        meshgrid = np.meshgrid(coeffs, coeffs)
        subject = ucb.BatchGPUCB(batch_size, meshgrid, mocked_env, beta=1)
        subject.london_hypercube_sample()
        self.assertEqual(list(subject.X[0][0]), [3, 2])
        self.assertEqual(list(subject.X[0][1]), [0, 1])
        self.assertEqual(list(subject.X[0][2]), [0, 2])
        name, args, kwargs = mocked_lhs.mock_calls[0]
        print(kwargs['xlimits'])
        self.assertEqual(kwargs['xlimits'].tolist(), [[0, n], [0, n]])
