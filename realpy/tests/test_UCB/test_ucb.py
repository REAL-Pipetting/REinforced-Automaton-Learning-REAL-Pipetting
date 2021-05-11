"""Testing functions for GP-UCB class."""

import unittest
import unittest.mock as mock

import realpy.UCB.ucb as ucb

import numpy as np


class TestGPUCB(unittest.TestCase):
    """Test GP UCB class."""
    def test___init__(self):
        """Test initialization of the GP UCB class"""
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


class TestBatchGPUCB(unittest.TestCase):
    """Test Batch GP UCB class."""

    def test_inheritence(self):
        """Ensure the subclass class inherits from parent class."""
        self.assertTrue(issubclass(ucb.BacthGPUCB, ucb.GPUCB))
