"""
Implements Gaussian Process Batch Upper Confidence Bound (GP-BUCB).

Based on Parallelizing Exploration-Exploitation Tradeoffs in
Gaussian Process Bandit Optimization, Desautels, et. al, 2014.

"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor


class GPUCB(object):
    """Heavilty adapted from https://github.com/tushuhei/gpucb."""

    def __init__(self, meshgrid, environment, beta=100.):
        """
        This is adapted from the repo found at
        https://github.com/tushuhei/gpucb

        Arguments:
            meshgrid - The parameter space on which to explore possible inputs.
                Expected to be the output from np.meshgrid
                type == list
            environment - Environment class. Should have a 'sample' function.
                type == class
            beta - Hyper-parameter to tune the exploration-exploitation
                balance. If beta is large, it emphasizes the variance of the
                unexplored solution solution (i.e. larger curiosity)
                default = 100
                type == float

        """
        self.meshgrid = np.array(meshgrid)
        self.environment = environment
        self.beta = beta
        self.input_dimension = self.meshgrid.shape[0]

        # save param space as the transpose
        self.X_grid = self.meshgrid.reshape(self.input_dimension, -1).T
        # initialize means and sigmas
        # TODO take in an optional prior for initialization
        self.mu = np.array([0. for _ in range(self.X_grid.shape[0])])
        self.sigma = np.array([0.5 for _ in range(self.X_grid.shape[0])])
        # input params that have been explored
        self.X = []
        # outputs of the explored inputs
        self.Y = []
        # time step
        self.T = 0

    def argmax_ucb(self):
        """Returns the argmax of the upper confidence bound."""
        return np.argmax(self.mu + self.sigma * np.sqrt(self.beta))

    def learn(self):
        grid_index = self.argmax_ucb()
        self.sample(self.X_grid[grid_index])
        gp = GaussianProcessRegressor()
        gp.fit(self.X, self.Y)
        self.mu, self.sigma = gp.predict(self.X_grid, return_std=True)
        self.T = self.T + 1

    def sample(self, x):
        y = self.environment.sample(x)
        self.X.append(x)
        self.Y.append(y)


class BatchGPUCB(GPUCB):

    def __init__(self, batch_size, *args, **kwargs):
        """
        Child init function.

        Arguments:
            batch_size - The number of trials in a single batch.
                type == int
        args:
            meshgrid - The parameter space on which to explore possible inputs.
                Expected to be the output from np.meshgrid
                type == list
            environment - Environment class. Should have a 'sample' function.
                type == class
        kwargs:
            beta - Hyper-parameter to tune the exploration-exploitation
                balance. If beta is large, it emphasizes the variance of the
                unexplored solution solution (i.e. larger curiosity)
                default = 100
                type == float
        """
        # setting batch size attribute
        self.batch_size = batch_size
        # inherent everything else from parent class
        super(BatchGPUCB, self).__init__(*args, **kwargs)


    def argsort_ucb(self):
        argsort_arr = np.flip(np.argsort(self.mu + self.sigma * np.sqrt(self.beta)))
        return argsort_arr[:self.batch_size]

    def learn(self):
        # insert here "grid search" if self.T is 0
        # currently, it takes the first in the parameter list
        grid_indices = self.argsort_ucb()
        self.batch_sample(self.X_grid[grid_indices])
        # get ucb's from GP
        gp = GaussianProcessRegressor()
        # print(np.array(self.X).shape)
        X = np.array(self.X).reshape(-1, self.input_dimension)
        Y = np.array(self.Y).reshape(-1)
        gp.fit(X, Y)
        self.mu, self.sigma = gp.predict(self.X_grid, return_std=True)
        # increase time step
        self.T = self.T + 1

    def batch_sample(self, xs):
        ys = []
        for x in xs:
            y = self.environment.sample(x)
            ys.append(y)
        self.Y.append(ys)
        self.X.append(xs)

    
    def argsort_ucb_with_random(self):
        argsort_arr = np.flip(np.argsort((self.mu + self.sigma * np.sqrt(self.beta)) * 
                                np.random.normal(loc=1, scale=.1, size=1)))
        
        return argsort_arr[:self.batch_size]

    
    def learn_with_random(self):
        # insert here "grid search" if self.T is 0
        # currently, it takes the first in the parameter list
        grid_indices = self.argsort_ucb_with_random()
        self.batch_sample(self.X_grid[grid_indices])
        # get ucb's from GP
        gp = GaussianProcessRegressor()
        # print(np.array(self.X).shape)
        X = np.array(self.X).reshape(-1, self.input_dimension)
        Y = np.array(self.Y).reshape(-1)
        gp.fit(X, Y)
        self.mu, self.sigma = gp.predict(self.X_grid, return_std=True)
        # increase time step
        self.T = self.T + 1
        
        
    def reset(self, env):
        self.X = []
        self.Y = []
        self.T = 0
        self.mu = np.array([0. for _ in range(self.X_grid.shape[0])])
        self.sigma = np.array([0.5 for _ in range(self.X_grid.shape[0])])
        self.env = env
        

   
