"""
Implements Gaussian Process Batch Upper Confidence Bound (GP-BUCB).

Based on Parallelizing Exploration-Exploitation Tradeoffs in
Gaussian Process Bandit Optimization, Desautels, et. al, 2014.

"""
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pylab as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor


class GPUCB(object):
    """Heavilty adapted from https://github.com/tushuhei/gpucb."""

    def __init__(self, meshgrid, environment, beta=100.):
        '''
        meshgrid: Output from np.methgrid.
        e.g. np.meshgrid(np.arange(-1, 1, 0.1), np.arange(-1, 1, 0.1)) for 2D space
        with |x_i| < 1 constraint.
        environment: Environment class which is equipped with sample() method to
        return observed value.
        beta (optional): Hyper-parameter to tune the exploration-exploitation
        balance. If beta is large, it emphasizes the variance of the unexplored
        solution solution (i.e. larger curiosity)
        '''
        self.meshgrid = np.array(meshgrid)
        self.environment = environment
        self.beta = beta

        # save param space as the transpose
        self.X_grid = self.meshgrid.reshape(self.meshgrid.shape[0], -1).T
        # initialize means and sigmas
        self.mu = np.array([0. for _ in range(self.X_grid.shape[0])])
        self.sigma = np.array([0.5 for _ in range(self.X_grid.shape[0])])
        # input params that have been tested
        self.X = []
        # time step
        self.T = []

    def argmax_ucb(self):
        return np.argmax(self.mu + self.sigma * np.sqrt(self.beta))

    def learn(self):
        grid_idx = self.argmax_ucb()
        self.sample(self.X_grid[grid_idx])
        gp = GaussianProcessRegressor()
        gp.fit(self.X, self.T)
        self.mu, self.sigma = gp.predict(self.X_grid, return_std=True)

    def sample(self, x):
        t = self.environment.sample(x)
        self.X.append(x)
        self.T.append(t)

    def plot(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_wireframe(self.meshgrid[0], self.meshgrid[1],
            self.mu.reshape(self.meshgrid[0].shape), alpha=0.5, color='g')
        ax.plot_wireframe(self.meshgrid[0], self.meshgrid[1],
            self.environment.sample(self.meshgrid), alpha=0.5, color='b')
        ax.scatter([x[0] for x in self.X], [x[1] for x in self.X], self.T, c='r',
            marker='o', alpha=1.0)
        # plt.savefig('fig_%02d.png' % len(self.X))