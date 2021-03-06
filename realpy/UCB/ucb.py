"""
Implements Gaussian Process Batch Upper Confidence Bound (GP-BUCB).

Based on Parallelizing Exploration-Exploitation Tradeoffs in
Gaussian Process Bandit Optimization, Desautels, et. al, 2014.

"""
import numpy as np
import sklearn.gaussian_process
import smt.sampling_methods
from sklearn.gaussian_process.kernels import RBF


class GPUCB(object):
    """Adapted from https://github.com/tushuhei/gpucb."""

    def __init__(self, meshgrid, environment, beta=100., kernel=RBF):
        """
        Init function.

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
        self.kernel = kernel

        # save param space as the transpose
        self.X_grid = self.meshgrid.reshape(self.input_dimension, -1).T
        # initialize means and sigmas
        # TODO take in an optional prior for initialization (e.g. a trained GP)
        self.mu = np.zeros(self.X_grid.shape[0])
        self.sigma = np.ones(self.X_grid.shape[0])
        # input params that have been explored
        self.X = []
        # outputs of the explored inputs
        self.Y = []
        # time step
        self.T = 0

    def argmax_ucb(self):
        """Return the argmax of the upper confidence bound."""
        return np.argmax(self.mu + self.sigma * np.sqrt(self.beta))

    def learn(self):
        """
        Learning function.

        Each time learn is called, the agent
            1. Finds the set of input parameters with the highest upper
               confidence bound.
            2. Samples those paramaters to get their corresponding outputs.
            3. Trains a new Guassian Process regressor on all data points it
               has seen, including the latest sampling.
            4. Saves the new predicted means and standard deviations.
            5. Increases the time step.
        """
        grid_index = self.argmax_ucb()  # 1
        self.sample(self.X_grid[grid_index])  # 2
        gp = sklearn.gaussian_process.GaussianProcessRegressor(
            kernel=self.kernel)
        gp.fit(self.X, self.Y)  # 3
        self.mu, self.sigma = gp.predict(self.X_grid, return_std=True)  # 4
        self.T = self.T + 1  # 5
        return None

    def sample(self, x):
        """
        Sample the input x using the environment object.

        Save the input and output to the X and Y attributes, respectively.

        Arguments:
            x - The set of input parameters to sample.
                type == tuple
        """
        y = self.environment.sample(x)
        self.X.append(x)
        self.Y.append(y)
        return None


class BatchGPUCB(GPUCB):
    """Batched Guassian Process Upper Confidence Bound agent V1."""

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
        """Return the indices of the batch with the highest UCB."""
        argsort_arr = np.flip(np.argsort(self.mu + self.sigma *
                                         np.sqrt(self.beta)))
        return argsort_arr[:self.batch_size]

    def learn(self):
        """
        Learning function.

        Each time learn is called, the agent
            1. Finds the batch of input parameters with the highest upper
               confidence bound.
            2. Samples those paramaters to get their corresponding batched
               outputs.
            3. Trains a new Guassian Process regressor on all data points it
               has seen, including the latest batched sampling.
            4. Saves the new predicted means and standard deviations.
            5. Increases the time step.
        """
        if self.T == 0:
            self.latin_hypercube_sample()
        else:
            grid_indices = self.argsort_ucb()  # 1
            self.batch_sample(self.X_grid[grid_indices])  # 2
        # get ucb's from GP
        gp = sklearn.gaussian_process.GaussianProcessRegressor(
            kernel=self.kernel)
        # reshape appropriately
        X = np.array(self.X).reshape(-1, self.input_dimension)
        Y = np.array(self.Y).reshape(-1)
        gp.fit(X, Y)  # 3
        self.mu, self.sigma = gp.predict(self.X_grid, return_std=True)  # 4
        # increase time step
        self.T = self.T + 1  # 5
        return None

    def batch_sample(self, xs):
        """
        Sample the sets of input in xs using the environment object.

        Save each input and output pair to the X and Y attributes,
        respectively.

        Arguments:
            xs - The set of input parameters to sample.
                type == list or array of tuples
        """
        ys = []
        for x in xs:
            y = self.environment.sample(x)
            ys.append(y)
        self.Y.append(ys)
        self.X.append(xs)
        return None

    def latin_hypercube_sample(self):
        """
        Do a London Hypercube (LH) sampling of the parameter space.

        LH based on the indexes of the parameter space. Use indices
        to get X values and pass into batch_sample.
        """
        limits = []
        for dim in self.meshgrid.shape[1:]:
            limits.append([0, dim - 1])

        LH_sampler = smt.sampling_methods.LHS(xlimits=np.array(limits),
                                              random_state=42)
        sampled = np.round(LH_sampler(self.batch_size)).astype(int)
        t = self.meshgrid.T
        xs = [t[tuple(sample)] for sample in sampled]
        self.batch_sample(xs)
        return None


class BatchGPUCBv2(BatchGPUCB):
    """
    Batched Guassian Process Upper Confidence Bound agent V2.

    With GP regressions in-batch. Slower compute time, but should
    in general converge in fewer batches than V1.
    """

    def __init__(self, *args, **kwargs):
        """Init function."""
        super(BatchGPUCBv2, self).__init__(*args, **kwargs)
        self.to_exclude = []

    def learn(self):
        """
        Learning function.

        Each time learn is called, the agent
            1. Finds <batch_size> best samples, performing a new GP Regression
               after each sample, assuming that sample returns its mean
            2. "Forgots" assumed samples and actually samples those paramaters
               to get their corresponding batched outputs.
            3. Trains a new Guassian Process regressor on all data points it
               has seen, including the latest batched sampling.
            4. Saves the new predicted means and standard deviations.
            5. Increases the time step.

        For the first timestep, a london hypercube sampling method is used
        """
        if self.T == 0:
            self.latin_hypercube_sample()
        else:
            best_idxs = []
            for i in range(self.batch_size):  # 1
                best_idx = self.get_best_ucb()
                best_idxs.append(best_idx.item())
                gp = sklearn.gaussian_process.GaussianProcessRegressor(
                    kernel=self.kernel)
                self.false_sample(best_idx)
                X = np.array(self.X).reshape(-1, self.input_dimension)
                Y = np.array(self.Y).reshape(-1)
                gp.fit(X, Y)
                self.mu, self.sigma = gp.predict(self.X_grid, return_std=True)

            self.batch_sample(self.X_grid[best_idxs])  # 2

        gp = sklearn.gaussian_process.GaussianProcessRegressor(
            kernel=self.kernel)  # 3
        X = np.array(self.X).reshape(-1, self.input_dimension)
        Y = np.array(self.Y).reshape(-1)
        gp.fit(X, Y)
        self.mu, self.sigma = gp.predict(self.X_grid, return_std=True)  # 4
        self.T = self.T + 1   # 5
        return None

    def batch_sample(self, xs):
        """
        Sample in batches, while forgetting the hallucinated samples.

        Forget assumed samples from within batch. Then sample the sets of
        input in xs using the environment object.
        Save each input and output pair to the X and Y attributes,
        respectively. Reset to_exclude for the next batch.

        Arguments:
            xs - The set of input parameters to sample.
                type == list or array of tuples
        """
        self.X = self.X[0:-self.batch_size]
        self.Y = self.Y[0:-self.batch_size]
        ys = []
        for x in xs:
            y = self.environment.sample(x)
            ys.append(y)
            self.X.append(x)
            self.Y.append(y)
        self.to_exclude = []
        return None

    def false_sample(self, i):
        """
        Create false or hallucinated sample.

        Appends sample parameters and currently known mean to X and Y
        respectively.
        """
        self.X.append(self.X_grid[i].tolist())
        self.Y.append(self.mu[i])
        return None

    def get_best_ucb(self):
        """
        Return the index of the batch with the highest UCB.

        (That has not already been selected in that batch.)
        """
        argsort_arr = np.flip(np.argsort(self.mu + self.sigma *
                                         np.sqrt(self.beta)))
        i = 0
        idx = argsort_arr[i]
        while idx in self.to_exclude:
            i += 1
            if i >= len(argsort_arr):
                break
            idx = argsort_arr[i]
        self.to_exclude.append(idx)
        return idx


class BatchGPUCBv3(BatchGPUCBv2):
    """
    Batched Guassian Process Upper Confidence Bound agent V3.

    With GP regressions in-batch. Prunes the parameter space using
    the lower confidence bound of the best results, so this should
    in general take less compute time than V2.
    """

    def batch_sample(self, xs):
        """
        Sample in batches, while forgetting the hallucinated samples.

        Forget assumed samples from within batch. Then sample the sets of
        input in xs using the environment object.
        Save each input and output pair to the X and Y attributes,
        respectively. Reset to_exclude for the next batch.
        Finally, prune the parameter space X_grid.

        Arguments:
            xs - The set of input parameters to sample.
                type == list or array of tuples
        """
        self.X = self.X[0:-self.batch_size]
        self.Y = self.Y[0:-self.batch_size]
        ys = []
        for x in xs:
            y = self.environment.sample(x)
            ys.append(y)
            self.X.append(x)
            self.Y.append(y)
        self.to_exclude = []

        # Prune 0 potential points in parameter space
        # If the best lcb is higher than a points ucb, we
        # no longer need to regress over that point
        # But don't want to remove information so keep points
        # In the space that we have sampled at
        threshold = np.max(self.mu - self.sigma)
        args = np.argwhere(
            (self.mu + self.sigma * np.sqrt(self.beta)) >= threshold)
        self.X_grid = self.X_grid[args.squeeze()]
        if len(self.X_grid) < self.batch_size:
            self.batch_size = len(self.X_grid)

        return None
