import numpy as np
from mbe_rust.mbe_rust import (epanechnikov_density_kde_3d_rev,
                               epanechnikov_density_kde_3d_rev_weights,
                               epanechnikov_density_kde_3d_rev_weights_groups,
                               epanechnikov_density_kde_3d_rev_weights_multi)
# epanechnikov_density_kde_2d, epanechnikov_density_kde_2d_rev,
# epanechnikov_density_kde_2d_weights, epanechnikov_density_kde_2d_rev_weights)


class MBEdens:
    '''
    purpose:
            Computes number density at each point using the modified Breiman density estimator with variable
            Epanechnikov kernel
    inputs:
            X: N-d array of positions: N-d array
            weights: optional weighting for each point: 1-d array
    outputs:
            rho: density at each point: 1-d array
    '''

    def __init__(self, X, weights=None, n_iter=5, n_threads=20, lambdaopt=None, sigmaopt=None):

        self.n_threads = n_threads
        # print(f"Using n_threads= {n_threads}")

        X = np.asarray(X)
        assert len(X.shape) == 2 and X.shape[0] > X.shape[1]
        self.n_points, self.ndim = X.shape
        assert self.ndim==3

        if weights is not None:
            weights = np.asarray(weights).copy().astype(np.float64)
            assert len(weights) == X.shape[0]
            self.weights = np.ones_like(weights) #note find smoothing first, set weights after!
            self.mbe_func = epanechnikov_density_kde_3d_rev_weights
        else:
            self.weights = None
            self.mbe_func = epanechnikov_density_kde_3d_rev

        alpha = None
        self.alpha = 1. / self.ndim if alpha is None else alpha

        self.points = X

        if sigmaopt is not None:
            self.sigmaopt = sigmaopt
        else:
            P = np.percentile(X, [20, 80], axis=0)
            sigma = (P[1] - P[0]) / np.log(self.n_points)
            # Take minimum value of sigma to avoid over-smoothing
            self.sigmaopt = np.min(sigma)

        if lambdaopt is not None:
            self.lambdaopt = lambdaopt
        else:
            self.lambdaopt = np.ones(X.shape[0])
            # print(f"Iterating {n_iter} to find density params")
            for i in range(n_iter):
                # print(f"Iterating to find density: {i+1}/{n_iter}")
                pilot_rho = self.find_dens(X)
                g = np.exp(np.sum(np.log(pilot_rho) / self.n_points))
                new_lambdaopt = (pilot_rho / g) ** -self.alpha
                # print("med diff:", np.median(np.abs(new_lambdaopt - self.lambdaopt)))
                # print(f" min labdopt: {np.min(new_lambdaopt)}, max {np.max(new_lambdaopt)}")
                self.lambdaopt = new_lambdaopt

        self.weights = weights
        if weights is not None:
            print("No weight change!")
            # print("Experimental weight hack!")
            # med_weights = np.median(weights)

            # print("1/3")
            # factor = (weights / med_weights)**(1 / self.ndim)
            # print("1/2")
            # factor = (weights / med_weights)**(1 / 2)
            # print("2/3")
            # factor = (weights / med_weights)**(2/3)
            # print("Mid weight stop!, not smaller!")
            # factor[weights<med_weights] = 1
            # self.lambdaopt *= factor

    def find_dens(self, X) -> np.ndarray:
        n_stars, n_dim = X.shape
        assert (n_dim == self.ndim)
        not_nan_filt = ~np.isnan(X).any(axis=1)
        if (~not_nan_filt).sum() > 0:
            # print("Copying and filtering bad Jvals")
            X_filt = X[not_nan_filt, :].copy()
        else:
            X_filt = X

        if self.weights is None:
            result = self.mbe_func(
                    X_filt,
                    self.points,
                    self.lambdaopt,
                    self.sigmaopt,
                    self.n_threads
                )
        else:
                result = self.mbe_func(
                    X_filt,
                    self.points,
                    self.lambdaopt,
                    self.weights,
                    self.sigmaopt,
                    self.n_threads
                )
        if (~not_nan_filt).sum() > 0:
            dens = np.zeros(n_stars)
            dens[not_nan_filt] = result
        else:
            dens = result

        return dens

    def find_dens_forward(self, X) -> np.ndarray:
        n_stars, n_dim = X.shape
        assert (n_dim == self.ndim)
        not_nan_filt = ~np.isnan(X).any(axis=1)
        if (~not_nan_filt).sum() > 0:
            # print("Copying and filtering bad Jvals")
            X_filt = X[not_nan_filt, :].copy()
        else:
            X_filt = X

        if self.weights is None:
            result = self.mbe_func(
                X_filt,
                self.points,
                self.lambdaopt,
                self.sigmaopt,
                self.n_threads
            )
        else:
            result = self.mbe_func(
                X_filt,
                self.points,
                self.lambdaopt,
                self.weights,
                self.sigmaopt,
                self.n_threads
            )
        if (~not_nan_filt).sum() > 0:
            dens = np.zeros(n_stars)
            dens[not_nan_filt] = result
        else:
            dens = result

        return dens


class MBEdens_multi():
    '''
    purpose:
            Computes number density at each point using the modified Breiman density estimator with variable
            Epanechnikov kernel
    inputs:
            X: N-d array of positions: N-d array
            weights: optional weighting for each point: 1-d array
    outputs:
            rho: density at each point: 1-d array
    '''

    def __init__(self, multi_X, multi_weights, n_iter=5, n_threads=20):

        self.n_threads = n_threads
        # print(f"Using n_threads= {n_threads}")
        self.multi_points = multi_X
        self.multi_weights = multi_weights

        self.multi_sigmaopt = []
        self.multi_lambdaopt = []
        self.ndim = self.multi_points[0].shape[1]
        self.n_multi = len(self.multi_points)

        for (x, w) in zip(multi_X, multi_weights):
            single_mbe = MBEdens(x, weights=w, n_iter=n_iter, n_threads=n_threads)
            self.multi_sigmaopt.append(single_mbe.sigmaopt)
            self.multi_lambdaopt.append(single_mbe.lambdaopt)

    def find_dens(self, X) -> np.ndarray:
        n_stars, n_dim = X.shape
        assert (n_dim == self.ndim)
        not_nan_filt = ~np.isnan(X).any(axis=1)
        if (~not_nan_filt).sum() > 0:
            # print("Copying and filtering bad Jvals")
            X_filt = X[not_nan_filt, :].copy()
        else:
            X_filt = X

        result = epanechnikov_density_kde_3d_rev_weights_multi(
            X_filt,
            self.multi_points,
            self.multi_lambdaopt,
            self.multi_weights,
            self.multi_sigmaopt,
            self.n_threads
        )
        if (~not_nan_filt).sum() > 0:
            dens = np.zeros((n_stars, self.n_multi))
            dens[not_nan_filt, :] = result
        else:
            dens = result

        return dens

class MBEdens_groups():
    '''
    purpose:
            Computes number density at each point using the modified Breiman density estimator with variable
            Epanechnikov kernel
    inputs:
            X: N-d array of positions: N-d array
            weights: optional weighting for each point: 1-d array
    outputs:
            rho: density at each point: 1-d array
    '''

    def __init__(self, X, weights,group_inds,sigmaopt=None,lambdaopt=None, n_iter=5, n_threads=20):

        self.n_threads = n_threads
        # print(f"Using n_threads= {n_threads}")
        self.points = X
        self.group_inds = group_inds
        self.n_groups = int(np.max(group_inds)+1)
        self.n_points, self.ndim = self.points.shape

        self.weights = np.ones(self.n_points) #note find smoothing first, set weights after!

        alpha = None
        self.alpha = 1. / self.ndim if alpha is None else alpha

        if sigmaopt is not None:
            self.sigmaopt = sigmaopt
        else:
            P = np.percentile(X, [20, 80], axis=0)
            sigma = (P[1] - P[0]) / np.log(self.n_points)
            # Take minimum value of sigma to avoid over-smoothing
            self.sigmaopt = np.min(sigma)

        if lambdaopt is not None:
            self.lambdaopt = lambdaopt
        else:
            self.lambdaopt = np.ones(X.shape[0])
            # print(f"Iterating {n_iter} to find density params")
            for i in range(n_iter):
                # print(f"Iterating to find density: {i+1}/{n_iter}")
                pilot_rho = self._find_dens(X).sum(axis=-1)
                g = np.exp(np.sum(np.log(pilot_rho) / self.n_points))
                new_lambdaopt = (pilot_rho / g) ** -self.alpha
                # print("med diff:", np.median(np.abs(new_lambdaopt - self.lambdaopt)))
                # print(f" min labdopt: {np.min(new_lambdaopt)}, max {np.max(new_lambdaopt)}")
                self.lambdaopt = new_lambdaopt

        if weights is not None:
            assert len(weights) == X.shape[0]
            self.weights = np.asarray(weights).copy().astype(np.float64)

    def find_dens(self, X) -> np.ndarray:
        return self._find_dens(X)

    def _find_dens(self, X) -> np.ndarray:
        n_stars, n_dim = X.shape
        assert (n_dim == self.ndim)
        not_nan_filt = ~np.isnan(X).any(axis=1)
        if (~not_nan_filt).sum() > 0:
            # print("Copying and filtering bad Jvals")
            X_filt = X[not_nan_filt, :].copy()
        else:
            X_filt = X

        result = epanechnikov_density_kde_3d_rev_weights_groups( X_filt, self.points, self.lambdaopt, self.weights,
                                                                self.group_inds, self.n_groups, self.sigmaopt, self.n_threads)
        if (~not_nan_filt).sum() > 0:
            dens = np.zeros((n_stars, self.n_groups))
            dens[not_nan_filt, :] = result
        else:
            dens = result
        return dens
