import numpy as np
from mbe_rust.mbe_rust import epanechnikov_density_kde_2d, epanechnikov_density_kde_3d, epanechnikov_density_kde_3d_rev


class MBEdens():
    '''
    purpose:
            Computes number density at each point using the modified Breiman density estimator with variable
            Epanechnikov kernel
    inputs:
            X: N-d array of positions: N-d array
            chunks: number of chunks to break up the calculation of densities into. This can be helpful if the number
            of elements is large, especially when using the KDTree method: int, default=1
            weights: optional weighting for each point: 1-d array
            alpha: the sensitivity parameter: if none alpha=1/d
    outputs:
            rho: density at each point: 1-d array
    '''

    def __init__(self, X, weights=None, alpha=None, n_iter=1, n_threads=20):

        self.n_threads=n_threads
        print(f"Using n_threads= {n_threads}")

        X = np.asarray(X)
        assert len(X.shape) == 2 and X.shape[0] > X.shape[1]
        N, d = X.shape
        self.ndim = d
        if d not in [2,3]:
            raise AttributeError(f"Dimension of {d} isnt currently supported")

        self.alpha = 1. / d if alpha is None else alpha

        if weights is not None:
            weights = np.asarray(weights)
            assert len(weights) == X.shape[0]
        self.weights = weights

        self.points = X

        P = np.percentile(X, [20, 80], axis=0)
        sigma = (P[1] - P[0]) / np.log(N)
        # Take minimum value of sigma to avoid over-smoothing
        self.sigmaopt = np.min(sigma)
        self.lambdaopt = np.ones(X.shape[0])

        for i in range(n_iter):
            print(f"Iterating to find density: {i+1}/{n_iter}")
            pilot_rho = self.find_dens(X)
            g = np.exp(np.sum(np.log(pilot_rho) / N))
            new_lambdaopt = (pilot_rho / g) ** -self.alpha
            print("med diff:", np.median(np.abs(new_lambdaopt - self.lambdaopt)))
            print(f" min labdopt: {np.min(new_lambdaopt)}, max {np.max(new_lambdaopt)}")
            self.lambdaopt = new_lambdaopt

    def find_dens(self, X) -> np.ndarray:
        n_stars, n_dim = X.shape
        assert(n_dim==self.ndim)
        not_nan_filt = ~np.isnan(X).any(axis=1)
        if (~not_nan_filt).sum()>0:
            print("Copying and filtering bad Jvals")
            X_filt = X[not_nan_filt,:].copy()
        else:
            X_filt = X

        if n_dim == 2:
            result = epanechnikov_density_kde_2d( X_filt, self.points, self.lambdaopt, self.sigmaopt, self.n_threads)
        elif n_dim ==3:
            print("Finding with fallback!")
            try:
                result = epanechnikov_density_kde_3d_rev(
                    X_filt,
                    self.points,
                    self.lambdaopt,
                    self.sigmaopt,
                    self.n_threads
                )
            except BaseException as e:
                print("Rev kde failed, falling back to 3d")
                # print(e)
                result = epanechnikov_density_kde_3d(
                    X_filt,
                    self.points,
                    self.lambdaopt,
                    self.sigmaopt,
                    self.n_threads
                )
        else:
            raise AttributeError("n_dim not supported")
        if (~not_nan_filt).sum()>0:
            dens = np.zeros(n_stars)
            dens[not_nan_filt] = result
        else:
            dens=result

        return dens


    def find_dens_3d_original(self, X) -> np.ndarray:
        print("Rev kde failed, falling back to 3d")
        # print(e)
        result = epanechnikov_density_kde_3d(
            X,
            self.points,
            self.lambdaopt,
            self.sigmaopt,
            self.n_threads
        )
        return result
