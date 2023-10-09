
class MBEdens_boundary_3d():
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

    def __init__(self, X, weights=None, n_iter=5, n_threads=20):

        self.n_threads = n_threads
        # print(f"Using n_threads= {n_threads}")

        X = np.asarray(X)
        assert len(X.shape) == 2 and X.shape[0] > X.shape[1]
        self.n_points, self.ndim = X.shape
        if self.ndim != 3:
            raise AttributeError("Boundary only currently in 3d!")

        if weights is not None:
            self.original_weights = np.asarray(weights).astype(np.float64)
            assert len(weights) == X.shape[0]
        else:
            self.original_weights = np.ones((self.n_points), dtype=float)

        self.boundary_weights = np.ones((self.n_points), dtype=float)
        self.weights = self.original_weights * self.boundary_weights

        self.mbe_rev_func = epanechnikov_density_kde_3d_rev_weights
        self.mbe_func = epanechnikov_density_kde_3d_weights

        alpha = None
        self.alpha = 1. / self.ndim if alpha is None else alpha

        self.points = X

        P = np.percentile(X, [20, 80], axis=0)
        sigma = (P[1] - P[0]) / np.log(self.n_points)
        # Take minimum value of sigma to avoid over-smoothing
        self.sigmaopt = np.min(sigma)
        self.lambdaopt = np.ones(X.shape[0])

        print(f"Iterating {n_iter} to find density params")
        for i in range(n_iter):
            print(f"Iterating to find density: {i+1}/{n_iter}")
            pilot_rho = self.find_dens(X)
            g = np.exp(np.sum(np.log(pilot_rho) / self.n_points))
            new_lambdaopt = (pilot_rho / g) ** -self.alpha
            # print("med diff:", np.median(np.abs(new_lambdaopt - self.lambdaopt)))
            # print(f" min labdopt: {np.min(new_lambdaopt)}, max {np.max(new_lambdaopt)}")
            self.lambdaopt = new_lambdaopt

            self.boundary_adjust()

    def find_dens(self, X) -> np.ndarray:
        n_stars, n_dim = X.shape
        assert (n_dim == self.ndim)
        not_nan_filt = ~np.isnan(X).any(axis=1)
        if (~not_nan_filt).sum() > 0:
            # print("Copying and filtering bad Jvals")
            X_filt = X[not_nan_filt, :].copy()
        else:
            X_filt = X

        try:
            result = self.mbe_rev_func(
                X_filt,
                self.points,
                self.lambdaopt,
                self.weights,
                self.sigmaopt,
                self.n_threads
            )
        except BaseException:
            print("Rev kde failed, falling back to forward")
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

    def chi_hstar(self, hstar) -> np.ndarray:
        chi = 1 / (1 - (((hstar**3) / 16) * (20 - (15 * hstar) + (3 * (hstar**2)))))
        return chi

    def boundary_adjust(self) -> None:
        boundary_weights = np.ones(self.n_points)

        for bound_dist in [self.boundary_dist_Jz0, self.boundary_dist_JR0]:
            boundary_dist = bound_dist()
            out_of_bound = boundary_dist < 1
            hstar = 1 - boundary_dist[out_of_bound]
            boundary_weights[out_of_bound] *= self.chi_hstar(hstar)
            print(f"Max correction: {np.max(boundary_weights)}")
            print(
                f"found at: {self.points[np.argmax(boundary_weights),:]}, lambdaoptsigma = {self.sigmaopt*self.lambdaopt[np.argmax(boundary_weights),:]}")

        self.weights = self.original_weights * boundary_weights

    def boundary_dist_JR0(self) -> np.ndarray:
        print("JR0 boundary")
        return self.points[:, 0] / (self.lambdaopt * self.sigmaopt)

    def boundary_dist_Jz0(self) -> np.ndarray:
        print("Jz0 boundary")
        return self.points[:, 1] / (self.lambdaopt * self.sigmaopt)
