"""
Code for the BISCUIT model by Prabhakaran et al 2016
http://proceedings.mlr.press/v48/prabhakaran16.html
"""

# Author: Pedro Ferreira

from models.dpgmm import DPGMM

import numpy as np
from scipy.stats import multivariate_normal, wishart, invgamma, norm
from tqdm import tqdm


class BISCUIT(DPGMM):
    """
    This class allows the fitting of a Hierarchical Conditionally Conjugate Dirichlet Process Mixture Model with
    cell-specific scalings.
    The user has access to the trained parameters: component means, covariances, weights and cell-specific scalings.

    """
    def __init__(self, **kwargs):
        DPGMM.__init__(self, **kwargs)

        # Cell-specific scaling parameters
        self.phi = 1
        self.beta = 1

        # Identifiability constraints for scaling parameters
        self.min_phi = 1
        self.max_phi = 1
        self.min_beta = 1
        self.max_beta = 1

    def sample_prior_cell_scalings(self, ups, delta_sq, omega, theta, N):
        self.phi = norm.rvs(ups, delta_sq, size=N)
        self.beta = invgamma.rvs(omega, theta, size=N)

    # Posterior distributions
    def update_mixture_components(self, X, mulinha, Sigmalinha, Hlinha, sigmalinha, nk, active_components):
        K = self.K_active

        Sigmalinha_inv = np.linalg.inv(Sigmalinha)
        for k in range(K):
            k_inds = np.argwhere(self.z == active_components[k]).ravel()
            X_k = X[k_inds]  # all the points in current cluster k
            phi_k = self.phi[k_inds]
            phi_k = phi_k.reshape(len(phi_k), -1)
            beta_k = self.beta[k_inds]
            beta_k = beta_k.reshape(len(beta_k), -1)

            covariance_inv = Sigmalinha_inv + self.cov_inv[k].dot(np.sum(phi_k ** 2 / beta_k))
            covariance = np.linalg.inv(covariance_inv)
            mean = covariance.dot(Sigmalinha_inv.dot(mulinha) + self.cov_inv[k].dot(np.sum(X_k / beta_k, axis=0)))
            self.mu[k] = multivariate_normal.rvs(mean=mean, cov=covariance)

            aux = np.dot((X_k - phi_k * self.mu[k]).T, (X_k - phi_k * self.mu[k]) / beta_k)
            self.cov_inv[k] = wishart.rvs(df=int(np.ceil(sigmalinha)) + nk[k] + 1, scale=np.linalg.inv(Hlinha + aux))
            self.cov[k] = np.linalg.inv(self.cov_inv[k])

    def update_z_inf(self, X, X_mean, X_cov, d, N, nk, alpha, active_components):
        mulinha_, Sigmalinha_, Hlinha_, sigmalinha_ = self.sample_prior_hyperparameters(X_mean, X_cov, d)
        for n in range(N):
            if nk[np.argwhere(active_components == self.z[n])] == 1:
                self.z[n] = np.random.choice(range(self.K_active, self.K_active + self.n_aux))
                continue

            means, covariance_invs, covariances = self.sample_prior_mixture_components(mulinha_, Sigmalinha_, Hlinha_,
                                                                          sigmalinha_, d, nsamples=self.n_aux)
            # avoid singular matrices!
            while np.all(np.linalg.eigvals(self.beta[n] * covariances[0]) > 0) is False:
                mulinha_, Sigmalinha_, Hlinha_, sigmalinha_ = self.sample_prior_hyperparameters(X_mean, X_cov, d)
                means, covariance_invs, covariances = self.sample_prior_mixture_components(mulinha_, Sigmalinha_, Hlinha_,
                                                                              sigmalinha_, d, nsamples=self.n_aux)

            probs = np.ones((self.K_active + self.n_aux,))

            for k_active in range(self.K_active):
                probs[k_active] = \
                    (nk[k_active] - 1) / (N - 1 + alpha) * multivariate_normal.pdf(X[n],
                                                                                   mean=self.phi[n] * self.mu[k_active],
                                                                                   cov=self.beta[n] * self.cov[k_active])

            for k_aux in range(self.n_aux):
                probs[self.K_active + k_aux] = \
                    (alpha / self.n_aux) / (N - 1 + alpha) * multivariate_normal.pdf(X[n], mean=self.phi[n] * means[k_aux],
                                                                                     cov=self.beta[n] * covariances[k_aux])

            probs = probs / np.sum(probs)

            self.z[n] = np.random.choice(range(self.K_active + self.n_aux), p=probs)

    def update_cell_scalings(self, X, N, d, ups, delta_sq, omega, theta, active):
        for n in range(N):
            k = np.argwhere(active == self.z[n])[0][0]

            # Update phi
            A = 1. / np.sqrt(self.beta[n] * np.abs(self.cov_inv[k]))
            A_mu_k = A.dot(self.mu[k])
            delta_xsq = 1. / np.sum(A_mu_k)

            A_x_j = A.dot(X[n])
            ups_x = delta_xsq * np.sum(A_x_j.dot(A_mu_k))

            delta_psq = np.abs(1. / ((1. / delta_xsq) + 1. / (delta_sq)))
            ups_p = np.abs(delta_psq * (ups_x / delta_xsq + ups / delta_sq))
            self.phi[n] = norm.rvs(ups_p, delta_psq)

            # identifiability constraint on phi
            if self.phi[n] < self.min_phi:
                self.phi[n] = self.min_phi + norm.rvs(0, 0.1)
            elif self.phi[n] > self.max_phi:
                self.phi[n] = self.max_phi - norm.rvs(0, 0.1)

            # Update beta
            omega_p = omega + d / 2.
            theta_p = np.abs(theta + 0.5 * np.dot(X[n] - self.phi[n] * self.mu[k],
                                                  self.cov_inv[k]).dot(X[n] - self.phi[n] * self.mu[k]))

            self.beta[n] = invgamma.rvs(omega_p, theta_p)

            # identifiability constraint on beta
            if self.beta[n] < self.min_beta:
                self.beta[n] = self.min_beta + invgamma.rvs(omega_p, 1. / omega_p)
            elif self.beta[n] > self.max_beta:
                self.beta[n] = np.abs(self.max_beta - invgamma.rvs(omega_p, 1. / omega_p))

    def fit(self, X, n_iterations=100, n_burnin=50, phi_hyperparams=[0, 1], beta_hyperparams=[1, 1],
            return_cm=False, print_log_likelihood=False, verbose=False):

        N = X.shape[0]  # number of samples
        d = X.shape[1]  # data dimensionality

        # Model parameters
        self.pi = np.zeros((self.K,))
        self.mu = np.zeros((self.K, d))
        self.cov = np.zeros((self.K, d, d))
        self.cov_inv = np.zeros((self.K, d, d))

        # Assignments
        self.z = np.zeros((N, ))

        # Cell-specific moment scaling parameters
        self.phi = np.zeros((N, ))
        self.beta = np.zeros((N, ))

        if return_cm:
            cm = np.ones((N, N))

        X_mean = np.mean(X, axis=0)
        X_cov = np.cov(X.T)
        X_cov_inv = np.linalg.inv(X_cov)

        ups = phi_hyperparams[0]
        delta_sq = phi_hyperparams[1]
        omega = beta_hyperparams[0]
        theta = beta_hyperparams[1]

        mulinha, Sigmalinha, Hlinha, sigmalinha = self.sample_prior_hyperparameters(X_mean, X_cov, d)
        self.mu, self.cov_inv, self.cov = self.sample_prior_mixture_components(mulinha, Sigmalinha, Hlinha, sigmalinha, d, nsamples=self.K_active)
        alpha = self.sample_prior_alpha()
        self.sample_prior_pi(alpha)
        self.prior_update_z(X, N)
        nk = self.update_counts()
        active_components = self.get_active_components(nk)
        nk = self.remove_empty_components(active_components, nk)
        # The parameter vectors are now of length self.K_active
        self.sample_prior_cell_scalings(ups, delta_sq, omega, theta, N)

        for i in tqdm(range(0, n_iterations)):
            # Sampling from the conditional posteriors
            alpha = self.update_alpha()
            self.update_pi(alpha, nk)

            # the hyperparameters are the same for all mixture components
            mulinha, Sigmalinha, Hlinha, sigmalinha = self.update_hyperparameters(X_mean, X_cov_inv, d,
                                                                                  mulinha, Sigmalinha, Hlinha,
                                                                                  sigmalinha)

            self.update_z_inf(X, X_mean, X_cov, d, N, nk, alpha, active_components)
            nk = self.update_counts()
            active_components = self.get_active_components(nk)
            self.add_new_components(active_components, d)
            nk = self.remove_empty_components(active_components, nk)  # The parameter vectors are now of length K_active

            self.update_mixture_components(X, mulinha, Sigmalinha, Hlinha, sigmalinha, nk, active_components)

            self.update_cell_scalings(X, N, d, ups, delta_sq, omega, theta, active_components)

            if return_cm and i > n_burnin:
                self.update_confusion_matrix(cm)

            if print_log_likelihood:
                print(self.log_likelihood(X))

        if return_cm:
            return cm

    def sample(self, n_samples=1, sort=False):
        ups = 0
        delta_sq = 2
        omega = 1
        theta = 1

        d = self.mu.shape[1]

        self.z = np.ones((n_samples,))
        phi = np.ones((n_samples,))
        beta = np.ones((n_samples,))
        X = np.zeros((n_samples, d))

        for n in range(n_samples):
            # select one of the clusters
            k = np.random.choice(range(self.K_active), p=self.pi.ravel())
            self.z[n] = k

            # sample scaling parameters
            phi[n] = norm.rvs(ups, delta_sq)
            beta[n] = invgamma.rvs(omega, theta)

            # sample an observation
            X[n] = multivariate_normal.rvs(mean=phi[n] * self.mu[k], cov=beta[n] * self.cov[k])

        if sort:
            ind = np.argsort(self.z)
            self.z = self.z[ind]
            X = X[ind]

        return X