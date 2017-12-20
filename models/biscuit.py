"""
Code for the BISCUIT model by Prabhakaran et al 2016
http://proceedings.mlr.press/v48/prabhakaran16.html
"""

# Author: Pedro Ferreira

import sys
import numpy as np
from scipy.stats import multivariate_normal, wishart, invgamma, invwishart, dirichlet, gamma, norm


class Biscuit(object):
    """
    This class allows the fitting of a Hierarchical Conditionally Conjugate Dirichlet Process Mixture Model with
    cell-specific scalings.
    The user has access to the trained parameters: component means, covariances, weights and cell-specific scalings.

    :param n_aux: number of auxiliary variables for the Chinese Restaurant Process
    :param K_init: initial number of clusters
    :param K: maximum number of clusters
    """
    def __init__(self, n_aux=1, K=20, K_init=1,):
        self.n_aux = n_aux
        self.K = K

        self.K_active = K_init

        # Model parameters
        self.pi = 1
        self.mu = 1
        self.cov = 1
        self.cov_inv = 1

        # Assignments
        self.z = 1

        # Identifiability constraints
        self.min_phi = 1
        self.max_phi = 1
        self.min_beta = 1
        self.max_beta = 1

    # Prior distributions
    def sample_prior_hyperparameters(self, X_mean, X_cov, d):
        mulinha = multivariate_normal.rvs(mean=X_mean, cov=X_cov)
        Sigmalinha = invwishart.rvs(df=d, scale=d * X_cov)
        Hlinha = wishart.rvs(df=d, scale=X_cov / d)
        sigmalinha = invgamma.rvs(1, 1 / d) + d
        return mulinha, Sigmalinha, Hlinha, sigmalinha

    def sample_prior_mixture_components(self, mulinha, Sigmalinha, Hlinha, sigmalinha, d, nsamples=1):
        mu = multivariate_normal.rvs(mean=mulinha, cov=Sigmalinha, size=nsamples).reshape(nsamples, d)
        cov_inv = wishart.rvs(df=sigmalinha, scale=np.linalg.inv(Hlinha), size=nsamples).reshape(nsamples, d, d)
        cov = np.linalg.inv(self.cov_inv)
        return mu, cov_inv, cov

    def sample_prior_alpha(self):
        return invgamma.rvs(1, 1)

    def sample_prior_pi(self, alpha):
        self.pi = dirichlet.rvs(alpha / self.K_active * np.ones((self.K_active,))).T

    def sample_prior_cell_scalings(self, ups, delta_sq, omega, theta, N):
        phi = norm.rvs(ups, delta_sq, size=N)
        beta = invgamma.rvs(omega, theta, size=N)
        return phi, beta

    # Posterior distributions
    def update_mixture_components(self, X, mulinha, Sigmalinha, Hlinha, sigmalinha, nk, active_components,
                                  phi, beta):
        K = self.K_active

        Sigmalinha_inv = np.linalg.inv(Sigmalinha)
        for k in range(K):
            k_inds = np.argwhere(self.z == active_components[k]).ravel()
            X_k = X[k_inds]  # all the points in current cluster k
            phi_k = phi[k_inds]
            phi_k = phi_k.reshape(len(phi_k), -1)
            beta_k = beta[k_inds]
            beta_k = beta_k.reshape(len(beta_k), -1)

            covariance_inv = Sigmalinha_inv + self.cov_inv[k].dot(np.sum(phi_k ** 2 / beta_k))
            covariance = np.linalg.inv(covariance_inv)
            mean = covariance.dot(Sigmalinha_inv.dot(mulinha) + self.cov_inv[k].dot(np.sum(X_k / beta_k, axis=0)))
            self.mu[k] = multivariate_normal.rvs(mean=mean, cov=covariance)

            aux = np.dot((X_k - phi_k * self.mu[k]).T, (X_k - phi_k * self.mu[k]) / beta_k)
            self.cov_inv[k] = wishart.rvs(df=int(np.ceil(sigmalinha)) + nk[k] + 1, scale=np.linalg.inv(Hlinha + aux))
            self.cov[k] = np.linalg.inv(self.cov_inv[k])

    def update_hyperparameters(self, X_mean, X_cov_inv, d, mulinha, Sigmalinha, Hlinha, sigmalinha):
        K = self.K_active

        # mulinha
        Sigmalinha_inv = np.linalg.inv(Sigmalinha)
        covariance = np.linalg.inv(X_cov_inv + K * Sigmalinha_inv)
        mean = covariance.dot(K ** 2 * Sigmalinha_inv.dot(np.mean(self.mu, axis=0)) + X_cov_inv.dot(X_mean))
        mulinha = multivariate_normal.rvs(mean=mean, cov=covariance)

        # Sigmalinha
        aux = np.matmul((self.mu - mulinha).T, self.mu - mulinha)
        Sigmalinha = np.linalg.inv(wishart.rvs(df=d + K, scale=np.linalg.inv(d * X_cov_inv + 2 * aux)))

        # Hlinha
        Hlinha = invwishart.rvs(df=d + K * sigmalinha, scale=d * X_cov_inv + np.sum(self.cov_inv, axis=0))

        # sigmalinha
        sigmalinha = invgamma.rvs(1, 1 / d)

        return mulinha, Sigmalinha, Hlinha, sigmalinha

    def cluster_probs_at_point(self, x):
        probs = [(self.pi[k] * multivariate_normal.pdf(x, mean=self.mu[k], cov=self.cov_inv[k])) for k in
                 range(self.K_active)]
        probs = probs / np.sum(probs)
        return probs

    def prior_update_z(self, X, N):
        for n in range(N):
            probs = np.array(self.cluster_probs_at_point(X[n])).ravel()
            self.z[n] = np.random.choice(range(self.K_active), p=probs)

    def update_z_inf(self, X, X_mean, X_cov, d, N, nk, alpha, phi, beta, active_components):
        mulinha_, Sigmalinha_, Hlinha_, sigmalinha_ = self.sample_prior_hyperparameters(X_mean, X_cov, d)
        for n in range(N):
            if nk[np.argwhere(active_components == self.z[n])] == 1:
                self.z[n] = np.random.choice(range(self.K_active, self.K_active + self.n_aux))
                continue

            means, covariance_invs, covariances = self.sample_prior_mixture_components(mulinha_, Sigmalinha_, Hlinha_,
                                                                          sigmalinha_, d, nsamples=self.n_aux)
            # avoid singular matrices!
            if np.linalg.cond(beta[n] * covariances[0]) < 1 / sys.float_info.epsilon:
                means, covariance_invs, covariances = self.sample_prior_mixture_components(mulinha_, Sigmalinha_, Hlinha_,
                                                                              sigmalinha_, d, nsamples=self.n_aux)

            probs = np.ones((self.K_active + self.n_aux,))

            for k_active in range(self.K_active):
                probs[k_active] = \
                    (nk[k_active] - 1) / (N - 1 + alpha) * multivariate_normal.pdf(X[n],
                                                                                   mean=phi[n] * self.mu[k_active],
                                                                                   cov=beta[n] * self.cov[k_active])

            for k_aux in range(self.n_aux):
                probs[self.K_active + k_aux] = \
                    (alpha / self.n_aux) / (N - 1 + alpha) * multivariate_normal.pdf(X[n], mean=phi[n] * means[k_aux],
                                                                                     cov=beta[n] * covariances[k_aux])

            probs = probs / np.sum(probs)

            self.z[n] = np.random.choice(range(self.K_active + self.n_aux), p=probs)

    def update_counts(self):
        nk = np.ones((self.K,))
        for k in range(self.K):
            nk[k] = np.count_nonzero(self.z == k)
        return nk

    def update_pi(self, alpha, nk):
        self.pi =  dirichlet.rvs(alpha / len(nk) * np.ones((len(nk),)) + nk).T

    def get_active_components(self, nk):
        active_clusters = []
        for k in range(self.K):
            if nk[k] > 0:
                active_clusters.append(k)

        self.K_active = len(active_clusters)
        return active_clusters

    def add_new_components(self, active, d):
        if np.any(np.array(active) > self.mu.shape[0] - 1):  # need to grow the vectors!
            new_max = max(active) + 1
            mu_new = np.zeros((new_max, d))
            mu_new[:-1, :] = self.mu
            cov_inv_new = np.zeros((new_max, d, d))
            cov_inv_new[:-1, :, :] = self.cov_inv
            pi_new = np.zeros((new_max, 1))
            pi_new[:-1] = self.pi
            self.mu = mu_new
            self.cov_inv = cov_inv_new
            self.pi = pi_new

    def remove_empty_components(self, active, nk):
        new_nk = nk[active]
        self.mu = self.mu[active]
        self.cov_inv = self.cov_inv[active]
        new_pi = self.pi[active]
        self.pi = new_pi / np.sum(new_pi)
        return new_nk

    def update_alpha(self):
        return invgamma.rvs(1, 1)

    def update_pi(self, alpha, nk):
        return dirichlet.rvs(alpha / len(nk) * np.ones((len(nk),)) + nk).T

    def update_cell_scalings(self, X, N, d, phi, beta, ups, delta_sq, omega, theta, active):
        for n in range(N):
            k = np.argwhere(active == self.z[n])[0][0]

            # Update phi
            A = 1. / np.sqrt(beta[n] * np.abs(self.cov_inv[k]))
            A_mu_k = A.dot(self.mu[k])
            delta_xsq = 1. / np.sum(A_mu_k)

            A_x_j = A.dot(X[n])
            ups_x = delta_xsq * np.sum(A_x_j.dot(A_mu_k))

            delta_psq = np.abs(1. / ((1. / delta_xsq) + 1. / (delta_sq)))
            ups_p = np.abs(delta_psq * (ups_x / delta_xsq + ups / delta_sq))
            phi[n] = norm.rvs(ups_p, delta_psq)

            # identifiability constraint on phi
            if phi[n] < self.min_phi:
                phi[n] = self.min_phi + norm.rvs(0, 0.1)
            elif phi[n] > self.max_phi:
                phi[n] = self.max_phi - norm.rvs(0, 0.1)

            # Update beta
            omega_p = omega + d / 2.
            theta_p = np.abs(theta + 0.5 * np.dot(X[n] - phi[n] * self.mu[k],
                                                  self.cov_inv[k]).dot(X[n] - phi[n] * self.mu[k]))

            beta[n] = invgamma.rvs(omega_p, theta_p)

            # identifiability constraint on beta
            if beta[n] < self.min_beta:
                beta[n] = self.min_beta + invgamma.rvs(omega_p, 1. / omega_p)
            elif beta[n] > self.max_beta:
                beta[n] = np.abs(self.max_beta - invgamma.rvs(omega_p, 1. / omega_p))

        return phi, beta

    def fit(self, X, n_iterations=100, phi_hyperparams=[0, 1], beta_hyperparams=[1, 1], verbose=False):
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
        phi = np.zeros((N, ))
        beta = np.zeros((N, ))

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
        phi, beta = self.sample_prior_cell_scalings(ups, delta_sq, omega, theta, N)

        for i in range(0, n_iterations):
            # Sampling from the conditional posteriors
            alpha = self.update_alpha()
            self.update_pi(alpha, nk)

            self.update_mixture_components(X, mulinha, Sigmalinha, Hlinha, sigmalinha, nk,
                                           active_components, phi, beta)

            phi, beta = self.update_cell_scalings(X, N, d, phi, beta, ups, delta_sq, omega, theta, active_components)

            # the hyperparameters are the same for all mixture components
            mulinha, Sigmalinha, Hlinha, sigmalinha = self.update_hyperparameters(X_mean, X_cov_inv, d,
                                                                            mulinha, Sigmalinha, Hlinha, sigmalinha)

            self.update_z_inf(X, X_mean, X_cov, d, N, nk, alpha, phi, beta, active_components)
            nk = self.update_counts()
            active_components = self.get_active_components(nk)
            self.add_new_components(active_components, d)
            nk = self.remove_empty_components(active_components, nk)  # The parameter vectors are now of length K_active

            print(i)
