"""
Code for the Hierarchical Conditionally Conjugate DPGMM by Gorur and Rasmussen, 2010
http://mlg.eng.cam.ac.uk/pub/pdf/GoeRas10.pdf
"""

# Author: Pedro Ferreira

import sys
import numpy as np
from scipy.stats import multivariate_normal, wishart, invgamma, invwishart, dirichlet
from tqdm import tqdm as tqdm


class DPGMM(object):
    """
    This class allows the fitting of a Hierarchical Conditionally Conjugate Dirichlet Process Mixture Model with
    cell-specific scalings.
    The user has access to the trained parameters: component means, covariances and weights.

    :param n_aux: number of auxiliary variables for the Chinese Restaurant Process
    :param K_init: initial number of clusters
    :param K: maximum number of clusters
    """
    def __init__(self, n_aux=1, K=20, K_init=1,):
        self.n_aux = n_aux
        self.K = K

        self.K_active = K_init

        # Model parameters
        self.pi = np.ones((1, ))
        self.mu = np.ones((1, 1))
        self.cov = np.ones((1, 1, 1))
        self.cov_inv = np.ones((1, 1, 1))

        # Assignments
        self.z = np.ones((1, ))

    def log_likelihood(self, X):
        """
        Data log-likelihood
        Equation 2.153 in Sudderth
        """
        ll = 0
        for i in range(X.shape[0]):
            probs = 0
            for k in range(self.K_active):
                probs = probs + self.pi[k] * multivariate_normal.pdf(X[i], mean=self.mu[k], cov=self.cov[k])
            ll = ll + np.log10(probs)

        return ll

    # Prior distributions
    def sample_prior_hyperparameters(self, X_mean, X_cov, d):
        mulinha = multivariate_normal.rvs(mean=X_mean, cov=X_cov)
        Sigmalinha = invwishart.rvs(df=d, scale=d * X_cov)
        Hlinha = wishart.rvs(df=d, scale=X_cov / d)
        while np.all(np.linalg.eigvals(Hlinha) > 0) is False:
            Hlinha = wishart.rvs(df=d, scale=X_cov / d)
        sigmalinha = invgamma.rvs(1, 1 / d) + d
        return mulinha, Sigmalinha, Hlinha, sigmalinha

    def sample_prior_mixture_components(self, mulinha, Sigmalinha, Hlinha, sigmalinha, d, nsamples=1):
        mu = multivariate_normal.rvs(mean=mulinha, cov=Sigmalinha, size=nsamples).reshape(nsamples, d)
        cov = wishart.rvs(df=sigmalinha, scale=Hlinha, size=nsamples).reshape(nsamples, d, d)
        cov_inv = np.linalg.inv(cov)
        return mu, cov_inv, cov

    def sample_prior_alpha(self):
        return invgamma.rvs(1, 1)

    def sample_prior_pi(self, alpha):
        self.pi = dirichlet.rvs(alpha / self.K_active * np.ones((self.K_active,))).T

    # Posterior distributions
    def update_mixture_components(self, X, mulinha, Sigmalinha, Hlinha, sigmalinha, nk, active_components):
        K = self.K_active

        for k in range(K):  # update all K_active components
            X_k = X[np.argwhere(self.z == active_components[k]).ravel()]  # all the points in current cluster k

            Sigmaklinha_inv = np.linalg.inv(Sigmalinha) + nk[k] * self.cov_inv[k]
            Sigmaklinha = np.linalg.inv(Sigmaklinha_inv)
            muklinha = Sigmaklinha.dot(
                np.linalg.inv(Sigmalinha).dot(mulinha) + nk[k] * self.cov_inv[k].dot(np.mean(X_k, axis=0)))
            aux = np.matmul((X_k - self.mu[k]).T, (X_k - self.mu[k]))

            self.mu[k] = multivariate_normal.rvs(mean=muklinha, cov=Sigmaklinha)
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

    def update_z_inf(self, X, X_mean, X_cov, d, N, nk, alpha, active_components):
        mulinha_, Sigmalinha_, Hlinha_, sigmalinha_ = self.sample_prior_hyperparameters(X_mean, X_cov, d)
        for n in range(N):
            if nk[np.argwhere(active_components == self.z[n])] == 1:
                self.z[n] = np.random.choice(range(self.K_active, self.K_active + self.n_aux))
                continue

            means, covariance_invs, covariances = self.sample_prior_mixture_components(mulinha_, Sigmalinha_, Hlinha_,
                                                                          sigmalinha_, d, nsamples=self.n_aux)
            # avoid singular matrices!
            if np.linalg.cond(covariances[0]) < 1 / sys.float_info.epsilon:
                means, covariance_invs, covariances = self.sample_prior_mixture_components(mulinha_, Sigmalinha_, Hlinha_,
                                                                              sigmalinha_, d, nsamples=self.n_aux)

            probs = np.ones((self.K_active + self.n_aux,))

            for k_active in range(self.K_active):
                probs[k_active] = \
                    (nk[k_active] - 1) / (N - 1 + alpha) * multivariate_normal.pdf(X[n], mean=self.mu[k_active],
                                                                                   cov=self.cov[k_active])

            for k_aux in range(self.n_aux):
                probs[self.K_active + k_aux] = \
                    (alpha / self.n_aux) / (N - 1 + alpha) * multivariate_normal.pdf(X[n], mean=means[k_aux],
                                                                                     cov=covariances[k_aux])

            probs = probs / np.sum(probs)

            self.z[n] = np.random.choice(range(self.K_active + self.n_aux), p=probs)

    def update_counts(self):
        nk = np.ones((self.K,))
        for k in range(self.K):
            nk[k] = np.count_nonzero(self.z == k)
        return nk

    def update_pi(self, alpha, nk):
        self.pi = dirichlet.rvs(alpha / len(nk) * np.ones((len(nk),)) + nk).T

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
            cov_new = np.zeros((new_max, d, d))
            cov_new[:-1, :, :] = self.cov
            pi_new = np.zeros((new_max, 1))
            pi_new[:-1] = self.pi
            self.mu = mu_new
            self.cov_inv = cov_inv_new
            self.cov = cov_new
            self.pi = pi_new

    def remove_empty_components(self, active, nk):
        new_nk = nk[active]
        self.mu = self.mu[active]
        self.cov_inv = self.cov_inv[active]
        self.cov = self.cov[active]
        new_pi = self.pi[active]
        self.pi = new_pi / np.sum(new_pi)
        return new_nk

    def update_alpha(self):
        return invgamma.rvs(1, 1)

    def fit(self, X, n_iterations=100, print_log_likelihood=False, verbose=False):
        N = X.shape[0]  # number of samples
        d = X.shape[1]  # data dimensionality

        # Model parameters
        self.pi = np.zeros((self.K,))
        self.mu = np.zeros((self.K, d))
        self.cov = np.zeros((self.K, d, d))
        self.cov_inv = np.zeros((self.K, d, d))

        # Assignments
        self.z = np.zeros((N, ))

        X_mean = np.mean(X, axis=0)
        X_cov = np.cov(X.T)
        X_cov_inv = np.linalg.inv(X_cov)

        mulinha, Sigmalinha, Hlinha, sigmalinha = self.sample_prior_hyperparameters(X_mean, X_cov, d)
        self.mu, self.cov_inv, self.cov = self.sample_prior_mixture_components(mulinha, Sigmalinha, Hlinha, sigmalinha,
                                                                               d, nsamples=self.K_active)
        alpha = self.sample_prior_alpha()
        self.sample_prior_pi(alpha)
        self.prior_update_z(X, N)
        nk = self.update_counts()
        active_components = self.get_active_components(nk)
        nk = self.remove_empty_components(active_components, nk)

        for _ in tqdm(range(0, n_iterations)):
            # Sampling from the conditional posteriors
            alpha = self.update_alpha()

            # the hyperparameters are the same for all mixture components
            mulinha, Sigmalinha, Hlinha, sigmalinha = self.update_hyperparameters(X_mean, X_cov_inv, d, mulinha,
                                                                                  Sigmalinha, Hlinha, sigmalinha)

            self.update_z_inf(X, X_mean, X_cov, d, N, nk, alpha, active_components)
            nk = self.update_counts()
            active_components = self.get_active_components(nk)
            self.add_new_components(active_components, d)
            nk = self.remove_empty_components(active_components, nk)  # The parameter vectors are now of length K_active

            self.update_pi(alpha, nk)

            self.update_mixture_components(X, mulinha, Sigmalinha, Hlinha, sigmalinha, nk, active_components)

            if print_log_likelihood:
                print(self.log_likelihood(X))

    def sample(self, n_samples=1):
        d = self.mu.shape[1]

        z = np.ones((n_samples,))
        X = np.zeros((n_samples, d))

        for n in range(n_samples):
            # select one of the clusters
            k = np.random.choice(range(self.K_active), p=self.pi.ravel())
            z[n] = k

            # sample an observation
            X[n] = multivariate_normal.rvs(mean=self.mu[k], cov=self.cov[k])

        return X, z

    def set_random_parameters(self, X, K):
        X_mean = np.mean(X, axis=0)
        X_cov = np.cov(X.T)
        d = X.shape[1]

        # Hyperparameters
        mulinha = multivariate_normal.rvs(mean=X_mean, cov=X_cov)
        Sigmalinha = invwishart.rvs(df=d, scale=d * X_cov)
        Hlinha = wishart.rvs(df=d, scale=X_cov / d)
        sigmalinha = invgamma.rvs(1, 1 / d) + d

        # Parameters
        self.K_active = K
        self.mu = multivariate_normal.rvs(mean=mulinha, cov=Sigmalinha, size=K).reshape(K, d)
        self.cov_inv = wishart.rvs(df=sigmalinha, scale=np.linalg.inv(Hlinha), size=K).reshape(K, d, d)
        self.cov = np.linalg.inv(self.cov_inv)

        alpha = invgamma.rvs(1, 1)
        self.pi = dirichlet.rvs(alpha / self.K_active * np.ones((self.K_active,))).T
