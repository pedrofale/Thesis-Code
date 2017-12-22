from scipy.stats import _multivariate
from scipy.linalg import eigh

import numpy as np


def matrix_is_well_conditioned(M):
    err = True

    s, u = eigh(M, lower=True, check_finite=True)
    eps = _multivariate._eigvalsh_to_eps(s, None, None)
    if np.min(s) < -eps:
        err = False
    d = s[s > eps]
    if len(d) < len(s):
        err = False

    return err


def matrix_is_psd(M):
    err = True

    s, u = eigh(M, lower=True, check_finite=True)
    eps = _multivariate._eigvalsh_to_eps(s, None, None)
    if np.min(s) < -eps:
        err = False

    return err


def matrix_is_not_singular(M):
    err = True

    s, u = eigh(M, lower=True, check_finite=True)
    eps = _multivariate._eigvalsh_to_eps(s, None, None)

    d = s[s > eps]
    if len(d) < len(s):
        err = False

    return err