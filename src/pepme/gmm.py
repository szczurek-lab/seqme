"""
Gaussian Mixture Model wrapper with frozen means.

This module provides a wrapper around scikit-learn's GaussianMixture
that allows initialization from pre-defined means and disables their update during training.
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters, _compute_precision_cholesky
from typing import Optional


class FrozenMeanGMM(GaussianMixture):
    """
    Gaussian Mixture Model with frozen means.
    
    This class inherits from scikit-learn's GaussianMixture and overrides
    the M-step to keep means frozen during training.
    
    Parameters
    ----------
    init_means : np.ndarray
        Initial means for the GMM components. Shape should be (n_components, n_features).
        The number of components will be determined by the first dimension.
    covariance_type : str, default='spherical'
        Type of covariance parameters. Must be one of:
        - 'full': each component has its own general covariance matrix
        - 'tied': all components share the same general covariance matrix
        - 'diag': each component has its own diagonal covariance matrix
        - 'spherical': each component has its own single variance
    random_state : Optional[int], default=None
        Random state for reproducibility.
    max_iter : int, default=100
        Maximum number of iterations for the EM algorithm.
    tol : float, default=1e-3
        Convergence threshold for the EM algorithm.
    verbose : int, default=0
        Forwarded to the base class for logging (kept but default silenced).
    """
    
    def __init__(
        self,
        init_means: np.ndarray,
        covariance_type: str = 'spherical',
        random_state: Optional[int] = None,
        max_iter: int = 100,
        tol: float = 1e-3,
        verbose: int | bool = 0,
    ):
        
        self._frozen_means = init_means.copy()
        
        super().__init__(
            n_components=self._frozen_means.shape[0],
            covariance_type=covariance_type,
            random_state=random_state,
            max_iter=max_iter,
            tol=tol,
            init_params='random',
            warm_start=True,
            verbose=verbose,
            means_init=self._frozen_means,
        )
    
    @property
    def init_means(self) -> np.ndarray:
        """ Ensures compatibility with scikit-learn's GaussianMixture. """
        return self._frozen_means
    
    def _m_step(self, X, log_resp):
        """M step with frozen means.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        print(self.means_)
        self.weights_, _, self.covariances_ = _estimate_gaussian_parameters(
            X, np.exp(log_resp), self.reg_covar, self.covariance_type
        )
        self.weights_ /= self.weights_.sum()
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type
        )
        