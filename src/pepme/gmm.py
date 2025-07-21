"""
Gaussian Mixture Model wrapper with frozen means.

This module provides a wrapper around scikit-learn's GaussianMixture
that allows initialization from pre-defined means and disables their update during training.
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky, _estimate_gaussian_parameters


class FrozenMeanGMM(GaussianMixture):
    """Gaussian Mixture Model with frozen means.

    This class inherits from scikit-learn's GaussianMixture and overrides
    the M-step to keep means frozen during training.

    Args:
        init_means: Initial means for the GMM components. Shape should be (n_components, n_features).
            The number of components will be determined by the first dimension.
        covariance_type: Type of covariance parameters. Must be one of:
            'full', 'tied', 'diag', or 'spherical'.
        random_state: Random state for reproducibility.
        max_iter: Maximum number of iterations for the EM algorithm.
        tol: Convergence threshold for the EM algorithm.
        verbose: Forwarded to the base class for logging (kept but default silenced).
    """

    def __init__(
        self,
        init_means: np.ndarray,
        covariance_type: str = "spherical",
        random_state: int | None = None,
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
            init_params="random",
            warm_start=True,
            verbose=verbose,
            means_init=self._frozen_means,
        )

    @property
    def init_means(self) -> np.ndarray:
        """Ensure compatibility with scikit-learn's GaussianMixture."""
        return self._frozen_means

    def _m_step(self, X, log_resp):
        """M step with frozen means.

        Args:
            X: Array-like of shape (n_samples, n_features).
            log_resp: Array-like of shape (n_samples, n_components).
                Logarithm of the posterior probabilities (or responsibilities) of
                the point of each sample in X.
        """
        self.weights_, _, self.covariances_ = _estimate_gaussian_parameters(
            X, np.exp(log_resp), self.reg_covar, self.covariance_type
        )
        self.weights_ /= self.weights_.sum()
        self.precisions_cholesky_ = _compute_precision_cholesky(self.covariances_, self.covariance_type)
        assert np.allclose(self.means_, self._frozen_means), "GMM means are not equal to the frozen means"
        return self

    def compute_log_likelihood(self, X: np.ndarray, dim_adjusted: bool = True) -> np.ndarray:
        """Compute the log likelihood of the data under the GMM.

        Args:
            X: Array-like of shape (n_samples, n_features).
                The data to compute the log likelihood of.

        Returns:
            Array-like of shape (n_samples,). The log likelihood of the data under the GMM.
        """
        return self.score_samples(X) / X.shape[1] if dim_adjusted else self.score_samples(X)

    def compute_negative_log_likelihood(self, X: np.ndarray, dim_adjusted: bool = True) -> np.ndarray:
        """Compute the negative log likelihood of the data under the GMM.

        Args:
            X: Array-like of shape (n_samples, n_features).
                The data to compute the negative log likelihood of.

        Returns:
            Array-like of shape (n_samples,). The negative log likelihood of the data under the GMM.
        """
        return (-1) * self.compute_log_likelihood(X, dim_adjusted)

    def compute_pairwise_log_likelihoods(self, reference_embeddings: np.ndarray) -> np.ndarray:
        """Return an n x m matrix of pairwise log likelihoods.

        Compute an n x m matrix where n is the number of samples and m is the number of GMM components.
        An entry a_ij of this matrix is such that a_ij = log[N(x_i|x_j, sigma_j)].

        Args:
            reference_embeddings: Reference embeddings to compute pairwise log likelihoods for.

        Returns:
            Pairwise log likelihood matrix of shape (n_samples, n_components).
        """
        return self._estimate_log_prob(reference_embeddings)
    
    def _compute_pairwise_negative_log_likelihoods(self, reference_embeddings: np.ndarray) -> np.ndarray:
        """Compute the negative log likelihood of the data under the GMM.

        Args:
            X: Array-like of shape (n_samples, n_features).
                The data to compute the negative log likelihood of.

        Returns:
            Array-like of shape (n_samples, n_components). The negative log likelihood of the data under the GMM.
        """
        return (-1) * self._estimate_log_prob(reference_embeddings)
    
    @property
    def log_sigmas_(self) -> np.ndarray:
        """Compute the log of the standard deviations of the GMM.

        Returns:
            Log standard deviations for each GMM component.
        """
        return np.log(np.sqrt(self.covariances_))
    