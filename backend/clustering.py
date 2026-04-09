
import numpy as np
from sklearn.mixture import GaussianMixture

class GMMClusterer:
    def __init__(self, k: int = 4):
        """
        Gaussian Mixture Model (GMM) Clusterer.
        
        k: number of components (regimes)
        """
        self.k = k
        self.model = GaussianMixture(
            n_components=k, 
            random_state=42, 
            covariance_type='full',
            n_init=1,              
            max_iter=200,          
            reg_covar=1e-6         
        )
        self.is_fitted = False

    def assign(self, x: np.ndarray) -> int:
        """Hard assign x to the most likely cluster index"""
        if not self.is_fitted:
            return 0 
        try:
            return int(self.model.predict(x.reshape(1, -1))[0])
        except Exception as e:
             print(f"Warning: GMM predict failed for input {x}. Returning 0. Error: {e}")
             return 0


    def soft_assign_probs(self, x: np.ndarray) -> np.ndarray:
        """Return soft assignment probabilities for each cluster"""
        if not self.is_fitted:
            return np.ones(self.k) / self.k
        try:
            return self.model.predict_proba(x.reshape(1, -1))[0]
        except Exception as e:
             print(f"Warning: GMM predict_proba failed for input {x}. Returning uniform prob. Error: {e}")
             return np.ones(self.k) / self.k

    def fit_batch(self, X: np.ndarray):
        """Fit the GMM model on the batch data (e.g., training set)"""
        if len(X) < self.k:
             raise ValueError(f"Cannot fit GMM: Number of samples ({len(X)}) is less than number of components ({self.k})")
        print(f"Fitting GMM with {self.k} components on {len(X)} samples...")
        try:
            self.model.fit(X)
            self.is_fitted = True
            print(f"GMM fitting complete. Converged: {self.model.converged_}")
        except Exception as e:
             print(f"ERROR: GMM fitting failed: {e}")
             self.is_fitted = False 

    @property
    def centroids(self) -> np.ndarray:
        """Returns the means of the Gaussian components (clusters)"""
        if self.is_fitted and hasattr(self.model, 'means_'):
            return self.model.means_
        return np.array([])