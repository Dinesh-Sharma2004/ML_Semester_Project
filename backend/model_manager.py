
import numpy as np
import lightgbm as lgb
from typing import List, Dict
import joblib
import os
import warnings


class RegimeModelManager:
    def __init__(self, n_regimes: int = 4, feature_dim: int = 3, model_dir: str = "models"):
        self.n_regimes = n_regimes
        self.feature_dim = feature_dim
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.lgbm_params = {
            'objective': 'regression_l1',
            'metric': 'mae',
            'n_estimators': 100,
            'learning_rate': 0.05, 
            'num_leaves': 31,
            'verbose': -1,
            'n_jobs': 1,  
            'random_state': 42,
            'boosting_type': 'gbdt'
        }

        self.global_model = lgb.LGBMRegressor(**self.lgbm_params)
        self.global_initialized = False

        # per-regime models
        self.regime_models: List[lgb.LGBMRegressor] = []
        self.regime_initialized = [False] * n_regimes
        for i in range(n_regimes):
            m = lgb.LGBMRegressor(**self.lgbm_params)
            self.regime_models.append(m)
        self.buffers: List[List] = [[] for _ in range(n_regimes)]
        self.buffer_max = 200    
        self.update_counters = [0] * n_regimes
        self.update_freq = 50     

    def train_global(self, X: np.ndarray, y: np.ndarray):
        print(f"Training global LGBM model on {len(X)} samples...")
        try:
            self.global_model.fit(X, y)
            self.global_initialized = True
            joblib.dump(self.global_model, os.path.join(self.model_dir, "global_model.pkl"))
            print("Global model training complete.")
        except Exception as e:
            print(f"ERROR: Global model training failed: {e}")
            self.global_initialized = False


    def train_regime_models_batch(self, X_train: np.ndarray, y_train: np.ndarray, assignments: np.ndarray):
        print("Batch training regime-specific LGBM models...")
        for i in range(self.n_regimes):
            regime_indices = np.where(assignments == i)[0]
            if len(regime_indices) > self.feature_dim * 2: 
                regime_X = X_train[regime_indices]
                regime_y = y_train[regime_indices]
                
                try:
                    self.regime_models[i].fit(regime_X, regime_y)
                    self.regime_initialized[i] = True
                    print(f"  - Regime {i} trained on {len(regime_X)} samples.")
                except Exception as e:
                    print(f"ERROR: Training failed for regime {i} on {len(regime_X)} samples: {e}")
                    self.regime_initialized[i] = False
            else:
                self.regime_initialized[i] = False
                print(f"  - Regime {i} has insufficient data ({len(regime_indices)} samples), will use global model.")

    def predict(self, regime_idx: int, x: np.ndarray) -> float:
        """Predict using regime model if available else global fallback."""
        x2 = np.asarray(x).reshape(1, -1)
        if regime_idx >= 0 and regime_idx < self.n_regimes and self.regime_initialized[regime_idx]:
            try:
                return float(self.regime_models[regime_idx].predict(x2)[0])
            except Exception as e:
                 print(f"Warning: Prediction failed for regime {regime_idx}. Falling back. Error: {e}")
                 
        if self.global_initialized:
            try:
                return float(self.global_model.predict(x2)[0])
            except Exception as e:
                 print(f"Warning: Global model prediction failed. Falling back to mean. Error: {e}")
                 
        return float(np.mean(x)) if x.size > 0 else 0.0


    def partial_update(self, regime_idx: int, x: np.ndarray, y: float):
        """
        Online update: Add to buffer. Periodically retrain the model on the
        buffer, using the previous model as init_model.
        """
        if regime_idx < 0 or regime_idx >= self.n_regimes:
            return 
            
        buf = self.buffers[regime_idx]
        buf.append((x, y))
        if len(buf) > self.buffer_max:
            buf.pop(0)

        self.update_counters[regime_idx] += 1
        if self.update_counters[regime_idx] % self.update_freq == 0 and len(buf) > self.feature_dim:
            model = self.regime_models[regime_idx]
            X_buf = np.array([item[0] for item in buf]).astype('float32')
            y_buf = np.array([item[1] for item in buf]).astype('float32')

            try:
                model.fit(X_buf, y_buf, init_model=model if self.regime_initialized[regime_idx] else None)
                self.regime_initialized[regime_idx] = True 
            except Exception as e:
                print(f"ERROR: Online update failed for regime {regime_idx}: {e}")
            


    def save_models(self):
        if self.global_initialized:
            try:
                 joblib.dump(self.global_model, os.path.join(self.model_dir, "global_model.pkl"))
            except Exception as e:
                 print(f"Warning: Could not save global model: {e}")
                 
        for i, m in enumerate(self.regime_models):
            if self.regime_initialized[i]:
                try:
                    joblib.dump(m, os.path.join(self.model_dir, f"regime_model_{i}.pkl"))
                except Exception as e:
                    print(f"Warning: Could not save model for regime {i}: {e}")


    def load_models(self):
        gpath = os.path.join(self.model_dir, "global_model.pkl")
        if os.path.exists(gpath):
            try:
                self.global_model = joblib.load(gpath)
                self.global_initialized = True
                print("Loaded existing global model.")
            except Exception as e:
                 print(f"Warning: Could not load global model from {gpath}. Error: {e}")
                 self.global_initialized = False
                 
        for i in range(self.n_regimes):
            p = os.path.join(self.model_dir, f"regime_model_{i}.pkl")
            if os.path.exists(p):
                 try:
                    self.regime_models[i] = joblib.load(p)
                    self.regime_initialized[i] = True
                    print(f"Loaded existing model for regime {i}.")
                 except Exception as e:
                     print(f"Warning: Could not load regime model {i} from {p}. Error: {e}")
                     self.regime_initialized[i] = False