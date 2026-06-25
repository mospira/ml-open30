import numpy as np
import pandas as pd
from typing import Dict, List, Type, Any, Optional
import pickle
import os

class MultiHeadModel:
    """
    Manages a distinct classification model for each reward multiple `m`.
    Each sub-model predicts the probabilities of 3 mutually exclusive target 
    outcomes: TP (Take Profit), SL (Stop Loss), TIME (Timeout) within the 30-min window.
    """
    def __init__(
        self, 
        rr_multiples: List[float], 
        base_estimator_class: Type, 
        estimator_params: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            rr_multiples: List of reward multiples to build heads for (e.g., [0.5, 1.0, 1.5, 2.0]).
            base_estimator_class: The uninstantiated sklearn-compatible classifier class 
                to use for each head (e.g., sklearn.ensemble.RandomForestClassifier, xgboost.XGBClassifier).
            estimator_params: Hyperparameters to pass to the base estimator upon instantiation.
        """
        self.rr_multiples = rr_multiples
        self.base_estimator_class = base_estimator_class
        self.estimator_params = estimator_params or {}
        
        # Dictionary mapping reward multiple `m` to its instantiated fitted model
        self.models: Dict[float, Any] = {}
        
        # Instantiate the models
        for m in self.rr_multiples:
            self.models[m] = self.base_estimator_class(**self.estimator_params)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, sample_weights: Optional[np.ndarray] = None, xgb_model_dict: Optional[Dict[float, Any]] = None):
        """
        Trains each sub-model on the subset of data where the outcome was NOT ambiguous for that `m`.

        Args:
            X: DataFrame of precalculated features.
            y: DataFrame of target labels containing columns `y_type_m_{m}` and `y_ambig_m_{m}`.
            sample_weights: Optional array of sample weights.
            xgb_model_dict: Optional dictionary mapping `m` values to a previously trained XGBoost tree Booster
                            to enable incremental learning.
        """
        print(f"Training MultiHeadModel across {len(self.rr_multiples)} reward multiples...")
        
        for m in self.rr_multiples:
            ambig_col = f"y_ambig_m_{m}"
            type_col = f"y_type_m_{m}"
            
            if ambig_col not in y.columns or type_col not in y.columns:
                raise ValueError(f"Missing required label columns for m={m}: {ambig_col} or {type_col}")
            
            # Filter out ambiguous rows based on the DROP policy defined in configs/labels.yaml
            # 1 means True (ambiguous), 0 means False.
            valid_mask = y[ambig_col] != True
            
            X_valid = X[valid_mask]
            y_valid = y.loc[valid_mask, type_col]
            
            dropped_count = (~valid_mask).sum()
            print(f"  [m={m}] Dropped {dropped_count} ambiguous rows. Training on {len(X_valid)} instances.")
            
            prior_model = xgb_model_dict.get(m) if xgb_model_dict is not None else None
            
            # Fit the model for this specific multiple
            if sample_weights is not None:
                sw_valid = sample_weights[valid_mask]
                self.models[m].fit(X_valid, y_valid, sample_weight=sw_valid, xgb_model=prior_model)
            else:
                self.models[m].fit(X_valid, y_valid, xgb_model=prior_model)
            
        print("Training complete.\n")
        return self

    def predict_proba(self, X: pd.DataFrame) -> Dict[float, np.ndarray]:
        """
        Predicts the class probabilities (SL, TP, TIME) for each reward multiple `m`.

        Note: Standard classification order assumes classes are sorted (0, 1, 2). 
        Based on configs/labels.yaml: 0=SL, 1=TP, 2=TIME.

        Args:
            X: feature DataFrame.

        Returns:
            Dictionary mapping `m` -> (N_samples x 3_classes) array of probabilities.
        """
        probas = {}
        for m in self.rr_multiples:
            # Output will be N x 3
            probas[m] = self.models[m].predict_proba(X)
        return probas
    
    def predict(self, X: pd.DataFrame) -> Dict[float, np.ndarray]:
        """
        Predicts the discrete class label (0, 1, or 2) for each reward multiple `m`.
        """
        preds = {}
        for m in self.rr_multiples:
            preds[m] = self.models[m].predict(X)
        return preds

    def save(self, directory: str, filename: str = "multi_head_model.pkl"):
        """Saves the ensemble to disk."""
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, filename)
        with open(path, "wb") as f:
            pickle.dump({
                'rr_multiples': self.rr_multiples,
                'base_estimator_class': self.base_estimator_class,
                'estimator_params': self.estimator_params,
                'models': self.models
            }, f)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, filepath: str) -> "MultiHeadModel":
        """Loads a saved MultiHeadModel from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            
        obj = cls(
            rr_multiples=data['rr_multiples'], 
            base_estimator_class=data['base_estimator_class'],
            estimator_params=data['estimator_params']
        )
        obj.models = data['models']
        return obj

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Quick Unit Test / Verification
    # ------------------------------------------------------------------
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    
    print("Running quick verification test...")
    
    # 1. Generate dummy data
    N = 1000
    np.random.seed(42)
    
    # Random features
    X_dummy = pd.DataFrame({
        'feature1': np.random.randn(N),
        'feature2': np.random.randn(N),
    })
    
    # Random labels for m=1.0 and m=2.0
    rr_multiples = [1.0, 2.0]
    y_dummy = pd.DataFrame()
    for m in rr_multiples:
        # Classes: 0=SL, 1=TP, 2=TIME
        y_dummy[f'y_type_m_{m}'] = np.random.choice([0, 1, 2], size=N)
        
        # 10% ambiguous
        is_ambig = (np.random.rand(N) < 0.1).astype(bool)
        y_dummy[f'y_ambig_m_{m}'] = is_ambig
        
        # If ambiguous, set type to 3 but we're filtering out ambiguous anyway
        y_dummy.loc[is_ambig, f'y_type_m_{m}'] = 3 
        
    # 2. Instantiate Model
    model = MultiHeadModel(
        rr_multiples=rr_multiples,
        base_estimator_class=RandomForestClassifier,
        estimator_params={'n_estimators': 10, 'max_depth': 3, 'random_state': 42}
    )
    
    # 3. Fit Model
    model.fit(X_dummy, y_dummy)
    
    # 4. Predict
    probas = model.predict_proba(X_dummy)
    
    for m in rr_multiples:
        p = probas[m]
        print(f"Predictions wrapper shape for m={m}: {p.shape}")
        # Make sure they sum to 1 over the 3 classes
        assert p.shape == (N, 3), f"Expected shape {(N,3)} but got {p.shape}"
        assert np.allclose(p.sum(axis=1), 1.0), f"Probabilities for m={m} do not sum to 1."
        
    print("Verification passed! Architecture looks solid.")
