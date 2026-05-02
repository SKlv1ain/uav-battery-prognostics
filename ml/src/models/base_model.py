"""
Base model interface for battery health prediction models.

All models (LSTM, XGBoost, Random Forest, Linear Regression) should
inherit from this base class to ensure a consistent API for training,
prediction, and evaluation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np


class BaseModel(ABC):
    """Abstract base class for all battery health prediction models."""

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.config = config or {}
        self.model = None
        self.is_trained = False

    @abstractmethod
    def build(self) -> None:
        """Build/initialize the model architecture."""
        pass

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Returns:
            Dictionary of training metrics (loss, etc.)
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model weights/artifacts to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model weights/artifacts from disk."""
        pass

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Returns:
            Dictionary with RMSE, MAE, R², and MAPE.
        """
        from sklearn.metrics import (
            mean_absolute_error,
            mean_absolute_percentage_error,
            mean_squared_error,
            r2_score,
        )

        y_pred = self.predict(X_test)

        return {
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
            "mape": float(mean_absolute_percentage_error(y_test, y_pred)),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.model_name}', trained={self.is_trained})"
