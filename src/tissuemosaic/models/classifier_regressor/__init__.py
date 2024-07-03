# Decide what to expose
from .scikit_learn_interface import MlpRegressor as MlpRegressor
from .scikit_learn_interface import MlpClassifier as MlpClassifier

__all__ = ["MlpRegressor", "MlpClassifier"]
