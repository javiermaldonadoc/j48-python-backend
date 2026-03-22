from __future__ import annotations

from .core import C45TreeClassifier
from .engine import EncodedNumpyJ48FastEngine, J48EngineSpec, NumpyJ48Engine, build_engine
from .sklearn_api import J48Classifier, J48FastClassifier

__version__ = "0.1.0"

J48CoreClassifier = C45TreeClassifier

__all__ = [
    "__version__",
    "C45TreeClassifier",
    "J48Classifier",
    "J48FastClassifier",
    "J48CoreClassifier",
    "J48EngineSpec",
    "NumpyJ48Engine",
    "EncodedNumpyJ48FastEngine",
    "build_engine",
]
