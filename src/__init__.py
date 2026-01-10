"""
RG Linee Guida Comportamentali AI.

Package principale per il sistema di linee guida comportamentali
basato su intelligenza artificiale per la sicurezza stradale.

Moduli:
    main: API FastAPI per la classificazione degli incidenti.
    model_factory: Factory pattern per la creazione di modelli ML.
    train_compare: Training e confronto di modelli ML.
"""

__version__ = "2.1.0"
__author__ = "RoadGuardian-FIA Team"
__description__ = "Sistema ML-Based per linee guida comportamentali di sicurezza stradale"

from .model_factory import get_model, ModelBase, DecisionTreeModel, RandomForestModel

__all__ = [
    "__version__",
    "__author__",
    "__description__",
    "get_model",
    "ModelBase",
    "DecisionTreeModel",
    "RandomForestModel",
]