"""
Public exports for the MBPO reward-model package.

Importing from ``r_predict_model`` exposes the FHSS training script's primary
model wrapper while keeping implementation details in submodules.
"""

from .model import EnsembleDynamicsModel

# Keep the package surface small and explicit.
__all__ = ["EnsembleDynamicsModel"]
