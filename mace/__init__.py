"""MACE: scalable, self-validating detection of driven variables.

    import mace
    result = mace.scan(X)      # X: (timepoints, channels)
    print(result.summary())

Paper, protocols and the full experiment ledger:
https://github.com/AkandaAshraf/DeepFeatSelection
DOI: 10.5281/zenodo.21988145
"""

from .core import MaceConfig, ScanResult, scan

__version__ = "0.1.0"
__all__ = ["scan", "MaceConfig", "ScanResult"]
