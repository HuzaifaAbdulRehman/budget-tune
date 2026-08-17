"""budget-tune: resource-constrained hyperparameter optimisation for recommenders.

The design this implements, including the hypotheses it is capable of falsifying, is in
``docs/design.md``. Read it before changing anything here: several choices that look
arbitrary (three data fractions, a fixed item index across folds, four benchmark files
instead of one) are load-bearing and have their reasons recorded there and in the module
docstrings.
"""

__version__ = "0.1.0"
