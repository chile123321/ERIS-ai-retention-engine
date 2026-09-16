# Evaluation protocol

Model development uses stratified folds within the development set. Report average precision as the primary imbalance-aware ranking measure alongside ROC-AUC, precision, recall, F1, F2, confusion matrix, calibration, and robustness to missing inputs. Calibration and threshold selection use development data only. The locked final test is evaluated once for an approved release candidate and must not drive iteration.
