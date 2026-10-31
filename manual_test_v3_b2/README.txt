ERIS V3 B2 — Manual Model Test

Purpose
-------
Small offline Tkinter tool for checking the existing research/demo bundle:
rf-v3-final15-unweighted-b2

Run from the repository root in PowerShell:

  .\.venv\Scripts\python.exe manual_test_v3_b2\manual_ui_b2.py

The tool reads the existing B2 metadata for its 15-field schema and reads a
saved B2 development example for "Load sample". It does not display employee
identifiers or the target. "Explain" invokes SHAP only when clicked.

TrainingTimesLastYear and YearsWithCurrManager accept Missing and are then
imputed by the preprocessing already stored inside the B2 bundle.

This is a local manual test tool. It does not send or save entered values. It
does not apply an HR threshold or make an employment decision. The IBM-coded
model is not approved for production, automated HR decisions, a new employer,
or a validated 3–6 month prediction horizon.
