# Telecom Churn: Trust and Production

Two notebooks that take a churn model from "it scores well" to "it can ship": rigorous interpretability and error analysis, then a production pipeline with data contracts, a model registry, and monitoring.

## Notebook 01 — Can We Trust This Model? (interpretability and error analysis)

* **Data contract first.** `TotalCharges` arrives as a string with blank cells; "No internet service" is a data artifact, not a category. Both fixed before any modeling.
* **Split discipline.** One seed, one held out test set, touched once.
* **Is the difference real?** Model comparison with confidence intervals instead of point estimate bragging.
* **Explanations are tested, not trusted.** Global SHAP and permutation importance, then a **faithfulness audit with a random feature control**: a SHAP plot is a claim about the model, and the claim is checked.
* **Who does the model hurt?** Slice analysis across 23 segment levels, separating model quality from base rates, plus a fairness review of `gender` and `SeniorCitizen` as inputs.
* **Calibration.** Brier score and expected calibration error (validation ECE 0.03); the threshold is treated as a business parameter, not a model parameter.
* Every conclusion is assembled by code from the run's result objects, so the text cannot drift from the numbers. A stated ceiling closes the notebook.

## Notebook 02 — From Notebook to Production (pipeline)

* **Ingestion.** One data contract with training and serving modes; rejected rows are quarantined with the reason each failed.
* **The feature layer travels with the model.** No training serving skew by construction.
* **Training gate that can actually block.** Floors on ROC-AUC/PR-AUC, ceilings on Brier; a model that fails the gate never reaches the registry.
* **Registry: version, then promote.** The serving bundle holds the model, threshold, calibration temperature, column contract, and drift profile.
* **Serving contract.** Refuse out of contract input, return calibrated scores with the decision, benchmark latency on exactly the rows reported.
* **Monitoring that proves it fires.** Drift detection is tested against simulated drift on internally consistent traffic, and the contract catches an inconsistent simulation.
* **Byte exact rollback.** A successor that passes the gate but performs worse is rolled back to the previously promoted bytes. A rollback that retrains is a redeployment, not a rollback.
* **Deliberately honest scoping.** Simulated production; the notebook lists what is still missing before real traffic (streaming monitoring, recalibration schedule, roll forward policy).

## Layout

The notebooks hold narrative; the package holds logic. That split is what makes the logic testable.

```
src/churnml/      # data contracts, features, models, evaluation, registry, serving
01_interpretability_and_error_analysis.ipynb
02_production_pipeline.ipynb
```

## Reproduce

```bash
pip install -r requirements.txt
jupyter notebook 01_interpretability_and_error_analysis.ipynb
```

Notebooks expect `src/` alongside them (the bootstrap cell inserts it into `sys.path`). Random seed 42 throughout.

## Author

Christopher Guerrero, October 2026.
