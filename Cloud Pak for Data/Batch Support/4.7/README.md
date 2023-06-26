# Watson OpenScale Notebooks 4.7 - Documentation

## The below Notbooks are available for this version

#### Watson OpenScale - Onboard models for monitoring using Configuration Package
The notebook demonstrates how to onboard a model for monitoring in IBM Watson OpenScale. Use the notebook to enable quality, drift, fairness and explainability monitoring. It requires a configuration package (archive) which can be generated with the help of Common Configuration Notebook [Simplified].

#### Watson OpenScale - Onboard models for monitoring using training data table and a sample csv
The notebook demonstrates how to onboard a model for monitoring in IBM Watson OpenScale. Use the notebook to enable quality, drift, fairness and explainability monitoring. It requires a table containing scored training data and a sample csv.

#### Common Configuration Notebook [Simplified]
This notebook shows how to generate the configuration package required to onboard model for monitoring.

#### Common Configuration Notebook [Detailed]
This notebook is used to generate following artefacts:

1. DDLs for creating Feedback, Payload, Drifted Transactions and Explanations tables
2. Configuration JSON containing basic model details
3. Drift Configuration Archive
4. Explainability Perturbations Archive

#### Analyze Drifted Transactions Notebook
This notebook helps users of IBM Watson OpenScale to analyze payload transactions that are causing drift - both drop in accuracy and drop in data consistency.
The notebook is designed to give users a jump start in their analysis of the payload transactions. It is by no means a comprehensive analysis.


These notebooks are availabale in _jdbc_ and _hive_ folders for the models which stores its runtime data in a remote DB2 table and Hive table respectively.
