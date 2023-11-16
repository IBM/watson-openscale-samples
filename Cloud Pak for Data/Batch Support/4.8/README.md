# Watson OpenScale Notebooks 4.8 - Documentation

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

### Additional Notebook Configuration

#### For ZLinux Clusters
In case the notebook are running on zLinux Clusters, please use the below commands to install the necessary packages - 

    !conda install -y cmake cython==0.29.33 openblas==0.3.21 qdldl-python==0.1.7 pandas==1.4.4 pyparsing==2.4.7 statsmodels==0.13.2  
    !git clone https://github.com/tommyod/KDEpy && cd KDEpy && git checkout d52233099978dec38fa622fc86ce5e10368db1bd && rm -f KDEpy/cutils.c && CC=gcc python -m pip install . && cd .. && rm -rf KDEpy  
    !CC=gcc python -m pip install jenkspy==0.2.0 retrying==1.3.4 marshmallow==3.10.0 more-itertools==8.12.0 numba==0.57.1  
    !python -m pip install --no-build-isolation osqp==0.6.2 cvxpy==1.3.2  
    !python -m pip install shap==0.41.0  
    !CC=gcc GXX=g++ python -m pip install ibm-metrics-plugin==4.8.0.6  

#### For Default Spark Environments
If running the notebooks against Default Spark runtimes on CP4D clusters, please replace the following command

    !pip install --upgrade "ibm-metrics-plugin>=4.8.0.6" "ibm-watson-openscale>=3.0.32"

with  

    !pip install --upgrade "ibm-metrics-plugin>=4.8.0.6" "ibm-watson-openscale>=3.0.32" -t /home/spark/shared/user-libs/python
