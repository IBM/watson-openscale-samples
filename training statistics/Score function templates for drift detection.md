# Score function templates for drift detection model generation

Users are expected to author custom score functions that needs to be supplied as an input while generating drift detection model using ibm-ai-openscale python client. This page has some templates of score functions that can be used for reference. 

### Input to score function:
  - **training_data_frame :** Dataframe of the training data
    - Contains feature columns

### Output from score function:
 - **predicted label (aka decoded-target) numpy array :**
    * The data type of the values in this array should be same as dataset class label data type 
    * **Example :** predicted label numpy array : [A,B,C,D,A,B,D,......]
    
 
 - **probability numpy array :** 
    * The number of entries in each element of this array should be same as the unique class labels of the dataset 
    * Each element in the probability numpy array should be a vector with values between 0 and 1
    * **Example:** probability numpy array: [[0.50,0.20,0.15,0.15] , [0.60,0.10,0.05,20.5], .......]

### Contents
- [WML Model Engine](#WML)
   * [Local mode](#LocalMode)
   * [Online Scoring](#OnlineScoring)
- [Azure Model Engine](#Azure)
   * [Azure Studio](#AzureStudio)
   * [Azure ML Service](#AzureMLService)
- [AWS SageMaker Model Engine](#AWS)
- [SPSS Model Engine](#SPSS)
- [Custom Model Engine](#Custom)

## WML Model Engine: <a name="WML"></a>
This section provides the score function templates for model deployed in WML. There are 2 formats specified (local model , online model) and user is free to choose any of the formats . **The templates specified below are common for binary / multi-class classification cases**.
### Local mode: <a name="LocalMode"></a>
- Model stored in WML is retrieved and loaded in local environment. This model is used to score.
```
def score(training_data_frame):
    WML_CREDENTIALS = {
        <EDIT THIS>
    }
    try:
        # Supply the model id
        space_id = <EDIT THIS>
        model_id = <EDIT THIS>
        
        # Retain feature columns from user selection
        feature_columns = list(training_data_frame.columns)
        
        # Load the WML model in local object
        from ibm_watson_machine_learning import APIClient
        wml_client = APIClient(WML_CREDENTIALS)
        wml_client.set.default_space(space_id)
        model = wml_client.repository.load(model_id)
        
        # Predict the training data locally 
        # Example of a spark based model ( the below set of lines to be customized based on model framework)
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.master("local").appName("drift").getOrCreate()
        spark_frame = spark.createDataFrame(training_data_frame)
        spark_frame.printSchema()
        
        score_predictions = model.transform(spark_frame)
        score_predictions_pd = score_predictions.select("*").toPandas()

        probability_column_name = <EDIT THIS>
        prediction_column_name = <EDIT THIS>
        
        import numpy as np
        probability_array = np.array(
            [x.tolist() for x in score_predictions_pd[probability_column_name]])
        prediction_vector = np.array(
            [x for x in score_predictions_pd[prediction_column_name]])
        
        return probability_array, prediction_vector
    except Exception as ex:
        raise Exception("Scoring failed. Error: {}".format(str(ex)))
```
  *  _**Limitations for running in WML local mode:**_
        - If a model is trained and deployed using WML Auto AI the local mode does not work as the right runtime used to train the model is not known
        - If a model is generated and deployed using WML Model Builder  - the local mode does not work as WML python client does not support this context.
        
 
### Online Scoring: <a name="OnlineScoring"></a>
- Using `deployment_id` and `space_id`. This snippet uses the online scoring endpoint of a WML model using IBM WML python client library. **As this is online scoring , a cost is associated with the same .**
- **Note:** Please install python library "ibm_watson_machine_learning" to execute below snippet.
- **Binary or Multi-class Classifier**
```
def score(training_data_frame):
    # To be filled by the user
    WML_CREDENTIALS = {
        <EDIT THIS>
    }
    try:
        deployment_id = <EDIT THIS>
        space_id = <EDIT THIS>

        # The data type of the label column and prediction column should be same .
        # User needs to make sure that label column and prediction column array 
        # should have the same unique class labels
        # edit these if your prediction column has different name
        prediction_column_name = "prediction"
        probability_column_name = "probability"

        feature_columns = list(training_data_frame.columns)
        training_data_rows = training_data_frame[feature_columns].values.tolist()
        # print(training_data_rows)

        # Load the WML model in local object
        from ibm_watson_machine_learning import APIClient
        wml_client = APIClient(WML_CREDENTIALS)
        wml_client.set.default_space(space_id)

        payload_scoring = {
            wml_client.deployments.ScoringMetaNames.INPUT_DATA: [{
                "fields": feature_columns,
                "values": [x for x in training_data_rows]
            }]
        }

        score = wml_client.deployments.score(deployment_id, payload_scoring)
        score_predictions = score.get('predictions')[0]

        prob_col_index = list(score_predictions.get('fields')).index(probability_column_name)
        predict_col_index = list(score_predictions.get('fields')).index(prediction_column_name)

        if prob_col_index < 0 or predict_col_index < 0:
            raise Exception("Missing prediction/probability column in the scoring response")
            
        import numpy as np
        probability_array = np.array([value[prob_col_index] for value in score_predictions.get('values')])
        prediction_vector = np.array([value[predict_col_index] for value in score_predictions.get('values')])

        return probability_array, prediction_vector
    except Exception as ex:
        raise Exception("Scoring failed. Error: {}".format(str(ex)))
```

- **Regression**
```
def score(training_data_frame):
    # To be filled by the user
    WML_CREDENTIALS = {
        <EDIT THIS>
    }
    try:
        deployment_id = <EDIT THIS>
        space_id = <EDIT THIS>

        # edit this if your prediction column has different name
        prediction_column_name = "prediction"

        feature_columns = list(training_data_frame.columns)
        training_data_rows = training_data_frame[feature_columns].values.tolist()
        # print(training_data_rows)

        from ibm_watson_machine_learning import APIClient
        wml_client = APIClient(WML_CREDENTIALS)
        wml_client.set.default_space(space_id)

        payload_scoring = {
            wml_client.deployments.ScoringMetaNames.INPUT_DATA: [{
                "fields": feature_columns,
                "values": [x for x in training_data_rows]
            }]
        }

        score = wml_client.deployments.score(deployment_id, payload_scoring)
        score_predictions = score.get('predictions')[0]

        predict_col_index = list(score_predictions.get('fields')).index(prediction_column_name)
        if predict_col_index < 0:
            raise Exception("Missing prediction column in the scoring response")
            
        import numpy as np
        prediction_vector = np.array([value[predict_col_index] for value in score_predictions.get('values')])

        return prediction_vector
    except Exception as ex:
        raise Exception("Scoring failed. Error: {}".format(str(ex)))
```

## Azure Model Engine: <a name="Azure"></a>
### Azure Studio: <a name="AzureStudio"></a>
This section provides the score function templates for model deployed in Azure Model Engine. User needs to consider that online scoring endpoints of Azure Studio will be used. **As this is online scoring, a cost is associated with the same .**

- **Binary Classifier**
```
def score(training_data_frame):
    azure_scoring_url = <REQUEST RESPONSE URL FROM AZURE MODEL>
    token = <PRIMARY_KEY FROM AZURE MODEL>

    # edit these if your prediction and probability column have different names
    prediction_column_name = "Scored Labels"
    probability_column_name = "Scored Probabilities"

    try:
        input_values = training_data_frame.values.tolist()
        feature_columns = list(training_data_frame.columns)

        # Payload
        import requests
        from datetime import datetime, timedelta

        payload = {
            "Inputs": {
                "input1": {
                    "ColumnNames": feature_columns,
                    "Values": input_values
                }
            },
            "GlobalParameters": {}
        }

        headers = {'Authorization': 'Bearer ' + token}
        start = datetime.utcnow()
        response = requests.post(azure_scoring_url, json=payload, headers=headers)
        if not response.ok:
            raise Exception(str(response.content))

        # assumed response json structure
        # {
        #     "Results": {
        #         "output1": {
        #         "type": "DataTable",
        #         "value": {
        #             "ColumnNames": [
        #             ],
        #             "ColumnTypes": [
        #             ],
        #             "Values": [
        #                 [],[]
        #             ]
        #         }
        #         }
        #     }
        # }
        # If your scoring response does not match above schema, 
        # please modify below code to extract prediction and probabilities array

        # Extract results part
        results = response.json()["Results"]["output1"]["value"]

        prob_col_index = list(results.get('ColumnNames')).index(probability_column_name)
        predict_col_index = list(results.get('ColumnNames')).index(prediction_column_name)

        if prob_col_index < 0 or predict_col_index < 0:
            raise Exception("Missing prediction/probability column in the scoring response")

        # Get Score label from first entry
        first_entry = results.get('Values')[0]
        score_label = first_entry[predict_col_index]
        print(score_label)

        score_prob_1 = float(first_entry[prob_col_index])
        main_label = True
        if score_prob_1 < 0.5:
            #The score label is not main label of interest
            main_label = False

        output = [[value[predict_col_index], 1 - float(value[prob_col_index]) if \
            (value[predict_col_index] == score_label and not main_label) else \
                float(value[prob_col_index])] for value in results.get('Values')]
        print(len(output))

        import numpy as np
        # Construct predicted_label array
        predicted_vector = np.array([value[0] for value in output])

        # Construct probabilities array
        probability_array = np.array([[value[1],(1-value[1])] for value in output])

        return probability_array, predicted_vector
    except Exception as ex:
        raise Exception("Scoring failed. {}".format(str(ex)))
```

- **Multi-class Classifier**
```
def score(training_data_frame):
    azure_scoring_url = <REQUEST RESPONSE URL FROM AZURE MODEL>
    token = <PRIMARY_KEY FROM AZURE MODEL>

    # edit these if your prediction and probability column have different names/prefixes
    prediction_column_name = "Scored Labels"
    probability_column_prefix = "Scored Probabilities"

    try:
        input_values = training_data_frame.values.tolist()
        feature_columns = list(training_data_frame.columns)

        # Payload
        import requests
        from datetime import datetime, timedelta

        payload = {
            "Inputs": {
                "input1": {
                    "ColumnNames": feature_columns,
                    "Values": input_values
                }
            },
            "GlobalParameters": {}
        }

        headers = {'Authorization': 'Bearer ' + token}
        start = datetime.utcnow()
        response = requests.post(azure_scoring_url, json=payload, headers=headers)
        if not response.ok:
            raise Exception(str(response.content))

        response_time = (datetime.utcnow() - start).total_seconds() * 1000
        print(response_time)

        # assumed response json structure
        # {
        #     "Results": {
        #         "output1": {
        #         "type": "DataTable",
        #         "value": {
        #             "ColumnNames": [
        #             ],
        #             "ColumnTypes": [
        #             ],
        #             "Values": [
        #                 [],[]
        #             ]
        #         }
        #         }
        #     }
        # }
        # If your scoring response does not match above schema, 
        # please modify below code to extract prediction and probabilities array

        # Extract results
        results = response.json()["Results"]["output1"]["value"]
        result_column_names = list(results.get('ColumnNames'))

        predict_col_index = result_column_names.index(prediction_column_name)
        prob_col_indexes = [result_column_names.index(column_name) for column_name in result_column_names \
            if column_name.startswith(probability_column_prefix, 0)]

        # Compute for all values
        score_label_list = []
        score_prob_list = []

        for value in results.get("Values"):
            score_label_list.append(str(value[predict_col_index]))

            #Construct prob
            score_prob_values = [float(value[index]) for index in range(len(value)) \
                if index in prob_col_indexes]
            score_prob_list.append(score_prob_values)

        import numpy as np
        # Construct predicted_label bucket
        predicted_vector = np.array(score_label_list)

        # Scored probabilities
        probability_array = np.array(score_prob_list)

        return probability_array, predicted_vector
    except Exception as ex:
        raise Exception("Scoring failed. {}".format(str(ex)))
```

- **Regression**
```
def score(training_data_frame):
    azure_scoring_url = <REQUEST RESPONSE URL FROM AZURE MODEL>
    token = <PRIMARY_KEY FROM AZURE MODEL>

    # edit this if your prediction has different names
    prediction_column_name = "Scored Labels"

    try:
        input_values = training_data_frame.values.tolist()
        feature_columns = list(training_data_frame.columns)

        # Payload
        import requests
        from datetime import datetime, timedelta

        payload = {
            "Inputs": {
                "input1": {
                    "ColumnNames": feature_columns,
                    "Values": input_values
                }
            },
            "GlobalParameters": {}
        }

        headers = {'Authorization': 'Bearer ' + token}
        start = datetime.utcnow()
        response = requests.post(azure_scoring_url, json=payload, headers=headers)
        if not response.ok:
            raise Exception(str(response.content))

        response_time = (datetime.utcnow() - start).total_seconds() * 1000
        print(response_time)

        # assumed response json structure
        # {
        #     "Results": {
        #         "output1": {
        #         "type": "DataTable",
        #         "value": {
        #             "ColumnNames": [
        #             ],
        #             "ColumnTypes": [
        #             ],
        #             "Values": [
        #                 [],[]
        #             ]
        #         }
        #         }
        #     }
        # }
        # If your scoring response does not match above schema, 
        # please modify below code to extract prediction and probabilities array

        # Extract results
        results = response.json()["Results"]["output1"]["value"]
        result_column_names = list(results.get('ColumnNames'))

        predict_col_index = result_column_names.index(prediction_column_name)

        # Compute for all values
        score_label_list = []
        for value in results.get("Values"):
            score_label_list.append(str(value[predict_col_index]))

        import numpy as np
        # Construct predicted_label bucket
        predicted_vector = np.array(score_label_list)

        return predicted_vector
    except Exception as ex:
        raise Exception("Scoring failed. {}".format(str(ex)))
```

### Azure ML Service: <a name="AzureMLService"></a>
This section provides the score function templates for model deployed in Azure ML Service. User needs to consider that online scoring endpoints of Azure ML Service will be used. The below snippet is valid for both multi-class and binary classification model. **As this is online scoring, a cost is associated with the same .**

- **Binary or Multi-class Classifier**
```
def score(training_data_frame):
    az_scoring_uri = <EDIT THIS>
    api_key = <DEPLOYMENT API KEY>

    # edit these if your prediction and probability column have different names
    prediction_column_name = "Scored Labels"
    probability_column_name = "Scored Probabilities"

    try:
        input_values = training_data_frame.values.tolist()
        feature_cols = list(training_data_frame.columns)
        input_data = [{field: value  for field,value in zip(feature_cols, input_value)} for input_value in input_values]

        payload = {
            "input": input_data
        }

        import requests
        import json
        import numpy as np
        import time

        headers = {'Content-Type':'application/json',  'Authorization':('Bearer '+ api_key)}
        start_time = time.time()  
        response = requests.post(az_scoring_uri, json=payload, headers=headers)
        if not response.ok:
            raise Exception(str(response.content))

        response_time = int((time.time() - start_time)*1000)
        print(response_time)

        # assumed response json structure
        # {
        #     "output": [
        #         {
        #             "Scored Labels": "Risk",
        #             "Scored Probabilities": []
        #         }
        #     ]
        # }
        # If your scoring response does not match above schema, 
        # please modify below code to extract prediction and probabilities array

        response_dict = json.loads(response.json())
        output = response_dict["output"]

        # Compute for all values
        score_label_list = []
        score_prob_list = []
        for value in output:
            score_label_list.append(str(value[prediction_column_name]))
            score_prob_list.append(value[probability_column_name])

        return np.array(score_prob_list), np.array(score_label_list)
    except Exception as ex:
        raise Exception("Scoring failed. {}".format(str(ex)))
```

- **Regression**
```
def score(training_data_frame):
    az_scoring_uri = <EDIT THIS>
    api_key = <DEPLOYMENT API KEY>

    # edit this if your prediction column has different name
    prediction_column_name = "Scored Labels"

    try:
        input_values = training_data_frame.values.tolist()
        feature_cols = list(training_data_frame.columns)
        input_data = [{field: value  for field,value in zip(feature_cols, input_value)} for input_value in input_values]

        payload = {
            "input": input_data
        }

        import requests
        import json
        import numpy as np
        import time

        headers = {'Content-Type':'application/json',  'Authorization':('Bearer '+ api_key)}
        start_time = time.time()  
        response = requests.post(az_scoring_uri, json=payload, headers=headers)
        if not response.ok:
            raise Exception(str(response.content))

        response_time = int((time.time() - start_time)*1000)
        print(response_time)

        # assumed response json structure
        # {
        #     "output": [
        #         {
        #             "Scored Labels": 123
        #         }
        #     ]
        # }
        # If your scoring response does not match above schema, 
        # please modify below code to extract prediction and probabilities array

        response_dict = json.loads(response.json())
        output = response_dict["output"]

        # Compute for all values
        score_label_list = []
        for value in output:
            score_label_list.append(str(value[prediction_column_name]))

        return np.array(score_label_list)
    except Exception as ex:
        raise Exception("Scoring failed. {}".format(str(ex)))
```

## AWS SageMaker Model Engine: <a name="AWS"></a>
This section provides the score function templates for for model deployed in SageMaker Model Engine. User needs to consider that online scoring endpoints of SageMaker will be used.**As this is online scoring, a cost is associated with the same .** Please note that we are the below snipets are created with a assumption that input datasets for AWS are **one hot encoded for categorical columns** and label-encoded for **label column**

- **Binary Classifier**
```
def score(training_data_frame):
    SAGEMAKER_CREDENTIALS = {
        "access_key_id": <EDIT THIS>,
        "secret_access_key": <EDIT THIS>,
        "region": <EDIT THIS>
    }

    # User input needed
    endpoint_name = <EDIT THIS>

    # edit these if your prediction and probability column have different names
    prediction_column_name = "predicted_label"
    probability_column_name = "score"

    try:
        access_id = SAGEMAKER_CREDENTIALS.get('access_key_id')
        secret_key = SAGEMAKER_CREDENTIALS.get('secret_access_key')
        region = SAGEMAKER_CREDENTIALS.get('region')

        # Covert the training data frames to bytes
        import io
        import numpy as np
        train_df_bytes = io.BytesIO()
        np.savetxt(train_df_bytes, training_data_frame.values, delimiter=',', fmt='%g')
        payload_data = train_df_bytes.getvalue().decode().rstrip()

        # Score the training data
        import requests
        import time
        import json
        import boto3

        runtime = boto3.client('sagemaker-runtime', region_name=region, aws_access_key_id=access_id, aws_secret_access_key=secret_key)
        start_time = time.time()

        response = runtime.invoke_endpoint(EndpointName=endpoint_name, ContentType='text/csv', Body=payload_data)
        if not response.ok:
            raise Exception(str(response.content))

        response_time = int((time.time() - start_time)*1000)
        results_decoded = json.loads(response['Body'].read().decode())

        # Extract the details
        results = results_decoded['predictions']

        predicted_label_list = []
        score_prob_list = []

        for result in results :
            predicted_label_list.append(result[prediction_column_name])
            
            # Please note probability always to belongs to the same class label
            score_prob_list.append(result[probability_column_name])

        import numpy as np
        predicted_vector = np.array(predicted_label_list)
        probability_array = np.array([[prob, 1-prob] for prob in score_prob_list])

        return probability_array, predicted_vector
    except Exception as ex:
        raise Exception("Scoring failed. {}".format(str(ex)))
```

- **Multi-class Classifier**
```
def score(training_data_frame):
    SAGEMAKER_CREDENTIALS = {
        "access_key_id": <EDIT THIS>,
        "secret_access_key": <EDIT THIS>,
        "region": <EDIT THIS>
    }
    # User input needed
    endpoint_name = <EDIT THIS>

    # edit these if your prediction and probability column have different names
    prediction_column_name = "predicted_label"
    probability_column_name = "score"

    try:
        access_id = SAGEMAKER_CREDENTIALS.get('access_key_id')
        secret_key = SAGEMAKER_CREDENTIALS.get('secret_access_key')
        region = SAGEMAKER_CREDENTIALS.get('region')
        
        # Convert the training data frames to bytes
        import io
        import numpy as np
        train_df_bytes = io.BytesIO()
        np.savetxt(train_df_bytes, training_data_frame.values, delimiter=',', fmt='%g')
        payload_data = train_df_bytes.getvalue().decode().rstrip()

        # Score the training data
        import requests
        import time
        import json
        import boto3

        runtime = boto3.client('sagemaker-runtime', region_name=region, aws_access_key_id=access_id, aws_secret_access_key=secret_key)
        start_time = time.time()

        response = runtime.invoke_endpoint(EndpointName=endpoint_name, ContentType='text/csv', Body=payload_data)
        if not response.ok:
            raise Exception(str(response.content))

        response_time = int((time.time() - start_time)*1000)
        results_decoded = json.loads(response['Body'].read().decode())

        # Extract the details
        results = results_decoded['predictions']

        predicted_vector_list = []
        probability_array_list = []

        for value in results:
            predicted_vector_list.append(value[prediction_column_name])
            probability_array_list.append(value[probability_column_name])

        # Convert to numpy arrays
        probability_array = np.array(probability_array_list)
        predicted_vector = np.array(predicted_vector_list)

        return probability_array, predicted_vector
    except Exception as ex:
        raise Exception("Scoring failed. {}".format(str(ex)))
```

- **Regression**
```
def score(training_data_frame):
    SAGEMAKER_CREDENTIALS = {
        "access_key_id": <EDIT THIS>,
        "secret_access_key": <EDIT THIS>,
        "region": <EDIT THIS>
    }
    # User input needed
    endpoint_name = <EDIT THIS>

    # edit this if your prediction column has different name
    prediction_column_name = "score"

    try:
        access_id = SAGEMAKER_CREDENTIALS.get('access_key_id')
        secret_key = SAGEMAKER_CREDENTIALS.get('secret_access_key')
        region = SAGEMAKER_CREDENTIALS.get('region')
        
        # Convert the training data frames to bytes
        import io
        import numpy as np
        train_df_bytes = io.BytesIO()
        np.savetxt(train_df_bytes, training_data_frame.values, delimiter=',', fmt='%g')
        payload_data = train_df_bytes.getvalue().decode().rstrip()

        # Score the training data
        import requests
        import time
        import json
        import boto3

        runtime = boto3.client('sagemaker-runtime', region_name=region, aws_access_key_id=access_id, aws_secret_access_key=secret_key)
        start_time = time.time()

        response = runtime.invoke_endpoint(EndpointName=endpoint_name, ContentType='text/csv', Body=payload_data)
        if not response.ok:
            raise Exception(str(response.content))

        response_time = int((time.time() - start_time)*1000)
        results_decoded = json.loads(response['Body'].read().decode())

        # Extract the details
        results = results_decoded['predictions']

        predicted_vector_list = []
        for value in results:
            predicted_vector_list.append(value[prediction_column_name])

        # Convert to numpy arrays
        predicted_vector = np.array(predicted_vector_list)

        return predicted_vector
    except Exception as ex:
        raise Exception("Scoring failed. {}".format(str(ex)))
```

## SPSS Model Engine: <a name="SPSS"></a>
This section provides the score function template for model deployed in SPSS model engine. The online scoring end point of custom engine will be used for scoring. The below snippets holds good for binary/multi-class. **As this is online scoring, a cost is associated with the same .**

- **Binary or Multi-class Classifier**
```
def score(training_data_frame):
    SPSS_CREDENTIALS = {
        "username": <EDIT THIS>,
        "password": <EDIT THIS>
    }
    # To be filled by the user - model scoring url
    scoring_url = <EDIT THIS>
    # "id" - Identifier for the scoring configuration being used to generate scores   
    scoring_id = <EDIT THIS>

    # edit these if your prediction and probability column have different prefixes
    prediction_column_prefix = "$N-"
    probability_column_prefix = "$NP-"

    try:
        feature_columns = list(training_data_frame.columns)
        training_data_dict = training_data_frame.to_dict(orient="records")
        request_input_row = [{"input": [{"name": key, "value": value} for key, value in json.items()]} \
            for json in training_data_dict]

        payload_scoring = {
            "id": scoring_id,
            "requestInputTable": [{
                "requestInputRow": request_input_row
            }]
        }

        # Retain username and password for custom
        username = SPSS_CREDENTIALS.get("username")
        password =  SPSS_CREDENTIALS.get("password")

        import requests
        import time
        import json
        import numpy as np

        start_time = time.time()
        response = requests.post(scoring_url, json=payload_scoring, auth=(username, password))
        if not response.ok:
            error_msg = "Scoring failed : " + str(response.status_code)
            if response.content is not None:
                error_msg = error_msg + ", " + response.content.decode("utf-8")
            raise Exception(error_msg)

        response_time = int((time.time() - start_time)*1000)
        print(response_time)

        # Convert response to dict
        score_predictions = json.loads(response.text)
        output_column_names = list(score_predictions.get("columnNames")["name"])

        # identify prediction and probability column names
        probability_column_names = [item for item in output_column_names \
            if item.startswith(probability_column_prefix)]
        if len(probability_column_names) == 0:
            raise Exception("No probability column found. Please specify probability column name.")

        prediction_column_name = [item for item in output_column_names \
            if item.startswith(prediction_column_prefix)]
        if len(prediction_column_name) != 1:
            raise Exception(
                "Either no prediction column found or more than one is found. Please specify prediction column name.")
        prediction_column_name = prediction_column_name[0]

        # identify prediction and probability column indexes
        prob_col_indexes = [output_column_names.index(prob_col_name) for prob_col_name in probability_column_names]
        predict_col_index = output_column_names.index(prediction_column_name)

        if len(prob_col_indexes) == 0 or predict_col_index < 0:
            raise Exception("Missing prediction/probability column in the scoring response")

        probability_array = []
        prediction_vector = []

        for value in score_predictions.get("rowValues"):
            response_prediction = value["value"][predict_col_index]["value"]
            prediction_vector.append(response_prediction)
            
            response_prob_array = []
            for prob_col_index in prob_col_indexes:
                response_prob_array.append(float(value["value"][prob_col_index]["value"]))
                
            probability_array.append(response_prob_array)
            
        import numpy as np
        probability_array = np.array(probability_array)
        prediction_vector = np.array(prediction_vector)

        return probability_array,prediction_vector
    except Exception as ex:
        raise Exception("Scoring failed. {}".format(str(ex)))
```

- **Regression**
```
def score(training_data_frame):
    SPSS_CREDENTIALS = {
        "username": <EDIT THIS>,
        "password": <EDIT THIS>
    }
    # To be filled by the user - model scoring url
    scoring_url = <EDIT THIS>
    # "id" - Identifier for the scoring configuration being used to generate scores   
    scoring_id = <EDIT THIS>

    # edit this if your prediction column has different prefix
    prediction_column_prefix = "$N-"

    try:
        feature_columns = list(training_data_frame.columns)
        training_data_dict = training_data_frame.to_dict(orient="records")
        request_input_row = [{"input": [{"name": key, "value": value} for key, value in json.items()]} \
            for json in training_data_dict]

        payload_scoring = {
            "id": scoring_id,
            "requestInputTable": [{
                "requestInputRow": request_input_row
            }]
        }

        # Retain username and password for custom
        username = SPSS_CREDENTIALS.get("username")
        password =  SPSS_CREDENTIALS.get("password")

        import requests
        import time
        import json
        import numpy as np

        start_time = time.time()
        response = requests.post(scoring_url, json=payload_scoring, auth=(username, password))
        if not response.ok:
            error_msg = "Scoring failed : " + str(response.status_code)
            if response.content is not None:
                error_msg = error_msg + ", " + response.content.decode("utf-8")
            raise Exception(error_msg)

        response_time = int((time.time() - start_time)*1000)
        print(response_time)

        # Convert response to dict
        score_predictions = json.loads(response.text)
        output_column_names = list(score_predictions.get("columnNames")["name"])

        # identify prediction column name
        prediction_column_name = [item for item in output_column_names \
            if item.startswith(prediction_column_prefix)]
        if len(prediction_column_name) != 1:
            raise Exception(
                "Either no prediction column found or more than one is found. Please specify prediction column name.")
        prediction_column_name = prediction_column_name[0]

        # identify prediction column index
        predict_col_index = output_column_names.index(prediction_column_name)

        if predict_col_index < 0:
            raise Exception("Missing prediction column in the scoring response")

        prediction_vector = []
        for value in score_predictions.get("rowValues"):
            response_prediction = value["value"][predict_col_index]["value"]
            prediction_vector.append(response_prediction)

        import numpy as np
        prediction_vector = np.array(prediction_vector)

        return prediction_vector
    except Exception as ex:
        raise Exception("Scoring failed. {}".format(str(ex)))
```

## Custom Model Engine: <a name="Custom"></a>
This section provides the score function template for model deployed in a custom engine. The online scoring end point of custom engine will be used for scoring. The below snippets holds good for binary/multi-class. **As this is online scoring, a cost is associated with the same .**

```
def score(training_data_frame):
    CUSTOM_ENGINE_CREDENTIALS = {
        "url": <EDIT THIS>,
        "username": <EDIT THIS>,
        "password": <EDIT THIS>
    }
    # To be filled by the user - model scoring url
    scoring_url = <EDIT THIS>

    # The data type of the label column and prediction column should be same .
    # User needs to make sure that label column and prediction column array 
    # should have the same unique class labels
    prediction_column_name = <EDIT THIS>
    probability_column_name = <EDIT THIS>

    try:
        feature_columns = list(training_data_frame.columns)
        training_data_rows = training_data_frame[feature_columns].values.tolist()

        payload_scoring = {
            "fields": feature_columns,
            "values": [x for x in training_data_rows]
        }

        # Retain username and password for custom
        username = CUSTOM_ENGINE_CREDENTIALS.get("username")
        password =  CUSTOM_ENGINE_CREDENTIALS.get("password")

        import requests
        import time

        start_time = time.time()
        response = requests.post(scoring_url, json=payload_scoring, auth=(username, password))
        if not response.ok:
            raise Exception(str(response.content))

        response_time = int((time.time() - start_time)*1000)
        print(response_time)

        # Convert response to dict
        import json
        score_predictions = json.loads(response.text)

        prob_col_index = list(score_predictions.get("fields")).index(probability_column_name)
        predict_col_index = list(score_predictions.get("fields")).index(prediction_column_name)

        if prob_col_index < 0 or predict_col_index < 0:
            raise Exception("Missing prediction/probability column in the scoring response")

        import numpy as np
        probability_array = np.array([value[prob_col_index] for value in score_predictions.get('values')])
        prediction_vector = np.array([value[predict_col_index] for value in score_predictions.get('values')])

        return probability_array,prediction_vector
    except Exception as ex:
        raise Exception("Scoring failed. {}".format(str(ex)))
```
