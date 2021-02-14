"""
The following AI Platform environment variables are passed to containers or python modules of the training task when this field is set:

Data information:
AIP_DATA_FORMAT : Exported data format.
AIP_TRAINING_DATA_URI : Sharded exported training data uris.
AIP_VALIDATION_DATA_URI : Sharded exported validation data uris.
AIP_TEST_DATA_URI : Sharded exported test data uris. destination can be only one of the following:


"""

import dask.dataframe as dd
from google.cloud import bigquery, bigquery_storage
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, plot_confusion_matrix
import matplotlib.pyplot as plt  
import pickle
from google.cloud import storage
from datetime import datetime
import os
import pandas as pd
import logging
import numpy
from typing import Union
import json

import argparse
# feature categories

BINARY_FEATURES = [
    'gender',
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'PhoneService',
    'PaperlessBilling']

NUMERIC_FEATURES = [
    'tenure',
    'MonthlyCharges',
    'TotalCharges']

CATEGORICAL_FEATURES = [
    'InternetService',
    'OnlineSecurity',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaymentMethod',
    'MultipleLines']

ALL_COLUMNS = BINARY_FEATURES+NUMERIC_FEATURES+CATEGORICAL_FEATURES
BINARY_FEATURES_IDX = list(range(0,len(BINARY_FEATURES)))
NUMERIC_FEATURES_IDX = list(range(len(BINARY_FEATURES), len(BINARY_FEATURES)+len(NUMERIC_FEATURES)))
CATEGORICAL_FEATURES_IDX = list(range(len(BINARY_FEATURES+NUMERIC_FEATURES), len(BINARY_FEATURES+NUMERIC_FEATURES+CATEGORICAL_FEATURES)))


# TODO: From the experiment.ipynb copy-paste the load_data_from_gcs function 
def load_data_from_gcs(data_gcs_path):
    logging.info("reading gs file: {}".format(data_gcs_path))
    return dd.read_csv(data_gcs_path, dtype={'TotalCharges': 'object'}).compute()


# TODO: From the experiment.ipynb copy-paste the load_data_from_bq function 
def load_data_from_bq(bq_uri):
    project,dataset,table =  bq_uri.split(".")
    bqclient = bigquery.Client(project=PROJECT)
    bqstorageclient = bigquery_storage.BigQueryReadClient()
    query_string = """
    SELECT * from {ds}.{tbl}
    """.format(ds=dataset, tbl=table)

    return (
        bqclient.query(query_string)
        .result()
        .to_dataframe(bqstorage_client=bqstorageclient)
    )


# TODO: From the experiment.ipynb copy-paste the sort_missing_total_charges function 
def sort_missing_total_charges(df):
    df.loc[df.tenure == 0, 'TotalCharges'] = df.loc[df.tenure == 0, 'MonthlyCharges']
    

# TODO: From the experiment.ipynb copy-paste the data_selection function 
def data_selection(df):
    data = df.loc[:, BINARY_FEATURES+NUMERIC_FEATURES+CATEGORICAL_FEATURES]
    # We create a series with the prediciton label
    labels = df.Churn

    return data, labels


# TODO: From the experiment.ipynb copy-paste the pipeline_builder function 
def pipeline_builder(params_svm: dict) -> Pipeline:
    # Definining a preprocessing step for our pipeline. 
    # it specifies how the features are going to be transformed
    preprocessor = ColumnTransformer(
        transformers=[
            ('bin', OrdinalEncoder(), BINARY_FEATURES_IDX),
            ('num', StandardScaler(), NUMERIC_FEATURES_IDX),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES_IDX)])


    # We now create a full pipeline, for preprocessing and training.
    # for training we selected a linear SVM classifier
    
    clf = SVC()
    clf.set_params(**params_svm)
    
    return Pipeline(steps=[ ('preprocessor', preprocessor),
                          ('classifier', clf)])


# TODO: From the experiment.ipynb copy-paste the train_model function 
def train_model(clf: Pipeline, X: Union[pd.DataFrame, numpy.ndarray], y: Union[pd.DataFrame, numpy.ndarray]) -> float:
    # run cross validation to get training score. we can use this score to optimise training
    score = cross_val_score(clf, X, y, cv=10, n_jobs=-1).mean()
    
    # Now we fit all our data to the classifier. Shame to leave a portion of the data behind
    clf.fit(X, y)
    
    return score
      
    
# TODO: From the experiment.ipynb copy-paste the process_gcs_uri function 
def process_gcs_uri(uri):
    url_arr = uri.split("/")
    if "." not in url_arr[-1]:
        file = ""
    else:
        file = url_arr.pop()
    
    scheme = url_arr[0]
    bucket = url_arr[2]
    path = "/".join(url_arr[3:])
    
    return scheme, bucket, path, file


# TODO: From the experiment.ipynb copy-paste the model_export function 
def model_export_gcs(clf, model_dir):
    scheme, bucket, path, file = process_gcs_uri(model_dir)
    if scheme != "gs:":
            raise ValueError("URI scheme must be gs")
    # Write model to a local file
    
    # Upload the model to GCS
    bucket = storage.Client().bucket(bucket)
    blob = bucket.blob(path + '/model.joblib')
    
    blob.upload_from_string(pickle.dumps(clf))

    
# TODO: From the experiment.ipynb copy-paste the prepare_report function 
def prepare_report(cv_score: float, model_params: dict, classification_report: str, columns: list, example_data: numpy.ndarray) -> str:
    return """
Training Job Report    
    
Cross Validation Score: {cv_score}

Training Model Parameters: {model_params}
    
Test Data Classification Report:
{classification_report}

Example of data array for prediciton:

Order of columns:
{columns}

Example for clf.predict()
{predict_example}


Example of GCP API request body:
{{
    "instances": {json_example}
}}


Model parameters
""".format(
    cv_score=cv_score,
    model_params=json.dumps(model_params),
    classification_report=classification_report,
    columns = columns,
    predict_example = example_data,
    json_example = json.dumps(example_data.tolist()))


# TODO: From the experiment.ipynb copy-paste the report_export function 
def report_export_gcs(report, model_dir):
    scheme, bucket, path, file = process_gcs_uri(model_dir)
    if scheme != "gs:":
            raise ValueError("URI scheme must be gs")
            
    # Upload the model to GCS
    bucket = storage.Client().bucket(bucket)
    blob = bucket.blob(path + '/report.txt')
    
    blob.upload_from_string(report)

    
# Define all the command line arguments your model can accept for training
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # Input Arguments
    
    parser.add_argument(
        '--model_param_kernel',
        help = 'SVC model parameter- kernel',
        choices=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        type = str,
        default = 'linear'
    )
    
    parser.add_argument(
        '--model_param_degree',
        help = 'SVC model parameter- Degree. Only applies for poly kernel',
        type = int,
        default = 3
    )
    
    parser.add_argument(
        '--model_param_C',
        help = 'SVC model parameter- C (regularization)',
        type = float,
        default = 1.0
    )

    
    
    parser.add_argument(
        '--model_dir',
        help = 'Directory to output model and artifacts',
        type = str,
        default = os.environ['AIP_MODEL_DIR'] if 'AIP_MODEL_DIR' in os.environ else ""
    )
    parser.add_argument(
        '--data_format',
        choices=['csv', 'bigquery'],
        help = 'format of data uri csv for gs:// paths and bigquery for project.dataset.table formats',
        type = str,
        default =  os.environ['AIP_DATA_FORMAT'] if 'AIP_DATA_FORMAT' in os.environ else "csv"
    )
    parser.add_argument(
        '--training_data_uri',
        help = 'location of training data in either gs:// uri or bigquery uri',
        type = str,
        default =  os.environ['AIP_TRAINING_DATA_URI'] if 'AIP_TRAINING_DATA_URI' in os.environ else ""
    )
    parser.add_argument(
        '--validation_data_uri',
        help = 'location of validation data in either gs:// uri or bigquery uri',
        type = str,
        default =  os.environ['AIP_VALIDATION_DATA_URI'] if 'AIP_VALIDATION_DATA_URI' in os.environ else ""
    )
    parser.add_argument(
        '--test_data_uri',
        help = 'location of test data in either gs:// uri or bigquery uri',
        type = str,
        default =  os.environ['AIP_TEST_DATA_URI'] if 'AIP_TEST_DATA_URI' in os.environ else ""
    )
    
    
    args = parser.parse_args()
    arguments = args.__dict__
    
    
    logging.info('Model artifacts will be exported here: {}'.format(arguments['model_dir']))
    logging.info('Data format: {}'.format(arguments["data_format"]))
    logging.info('Training data uri: {}'.format(arguments['training_data_uri']) )
    logging.info('Validation data uri: {}'.format(arguments['validation_data_uri']))
    logging.info('Test data uri: {}'.format(arguments['test_data_uri']))
    
    
    if(arguments['data_format']=='csv'):
        df_train = load_data_from_gcs(arguments['training_data_uri'])
        df_valid = load_data_from_gcs(arguments['validation_data_uri'])
    elif(arguments['data_format']=='bigquery'):
        df_train = load_data_from_bq(arguments['training_data_uri'])
        df_valid = load_data_from_bq(arguments['validation_data_uri'])
    else:
        raise ValueError("Invalid data type ")
        
        
    model_params = dict()
    model_params['kernel'] = arguments['model_param_kernel']
    model_params['degree'] = arguments['model_param_degree']
    model_params['C'] = arguments['model_param_C']

    sort_missing_total_charges(df_train)
    sort_missing_total_charges(df_valid)

    X_train, y_train = data_selection(df_train)
    X_test, y_test = data_selection(df_valid)

    clf = pipeline_builder(model_params)

    cv_score = train_model(clf, X_train, y_train)

    model_export_gcs(clf, arguments['model_dir'])

    y_pred = clf.predict(X_test)
    
    
    report = prepare_report(cv_score,
                        model_params,
                        classification_report(y_test,y_pred),
                        ALL_COLUMNS, 
                        X_test.to_numpy()[0:2])
    
    report_export_gcs(report, arguments['model_dir'])