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
from sklearn.metrics import classification_report, f1_score
import pickle
from google.cloud import storage
from datetime import datetime
import os
import pandas as pd
import logging
import numpy as np
from typing import Union, List
import json

# Helps parsing input arguments
import argparse
import hypertune

# feature categories

# List all binary features: 0,1 or True,Fales or Male,Female etc
BINARY_FEATURES = [
    'gender',
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'PhoneService',
    'PaperlessBilling']

# List all numeric features
NUMERIC_FEATURES = [
    'tenure',
    'MonthlyCharges',
    'TotalCharges']

# List all categorical features 
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

LABEL = 'Churn'

# We define the index position of each feature. This will be needed when we wil be processing a 
# numpy array (instead of pandas) that has no column names.
BINARY_FEATURES_IDX = list(range(0,len(BINARY_FEATURES)))
NUMERIC_FEATURES_IDX = list(range(len(BINARY_FEATURES), len(BINARY_FEATURES)+len(NUMERIC_FEATURES)))
CATEGORICAL_FEATURES_IDX = list(range(len(BINARY_FEATURES+NUMERIC_FEATURES), len(ALL_COLUMNS)))


# TODO: From the experiment.ipynb copy-paste the load_data_from_gcs function

# TODO: From the experiment.ipynb copy-paste the load_data_from_bq function 

# TODO: From the experiment.ipynb copy-paste the sort_missing_total_charges function 
    
# TODO: From the experiment.ipynb copy-paste the data_selection function

# TODO: From the experiment.ipynb copy-paste the pipeline_builder function 

# TODO: From the experiment.ipynb copy-paste the train_pipeline function 

# TODO: From the experiment.ipynb copy-paste the process_gcs_uri function 

# TODO: From the experiment.ipynb copy-paste the pipeline_export_gcs function

# TODO: From the experiment.ipynb copy-paste the prepare_report function 

# TODO: From the experiment.ipynb copy-paste the report_export_gcs function



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
    
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")

    
    
    args = parser.parse_args()
    arguments = args.__dict__
    
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
        
    logging.info('Model artifacts will be exported here: {}'.format(arguments['model_dir']))
    logging.info('Data format: {}'.format(arguments["data_format"]))
    logging.info('Training data uri: {}'.format(arguments['training_data_uri']) )
    logging.info('Validation data uri: {}'.format(arguments['validation_data_uri']))
    logging.info('Test data uri: {}'.format(arguments['test_data_uri']))
    
    
    logging.info('Loading {} data'.format(arguments["data_format"]))
    if(arguments['data_format']=='csv'):
        df_train = load_data_from_gcs(arguments['training_data_uri'])
        df_valid = load_data_from_gcs(arguments['validation_data_uri'])
    elif(arguments['data_format']=='bigquery'):
        print(arguments['training_data_uri'])
        df_train = load_data_from_bq(arguments['training_data_uri'])
        df_valid = load_data_from_bq(arguments['validation_data_uri'])
    else:
        raise ValueError("Invalid data type ")
        
    
    logging.info('Defining model parameters')    
    model_params = dict()
    model_params['kernel'] = arguments['model_param_kernel']
    model_params['degree'] = arguments['model_param_degree']
    model_params['C'] = arguments['model_param_C']

    sort_missing_total_charges(df_train)
    sort_missing_total_charges(df_valid)

    
    logging.info('Running feature selection')    
    X_train, y_train = data_selection(df_train, ALL_COLUMNS, LABEL)
    X_test, y_test = data_selection(df_valid, ALL_COLUMNS, LABEL)

    logging.info('Training pipelines in CV')   
    clf = pipeline_builder(model_params, BINARY_FEATURES_IDX, NUMERIC_FEATURES_IDX, CATEGORICAL_FEATURES_IDX)

    cv_score = train_pipeline(clf, X_train, y_train)
    
    
    logging.info('Export trained pipeline and report')   
    pipeline_export_gcs(clf, arguments['model_dir'])

    y_pred = clf.predict(X_test)
        
    
    test_score = f1_score(y_test, y_pred, average='weighted')
    
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag='f1score',
    metric_value=test_score)
    
    
    logging.info('f1score: '+ str(test_score))    
    
    report = prepare_report(cv_score,
                        model_params,
                        classification_report(y_test,y_pred),
                        ALL_COLUMNS, 
                        X_test.to_numpy()[0:2])
    
    report_export_gcs(report, arguments['model_dir'])
    
    
    logging.info('Train completed. Exiting...')