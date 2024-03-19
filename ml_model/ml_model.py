import mlflow
from sqlalchemy import create_engine, Table, Column, Integer, Float, String, MetaData, DateTime, select, func
import pandas as pd
from dotenv import load_dotenv
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import mlflow.sklearn
from flask import Flask, request, jsonify
from mlflow.tracking import MlflowClient
from datetime import datetime, timedelta
import os
import sys

# Hack to use relative imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../shared_utils'))
sys.path.append(parent_dir)

# Now you can import the module from the parent directory
from db_utils import my_table, retrieve_data

# Load environment variables from .env file
# load_dotenv()

# # Database credentials and connection details
# USERNAME = os.getenv('USERNAME')
# PASSWORD = os.getenv('PASSWORD')
# HOST = os.getenv('HOST')
# PORT = '5432'  # Default port for PostgreSQL
# DATABASE = 'gn-rds'

# # Database connection URL
# DATABASE_URL = f"postgresql+psycopg2://{USERNAME}:{PASSWORD}@{HOST}:{PORT}"

# # Create an engine instance
# engine = create_engine(DATABASE_URL, echo=True)

# # Define metadata instance
# metadata = MetaData()

# # Define the table -> need to centralise this somehow
# my_table = Table('transactions', 
#                     metadata,
#                     Column('id', Integer, primary_key=True),
#                     Column('x0', Float),
#                     Column('x1', Float),
#                     Column('x2', Float),
#                     Column('x3', Float),
#                     Column('timestamp', DateTime),
#                     Column('y', Float),
#                     Column('y_pred', Float)
#                     )

EXPERIMENT_NAME = 'mlflow-lr_model'
REGISTERED_MODEL_NAME = 'lr_model'
MIN_MSE_REDUCTION = -0.1
CLEANUP_HOURS = 5

app = Flask(__name__)

def load_latest_model(model_name):
    client = MlflowClient()
    latest_version_info = client.get_latest_versions(model_name, stages=["None"])  # Newly registered models have the stage "None" by default
    if latest_version_info:
        latest_version = latest_version_info[0].version
        model_uri = f"models:/{model_name}/{latest_version}"
        model = mlflow.sklearn.load_model(model_uri)
    else:
        model = None
    return model

from datetime import datetime

def convert_timestamp_to_string(timestamp_ms):
    """
    Converts a timestamp in milliseconds to a human-readable date and time string.

    Parameters:
    - timestamp_ms (int): The timestamp in milliseconds since the epoch.

    Returns:
    - str: The human-readable date and time.
    """
    # Convert milliseconds to seconds
    timestamp_s = timestamp_ms / 1000.0
    
    # Convert to datetime object
    dt_object = datetime.fromtimestamp(timestamp_s)
    
    # Format the datetime object to a string (e.g., "2024-03-16 12:34:56")
    # You can adjust the format string according to your needs
    date_time_str = dt_object.strftime("%Y-%m-%d %H:%M:%S")
    
    return date_time_str


def get_latest_model_info(model_name):
    """
    Fetches the most recent version of the specified model from MLflow,
    along with its training date and time, and RMSE value.

    Parameters:
    - model_name (str): The name of the model in the MLflow Model Registry.

    Returns:
    - dict: A dictionary containing the model version, training date and time, and RMSE value.
    """
    # Set your MLflow tracking URI if it's not the default
    # mlflow.set_tracking_uri('your_mlflow_tracking_uri')

    client = mlflow.tracking.MlflowClient()
    
    # Fetch the latest version of the model
    latest_versions = client.get_latest_versions(model_name, stages=["None", "Staging", "Production", "Archived"])
    if not latest_versions:
        return {"error": "Model not found or no versions available."}
    
    # Assuming you want the latest version regardless of stage
    latest_version = latest_versions[-1]
    
    # Fetch the run associated with the latest model version
    run = client.get_run(latest_version.run_id)
    
    # Extract training date, time, and RMSE value
    training_date_time = convert_timestamp_to_string(run.info.start_time)
    rmse_value = round(run.data.metrics.get("rmse", "RMSE metric not found"), 2)
    
    return {
        "model_version": latest_version.version,
        "training_date_time": training_date_time,
        "rmse": rmse_value
    }

def cleanup():
    # Define the age threshold for deletion
    age_threshold = timedelta(hours=CLEANUP_HOURS)
    current_time = datetime.now()

    # List all experiments
    experiments = mlflow.list_experiments()

    for experiment in experiments:
        runs = mlflow.search_runs([experiment.experiment_id])
        for run in runs.iterrows():
            run_info = run[1]
            end_time = datetime.fromtimestamp(run_info['end_time'] / 1000)
            if current_time - end_time > age_threshold:
                mlflow.delete_run(run_info['run_id'])

@app.route('/retrain', methods=['POST'])
def retrain():
    # Get data from RDS and train model
    df = retrieve_data()

    x = df[['x0', 'x1', 'x2', 'x3']]
    y = df['y']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    # Initialize and train the linear regression model
    try:
        current_model = load_latest_model(REGISTERED_MODEL_NAME)
        current_model_test_preds = current_model.predict(X_test)
        current_mse = mean_squared_error(y_test, current_model_test_preds)
    except:
        current_mse = None

    
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        new_model = LinearRegression()
        new_model.fit(X_train, y_train)
        new_model_test_preds = new_model.predict(X_test)

        # Compare predictions
        new_mse = mean_squared_error(y_test, new_model_test_preds)
        rmse = new_mse ** 0.5
        
        # Only save the new model if it outperforms the old model by some threshold
        if current_mse is None or (new_mse / current_mse) - 1 < MIN_MSE_REDUCTION:
            mlflow.sklearn.log_model(new_model, 
                                    "linear_regression_model", 
                                    registered_model_name=REGISTERED_MODEL_NAME)
            mlflow.log_metric("rmse", rmse)
                    
        message_str = f"Model retrained \n old mse: {current_mse}\n new mse: {new_mse}"

    # Run cleanup routine
    # cleanup()
    
    return jsonify({"message": message_str}), 200

@app.route('/predict', methods=['POST'])
def predict():
    # Fetch the latest version of the registered model
    try:
        model = load_latest_model(REGISTERED_MODEL_NAME)
    except:
        print('No model found: retraining...')
        _ = retrain()
        model = load_latest_model(REGISTERED_MODEL_NAME)
    
    data = request.get_json(force=True)
    prediction = model.predict(pd.DataFrame(data['data'], columns=data['columns']))
    
    return jsonify(prediction.tolist())

@app.route('/get-model-info', methods=['GET'])
def get_model_info():
    info = get_latest_model_info(REGISTERED_MODEL_NAME)
    return jsonify(info)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
    # Example usage
    # retrain()
    # info = get_latest_model_info(REGISTERED_MODEL_NAME)
    # print(info)