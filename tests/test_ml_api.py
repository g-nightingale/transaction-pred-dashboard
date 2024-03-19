import requests
import numpy as np
import pandas as pd

def retrain_model():
    response = requests.post('http://localhost:5002/retrain')
    print(response.json())

def get_prediction():
    # The URL of the predict endpoint
    url = 'http://localhost:5002/predict'

    # Example input data and column names
    df = pd.DataFrame({
        'x0': np.random.randint(1, 101, size=4).tolist(),
        'x1': np.random.randint(1, 101, size=4).tolist(),
        'x2': np.random.randint(1, 101, size=4).tolist(),
        'x3': np.random.randint(1, 101, size=4).tolist(),   
    })
    
    data = {
        "data": df.values.tolist(),  # Replace these values with your actual input data
        "columns": df.columns.to_list()  # Replace with your actual column names
    }

    # Make a POST request to the predict endpoint
    response = requests.post(url, json=data)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the prediction from the response
        prediction = response.json()
        print("Prediction:", prediction)
    else:
        print("Failed to get prediction. Status code:", response.status_code, "Response:", response.text)


if __name__ == '__main__':
    retrain = False

    if retrain:
        retrain_model()
    else:
        get_prediction()