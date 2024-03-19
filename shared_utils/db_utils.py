from sqlalchemy import create_engine, Table, Column, Integer, Float, String, MetaData, DateTime, select, func
from datagenerator import DataGenerator
from joblib import dump, load
import pandas as pd
from dotenv import load_dotenv
import os
import requests
import random

# Load environment variables from .env file
load_dotenv()

# Database credentials and connection details
USERNAME = os.getenv('USERNAME')
PASSWORD = os.getenv('PASSWORD')
HOST = os.getenv('HOST')
PORT = '5432'  # Default port for PostgreSQL
DATABASE = 'gn-rds'

RAND_MIN = 500
RAND_MAX = 1500
RECORD_LIMIT = 1000
FEATURES_TO_DROP = ['timestamp', 'y']

# Database connection URL
DATABASE_URL = f"postgresql+psycopg2://{USERNAME}:{PASSWORD}@{HOST}:{PORT}"

# Create an engine instance
engine = create_engine(DATABASE_URL, echo=True)

# Define metadata instance
metadata = MetaData()

# Define the table
my_table = Table('transactions', 
                    metadata,
                    Column('id', Integer, primary_key=True),
                    Column('x0', Float),
                    Column('x1', Float),
                    Column('x2', Float),
                    Column('x3', Float),
                    Column('timestamp', DateTime),
                    Column('y', Float),
                    Column('y_pred', Float)
                    )

# Create the table in the database if it does not exist
metadata.create_all(engine)

def generate_data(n_obs_to_generate:int = None):
    """
    Generate and score data.
    """

    if n_obs_to_generate is None:
        n_obs_to_generate = random.randint(RAND_MIN, RAND_MAX)

    # Generate data
    dg = DataGenerator()
    df = dg.generate_data(n_obs_to_generate)
    return df

def get_predictions(df):
    """
    Get prediction from API.
    """
    # Fetch the ML_API_URL environment variable, with a fallback in case it's not set
    ml_api_url = os.environ.get('ML_API_URL', 'http://localhost:5000/predict')
    
    data = {
        "data": df.values.tolist(),
        "columns": df.columns.to_list()
    }

    # Make a POST request to the predict endpoint
    response = requests.post(ml_api_url, json=data)

    predictions = None
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the prediction from the response
        predictions = response.json()
        print("Success")
    else:
        print("Failed to get prediction. Status code:", response.status_code, "Response:", response.text)

    return predictions

def score_data(df):
    """
    Score data -> this should call an API.
    """
    predictions = get_predictions(df.drop(FEATURES_TO_DROP, axis=1))
    df['y_pred'] = predictions
    return df

def insert_new_data(df):
    """
    Generate, score and insert new data.
    """
    # Insert data into table
    with engine.connect() as conn:
        # Begin a transaction
        with conn.begin():
            df.to_sql('transactions', con=conn, if_exists='append', index=False)

def retrieve_data():
    """
    Retrieve data.
    """

    # Use a context manager to ensure the connection is closed after use
    with engine.connect() as conn:
        # Begin a transaction
        with conn.begin():
            # Build a select statement
            select_statement = select(my_table)
            # Execute the query and fetch the result into a DataFrame
            df = pd.read_sql(select_statement, conn)
            
    return df

def delete_older_records(limit=50):
    """
    Delete older records.
    """

    records = metadata.tables['transactions']

    # Step 1: Identify the ID of the 50th most recent record
    subq = select(records.c.id).order_by(records.c.id.desc()).limit(limit).alias()
    cut_off_query = select(func.min(subq.c.id))
    count_query = select(func.count()).select_from(records)

    # Step 1: Identify the cut-off ID
    # Select the IDs of the 50 most recent records, ordered by ID in descending order
    # subq = select(my_table.c.id).order_by(my_table.c.id.desc()).limit(50).alias('subq')
    # # Calculate the minimum ID from these 50 to determine our cut-off
    # cut_off_query = select([func.min(subq.c.id)])
    
    with engine.connect() as connection:
        cut_off_id = connection.execute(cut_off_query).scalar()
                
        if cut_off_id:
            # Step 2: Delete all records older than the 50 most recent
            delete_query = my_table.delete().where(my_table.c.id < cut_off_id)
            connection.execute(delete_query)
            count = connection.execute(count_query).scalar()
            # connection.commit()
            print(f"Deleted records older than ID {cut_off_id}, table count {count}")
        else:
            print("No records to delete.")

def generate_and_score_data():
    """
    Generate and score data.
    """
    df = generate_data()
    scored_df = score_data(df)
    insert_new_data(scored_df)

if __name__ == '__main__':
    df = generate_data()
    scored_df = score_data(df)
    insert_new_data(scored_df)
    # generate_and_score_data()
    delete_older_records(limit=RECORD_LIMIT)
    df_new = retrieve_data()
    print(df_new)


