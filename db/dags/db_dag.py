from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Hack to use relative imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Now you can import the module from the parent directory
from rds import generate_and_score_data, delete_older_records

MAX_RECORDS = 20000

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 3, 11),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}

with DAG(dag_id="db_tasks_dag", 
         default_args=default_args,
         schedule_interval='* * * * *',
         catchup=False,
         max_active_runs=1) as tasks_dag:

    task1 = PythonOperator(
        task_id='generate_and_score_data',
        python_callable=generate_and_score_data
    )

    task2 = PythonOperator(
        task_id='delete_older_records',
        python_callable=delete_older_records,
        op_kwargs={'limit': MAX_RECORDS}
    )

    task1 >> task2