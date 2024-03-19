#!/bin/bash

# Initialize the database
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --password admin \
    --firstname Ad \
    --lastname Ministrator \
    --email admin@example.com \
    --role Admin

# Start scheduler in the background
airflow scheduler &

# Unpause all DAGs. This loops through all DAG IDs and unpauses them.
airflow dags list | grep -v "^ID" | awk '{print $1}' | while read dag_id; do
    airflow dags unpause "$dag_id"
done

# Start Airflow webserver
exec airflow webserver


