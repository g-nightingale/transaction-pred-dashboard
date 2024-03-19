# Transaction Prediction Dashboard
Example of a webapp using microservices architecture to predict transaction amounts using 4 features. Docker Compose has been used to build the containers.

The webapp consists of the following services:
- Front End
    - A web UI built using Flask and Plotly.
    - The web UI gets the latest information from the database to display to the user.
- Database
    - Data is generated and stored in an Amazon RDS database.
    - Concept drift is built into the data generation process to ensure that models degrade.
    - Airflow is used to orchestrate data generation and DB management activities.
- ML Model
    - Transaction predictions are via a linear regression model, trained using scikit-learn.
    - MLFlow is used to log and register models.
    - Models predictions and information are served via API using Flask.
    - Models are retrained and deployed once the RMSE breaches a predetermined threshold.