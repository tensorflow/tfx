#!/bin/bash

airflow initdb &
rm /home/airflow/airflow-webserver.pid &
airflow webserver -p 8080  &
airflow scheduler  &
jupyter notebook --ip 0.0.0.0 --port 8000 --allow-root --NotebookApp.token=""
