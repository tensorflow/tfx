#!/bin/bash
airflow webserver -p 8080  &
airflow scheduler  &
jupyter notebook --ip 0.0.0.0 --port 8000 --allow-root
