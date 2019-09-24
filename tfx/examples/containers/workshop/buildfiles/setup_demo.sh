#!/bin/bash
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Set up the environment for the TFX tutorial


GREEN=$(tput setaf 2)
NORMAL=$(tput sgr0)

source /root/tfx_env/bin/activate

airflow initdb

# Adjust configuration
sed -i'.orig' 's/dag_dir_list_interval = 300/dag_dir_list_interval = 1/g' /root/airflow/airflow.cfg
sed -i'.orig' 's/job_heartbeat_sec = 5/job_heartbeat_sec = 1/g' /root/airflow/airflow.cfg
sed -i'.orig' 's/scheduler_heartbeat_sec = 5/scheduler_heartbeat_sec = 1/g' /root/airflow/airflow.cfg
sed -i'.orig' 's/dag_default_view = tree/dag_default_view = graph/g' /root/airflow/airflow.cfg
sed -i'.orig' 's/load_examples = True/load_examples = False/g' /root/airflow/airflow.cfg
sed -i'.orig' 's/max_threads = 2/max_threads = 1/g' /root/airflow/airflow.cfg

airflow resetdb --yes
airflow initdb

# Copy Dag to /airflow/dags
mkdir /root/airflow/dags
cp /root/tfx/tfx/examples/airflow_workshop/setup/dags/taxi_pipeline.py /root/airflow/dags/
cp /root/tfx/tfx/examples/airflow_workshop/setup/dags/taxi_utils.py /root/airflow/dags/

# Copy the simple pipeline example and adjust for user's environment
cp /root/tfx/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_simple.py /root/airflow/dags/taxi_pipeline_solution.py
cp /root/tfx/tfx/examples/chicago_taxi_pipeline/taxi_utils.py /root/airflow/dags/taxi_utils_solution.py
sed -i'.orig' "s/os.environ\['HOME'\], 'taxi'/os.environ\['HOME'\], 'airflow'/g" /root/airflow/dags/taxi_pipeline_solution.py
sed -i'.orig' "s/_taxi_root, 'data', 'simple'/_taxi_root, 'data', 'taxi_data'/g" /root/airflow/dags/taxi_pipeline_solution.py
sed -i'.orig' "s/taxi_utils.py/dags\/taxi_utils_solution.py/g" /root/airflow/dags/taxi_pipeline_solution.py
sed -i'.orig' "s/os.environ\['HOME'\], 'tfx'/_taxi_root, 'tfx'/g" /root/airflow/dags/taxi_pipeline_solution.py
sed -i'.orig' "s/chicago_taxi_simple/taxi_solution/g" /root/airflow/dags/taxi_pipeline_solution.py

# Copy data to /airflow/data
cp -R /root/tfx/tfx/examples/airflow_workshop/setup/data /root/airflow

mkdir /root/logs

printf "${GREEN}Starting Jupyter notebook${NORMAL}\n"
cd /root/tfx/tfx/examples/airflow_workshop/notebooks
jupyter notebook --generate-config
printf "${GREEN}\nPlease create a password for Jupyter notebook (or return for no password).  You'll use this to login.${NORMAL}\n"
jupyter notebook password
jupyter notebook --ip=0.0.0.0 --no-browser --allow-root &> /root/logs/notebook &

printf "${GREEN}Starting Airflow webserver${NORMAL}\n"
airflow webserver &> /root/logs/webserver &

printf "${GREEN}Starting Airflow scheduler${NORMAL}\n"
airflow scheduler &> /root/logs/scheduler &

printf "${GREEN}\nYou can now open two web browser tabs and go to:${NORMAL}\n"
printf "${GREEN}Airflow: http://127.0.0.1:8080${NORMAL}\n"
printf "${GREEN}Jupyter: http://127.0.0.1:8888${NORMAL}\n"
