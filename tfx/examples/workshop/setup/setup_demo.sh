#!/bin/bash
# Copyright 2018 Google LLC
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

printf "${GREEN}Installing TFX workshop${NORMAL}\n\n"

printf "${GREEN}Refreshing setuptools to avoid _NamespacePath issues${NORMAL}\n"
pip uninstall setuptools -y && pip install setuptools

printf "${GREEN}Installing httplib2 for Beam compatibility${NORMAL}\n"
pip install httplib2==0.12.0

# TODO(b/135470014): Use range or pin for pip installs.
# TF/TFX prereqs
printf "${GREEN}Installing TensorFlow${NORMAL}\n"
pip install tensorflow==1.14.0

printf "${GREEN}Installing TFX${NORMAL}\n"
pip install tfx==0.14.0rc1

printf "${GREEN}Installing Google API Client${NORMAL}\n"
pip install google-api-python-client

printf "${GREEN}Installing required Jupyter version${NORMAL}\n"
pip install ipykernel
ipython kernel install --user --name=tfx
pip install --upgrade notebook==5.7.8
jupyter nbextension install --py --symlink --sys-prefix tensorflow_model_analysis
jupyter nbextension enable --py --sys-prefix tensorflow_model_analysis

printf "${GREEN}Installing packages used by the notebooks${NORMAL}\n"
pip install matplotlib
pip install papermill
pip install pandas
pip install networkx

# # Docker images
printf "${GREEN}Installing docker${NORMAL}\n"
pip install docker

# Airflow
# Set this to avoid the GPL version; no functionality difference either way
printf "${GREEN}Preparing environment for Airflow${NORMAL}\n"
export SLUGIFY_USES_TEXT_UNIDECODE=yes
printf "${GREEN}Installing Airflow${NORMAL}\n"
# TODO(b/136777165): Remove pinned version of Flask and Werkzeug
# after newer version of Airflow: see AIRFLOW-4900.
pip install apache-airflow==1.10.3 Flask==1.0.4 Werkzeug==0.14.1
printf "${GREEN}Initializing Airflow database${NORMAL}\n"
airflow initdb

# Adjust configuration
printf "${GREEN}Adjusting Airflow config${NORMAL}\n"
sed -i'.orig' 's/dag_dir_list_interval = 300/dag_dir_list_interval = 1/g' ~/airflow/airflow.cfg
sed -i'.orig' 's/job_heartbeat_sec = 5/job_heartbeat_sec = 1/g' ~/airflow/airflow.cfg
sed -i'.orig' 's/scheduler_heartbeat_sec = 5/scheduler_heartbeat_sec = 1/g' ~/airflow/airflow.cfg
sed -i'.orig' 's/dag_default_view = tree/dag_default_view = graph/g' ~/airflow/airflow.cfg
sed -i'.orig' 's/load_examples = True/load_examples = False/g' ~/airflow/airflow.cfg
sed -i'.orig' 's/max_threads = 2/max_threads = 1/g' ~/airflow/airflow.cfg

printf "${GREEN}Refreshing Airflow to pick up new config${NORMAL}\n"
airflow resetdb --yes
airflow initdb

# Copy Dag to ~/airflow/dags
mkdir -p ~/airflow/dags
cp dags/taxi_pipeline.py ~/airflow/dags/
cp dags/taxi_utils.py ~/airflow/dags/

# Copy the simple pipeline example and adjust for user's environment
cp ../../chicago_taxi_pipeline/taxi_pipeline_simple.py ~/airflow/dags/taxi_pipeline_solution.py
cp ../../chicago_taxi_pipeline/taxi_utils.py ~/airflow/dags/taxi_utils_solution.py
sed -i'.orig' "s/os.environ\['HOME'\], 'taxi'/os.environ\['HOME'\], 'airflow'/g" ~/airflow/dags/taxi_pipeline_solution.py
sed -i'.orig' "s/data\/simple/data\/taxi_data/g" ~/airflow/dags/taxi_pipeline_solution.py
sed -i'.orig' "s/taxi_utils.py/dags\/taxi_utils_solution.py/g" ~/airflow/dags/taxi_pipeline_solution.py
sed -i'.orig' "s/serving_model\/taxi_simple/saved_models\/taxi_solution/g" ~/airflow/dags/taxi_pipeline_solution.py
sed -i'.orig' "s/os.environ\['HOME'\], 'tfx'/_taxi_root, 'tfx'/g" ~/airflow/dags/taxi_pipeline_solution.py
sed -i'.orig' "s/chicago_taxi_simple/taxi_solution/g" ~/airflow/dags/taxi_pipeline_solution.py

# Copy data to ~/airflow/data
# TODO(b/130311020): Combine Chicago Taxi data files
cp -R data ~/airflow

printf "\n${GREEN}TFX workshop installed${NORMAL}\n"
