#!/bin/bash
# Copyright 2020 Google LLC. All Rights Reserved.
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

printf "${GREEN}Installing pendulum to avoid problem with tzlocal${NORMAL}\n"
pip install pendulum

printf "${GREEN}Installing TFX${NORMAL}\n"
pip install pyarrow==5.0.0
pip install apache_beam==2.38.0
pip install google-api-core==1.31.6
pip install google-api-python-client==1.12.11
pip install google-apitools==0.5.31
pip install google-auth==1.35.0
pip install google-auth-httplib2==0.1.0
pip install google-auth-oauthlib==0.4.6
pip install google-cloud-aiplatform==1.15.0
pip install google-cloud-appengine-logging==1.1.2
pip install google-cloud-audit-log==0.2.2
pip install google-cloud-bigquery==2.34.4
pip install google-cloud-bigquery-storage==2.13.2
pip install google-cloud-bigtable==1.7.2
pip install google-cloud-core==1.7.3
pip install google-cloud-dataproc==4.0.3
pip install google-cloud-datastore==1.15.5
pip install google-cloud-dlp==3.7.1
pip install google-cloud-firestore==2.5.3
pip install google-cloud-kms==2.11.2
pip install google-cloud-language==1.3.2
pip install google-cloud-logging==3.1.2
pip install google-cloud-monitoring==2.9.2
pip install google-cloud-pubsub==2.13.0
pip install google-cloud-pubsublite==1.4.2
pip install google-cloud-recommendations-ai==0.2.0
pip install google-cloud-resource-manager==1.5.1
pip install google-cloud-scheduler==2.6.4
pip install google-cloud-spanner==1.19.3
pip install google-cloud-speech==2.14.1
pip install google-cloud-storage==2.2.1
pip install google-cloud-tasks==2.9.1
pip install google-cloud-translate==3.7.4
pip install google-cloud-videointelligence==1.16.3
pip install google-cloud-vision==1.0.2
pip install google-crc32c==1.1.2
pip install google-pasta==0.2.0
pip install tfx==1.8.0
pip install numpy==1.20


printf "${GREEN}Installing required packages for tft${NORMAL}\n"
pip install tensorflow-text==2.8.1 tensorflow_decision_forests==0.2.4 struct2tensor==0.39.0

ipython kernel install --user --name=tfx

jupyter labextension install tensorflow_model_analysis
jupyter lab build --dev-build=False --minimize=False

# Airflow
# Set this to avoid the GPL version; no functionality difference either way
printf "${GREEN}Preparing environment for Airflow${NORMAL}\n"
export SLUGIFY_USES_TEXT_UNIDECODE=yes
printf "${GREEN}Installing Airflow${NORMAL}\n"
pip install -q apache-airflow==2.2.5 Flask Werkzeug
printf "${GREEN}Initializing Airflow database${NORMAL}\n"
airflow db init

# Adjust configuration
printf "${GREEN}Adjusting Airflow config${NORMAL}\n"
sed -i'.orig' 's/dag_dir_list_interval = 300/dag_dir_list_interval = 1/g' ~/airflow/airflow.cfg
sed -i'.orig' 's/job_heartbeat_sec = 5/job_heartbeat_sec = 1/g' ~/airflow/airflow.cfg
sed -i'.orig' 's/scheduler_heartbeat_sec = 5/scheduler_heartbeat_sec = 1/g' ~/airflow/airflow.cfg
sed -i'.orig' 's/dag_default_view = tree/dag_default_view = graph/g' ~/airflow/airflow.cfg
sed -i'.orig' 's/load_examples = True/load_examples = False/g' ~/airflow/airflow.cfg
sed -i'.orig' 's/max_threads = 2/max_threads = 1/g' ~/airflow/airflow.cfg

printf "${GREEN}Refreshing Airflow to pick up new config${NORMAL}\n"
airflow db reset --yes
airflow db init

# Copy Dags to ~/airflow/dags
mkdir -p ~/airflow/dags
cp dags/taxi_pipeline.py ~/airflow/dags/
cp dags/taxi_utils.py ~/airflow/dags/

# Copy data to ~/airflow/data
cp -R data ~/airflow

printf "\n${GREEN}TFX workshop installed${NORMAL}\n"
