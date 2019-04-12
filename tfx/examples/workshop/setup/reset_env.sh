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


# Use this to completely nuke the pypi libraries that TFX requires
# and start with a 'clean' environment.  This will uninstall TF/TFX
# libraries and airflow libraries.
#
# It will not delete the Airflow install itself.  You'll want to delete
# ~/airflow on your own.
#


GREEN=$(tput setaf 2)
NORMAL=$(tput sgr0)

printf "${GREEN}Resetting TFX workshop${NORMAL}\n\n"

pip uninstall tensorflow
pip uninstall tfx
pip uninstall tensorflow-model-analysis
pip uninstall tensorflow-data-validation
pip uninstall tensorflow-metadata
pip uninstall tensorflow-transform
pip uninstall apache-airflow

printf "\n\n${GREEN}TFX workshop has been reset${NORMAL}\n"
printf "${GREEN}Remember to delete ~/airflow${NORMAL}\n"
