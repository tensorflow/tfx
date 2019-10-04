#!/bin/bash
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -u

echo Running Google Cloud AI Platform inference...

EXAMPLES_FILE=$1
SCHEMA_FILE=$2

# Matches the MODEL_NAME in start_model_server_aiplatform.sh
MODEL_NAME=chicago_taxi

python `dirname "$(readlink -f "$0")"`/chicago_taxi_client.py \
  --num_examples 3 \
  --examples_file $EXAMPLES_FILE \
  --schema_file $SCHEMA_FILE \
  --server aiplatform:$MODEL_NAME:v1
