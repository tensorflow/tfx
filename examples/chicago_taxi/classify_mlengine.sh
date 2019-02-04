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

echo Running cloud inference...

if [ -z "$SCHEMA_PATH" ]; then
  echo SCHEMA_PATH was not set. Please set SCHEMA_PATH to schema produced
  echo by tfdv_analyze_and_validate_dataflow.sh using:
  echo export SCHEMA_PATH=gs://...
  exit 1
fi

python chicago_taxi_client.py \
  --num_examples 3 \
  --examples_file ./data/train/data.csv \
  --schema_file $SCHEMA_PATH \
  --server mlengine:chicago_taxi:v1
