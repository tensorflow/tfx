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

echo Starting distributed TFT preprocessing...

if [ -z "$MYBUCKET" ]; then
  echo MYBUCKET was not set
  echo Please set MYBUCKET to your GCP bucket using:
  echo export MYBUCKET=gs://bucket
  exit 1
fi

if [ -z "$SCHEMA_PATH" ]; then
  echo SCHEMA_PATH was not set. Please set SCHEMA_PATH to schema produced
  echo by tfdv_analyze_and_validate_dataflow.sh using:
  echo export SCHEMA_PATH=gs://...
  exit 1
fi

JOB_ID="chicago-taxi-preprocess-$(date +%Y%m%d-%H%M%S)"
TEMP_PATH=$MYBUCKET/$JOB_ID/tmp/
MYPROJECT=$(gcloud config list --format 'value(core.project)' 2>/dev/null)

# Variables needed for subsequent stages.
export JOB_OUTPUT_PATH=$MYBUCKET/$JOB_ID/chicago_taxi_output
export TFT_OUTPUT_PATH=$JOB_OUTPUT_PATH/tft_output

echo Using GCP project: $MYPROJECT
echo Job output path: $JOB_OUTPUT_PATH
echo TFT output path: $TFT_OUTPUT_PATH

echo Preprocessing train data...

python preprocess.py \
  --output_dir $TFT_OUTPUT_PATH \
  --outfile_prefix train_transformed \
  --input bigquery-public-data.chicago_taxi_trips.taxi_trips \
  --schema_file $SCHEMA_PATH \
  --project $MYPROJECT \
  --temp_location $TEMP_PATH \
  --job_name $JOB_ID \
  --setup_file ./setup.py \
  --runner DataflowRunner


echo
echo
echo "  JOB_OUTPUT_PATH=$JOB_OUTPUT_PATH"
echo "  TFT_OUTPUT_PATH=$TFT_OUTPUT_PATH"
echo
