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

JOB_ID="chicago-taxi-tfma-eval-$(date +%Y%m%d-%H%M%S)"

if [ -z "$TFT_OUTPUT_PATH" ]; then
  echo TFT_OUTPUT_PATH was not set
  echo Please set TFT_OUTPUT_PATH using:
  echo export TFT_OUTPUT_PATH=gs://bucket/output
  exit 1
fi

if [ -z "$SCHEMA_PATH" ]; then
  echo SCHEMA_PATH was not set. Please set SCHEMA_PATH to schema produced
  echo by tfdv_analyze_and_validate_dataflow.sh using:
  echo export SCHEMA_PATH=gs://...
  exit 1
fi

EVAL_RESULT_DIR=$TFT_OUTPUT_PATH/eval_result_dir
MYPROJECT=$(gcloud config list --format 'value(core.project)' 2>/dev/null)

# We evaluate with the last eval model written (hence tail -n1)
EVAL_MODEL_DIR=$TRAIN_OUTPUT_PATH/working_dir/eval_model_dir
LAST_EVAL_MODEL_DIR=$(gsutil ls $EVAL_MODEL_DIR | tail -n1)

echo Eval model dir: $EVAL_MODEL_DIR

python process_tfma.py \
  --big_query_table bigquery-public-data.chicago_taxi_trips.taxi_trips \
  --schema_file $SCHEMA_PATH \
  --eval_model_dir $LAST_EVAL_MODEL_DIR \
  --eval_result_dir $EVAL_RESULT_DIR \
  --project $MYPROJECT \
  --temp_location $MYBUCKET/$JOB_ID/tmp/ \
  --job_name $JOB_ID \
  --setup_file ./setup.py \
  --save_main_session True \
  --runner DataflowRunner
