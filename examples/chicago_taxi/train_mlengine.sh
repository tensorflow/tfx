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

echo Starting distributed training...

JOB_ID="chicago_taxi_trainer_$(date +%Y%m%d_%H%M%S)"

if [ -z "$JOB_OUTPUT_PATH" ]; then
  echo JOB_OUTPUT_PATH was not set.
  echo Please set JOB_OUTPUT_PATH to the job output path used by
  echo preprocess_dataflow.sh using export JOB_OUTPUT_PATH=gs://bucket/output
  exit 1
fi

if [ -z "$TFT_OUTPUT_PATH" ]; then
  echo TFT_OUTPUT_PATH was not set.
  echo Please set TFT_OUTPUT_PATH to the TFT output path used by
  echo preprocess_dataflow.sh using export TFT_OUTPUT_PATH=gs://bucket/output
  exit 1
fi

if [ -z "$SCHEMA_PATH" ]; then
  echo SCHEMA_PATH was not set. Please set SCHEMA_PATH to schema produced
  echo by tfdv_analyze_and_validate_local.sh using: export SCHEMA_PATH=...
  exit 1
fi

# Variables needed for subsequent stages.
export TRAIN_OUTPUT_PATH=$JOB_OUTPUT_PATH/trainer_output
export WORKING_DIR=$TRAIN_OUTPUT_PATH/working_dir

MODEL_DIR=$TRAIN_OUTPUT_PATH/model_dir

echo Working directory: $WORKING_DIR
echo Model directory: $MODEL_DIR

# Start clean
gsutil rm $TRAIN_OUTPUT_PATH

# Inputs
TRAIN_FILE=$TFT_OUTPUT_PATH/train_transformed-*

# Force a small eval so that the Estimator.train_and_eval() can be used to
# save the model with its standard paths.
EVAL_FILE=$TFT_OUTPUT_PATH/train_transformed-*

# Options
TRAIN_STEPS=100000
EVAL_STEPS=1000

# LINT.IfChange
TF_VERSION=1.10
# LINT.ThenChange(setup.py)

gcloud ml-engine jobs submit training $JOB_ID \
                                    --stream-logs \
                                    --job-dir $MODEL_DIR \
                                    --runtime-version $TF_VERSION \
                                    --module-name trainer.task \
                                    --package-path trainer/ \
                                    --region us-central1 \
                                    -- \
                                    --train-files $TRAIN_FILE \
                                    --train-steps $TRAIN_STEPS \
                                    --eval-files $EVAL_FILE \
                                    --eval-steps $EVAL_STEPS \
                                    --output-dir $WORKING_DIR \
                                    --schema-file $SCHEMA_PATH \
                                    --tf-transform-dir $TFT_OUTPUT_PATH


echo
echo
echo "  TRAIN_OUTPUT_PATH=$TRAIN_OUTPUT_PATH"
echo "  WORKING_DIR=$WORKING_DIR"
echo
