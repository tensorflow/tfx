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
set -x

echo Starting distributed TFDV stats computation and schema generation...

DATA_DIR=$(pwd)/data
OUTPUT_DIR=$DATA_DIR/flink_tfdv_output

JOB_ID="chicago-taxi-tfdv-$(date +%Y%m%d-%H%M%S)"
JOB_OUTPUT_PATH=$OUTPUT_DIR

# Variables needed for subsequent stages.
TFDV_OUTPUT_PATH=$JOB_OUTPUT_PATH
SCHEMA_PATH=$TFDV_OUTPUT_PATH/schema.pbtxt

echo Job output path: $JOB_OUTPUT_PATH
echo TFDV output path: $TFDV_OUTPUT_PATH

mkdir -p data/flink_tfdv_output

EXTRA_ARGS="--infer_schema \
            --stats_path $TFDV_OUTPUT_PATH/train_stats.tfrecord \
            --schema_path $SCHEMA_PATH \
            --save_main_session True \
            --input $DATA_DIR/train/data.csv " SCRIPT=tfdv_analyze_and_validate.py $(pwd)/execute_on_flink.sh

EXTRA_ARGS="--for_eval \
            --validate_stats \
            --stats_path $TFDV_OUTPUT_PATH/eval_stats.tfrecord \
            --schema_path $SCHEMA_PATH \
            --anomalies_path $TFDV_OUTPUT_PATH/anomalies.pbtxt \
            --save_main_session True \
            --input $DATA_DIR/eval/data.csv " SCRIPT=tfdv_analyze_and_validate.py $(pwd)/execute_on_flink.sh


echo
echo
echo "  TFDV_OUTPUT_PATH=$TFDV_OUTPUT_PATH"
echo "  SCHEMA_PATH=$SCHEMA_PATH"
