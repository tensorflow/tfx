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

DATA_DIR=./data
OUTPUT_DIR=$DATA_DIR/local_tfdv_output
export SCHEMA_PATH=$OUTPUT_DIR/schema.pbtxt

echo Starting local TFDV preprocessing...

# Compute stats on the train file and generate a schema based on the stats.
rm -R -f $OUTPUT_DIR
python tfdv_analyze_and_validate.py \
  --input $DATA_DIR/train/data.csv \
  --stats_path $OUTPUT_DIR/train_stats.tfrecord \
  --infer_schema \
  --schema_path $SCHEMA_PATH \
  --runner DirectRunner

# Compute stats on the eval file and validate against the training schema.
python tfdv_analyze_and_validate.py \
  --input $DATA_DIR/eval/data.csv \
  --stats_path $OUTPUT_DIR/eval_stats.tfrecord \
  --validate_stats \
  --schema_path $SCHEMA_PATH \
  --anomalies_path $OUTPUT_DIR/anomalies.pbtxt \
  --runner DirectRunner
