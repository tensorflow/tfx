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

# Input: trained model to be evaluated
WORKING_DIR=./data/train/local_chicago_taxi_output
EVAL_MODEL_DIR=$WORKING_DIR/eval_model_dir/$(ls $WORKING_DIR/eval_model_dir | tail -n1)

# Output: evaluation result
EVAL_RESULT_DIR=$WORKING_DIR/eval_result

echo Eval model dir: $EVAL_MODEL_DIR
echo Eval result dir: $EVAL_RESULT_DIR

# Start clean.
rm -R -f $EVAL_RESULT_DIR > /dev/null

python process_tfma.py \
  --eval_model_dir $EVAL_MODEL_DIR \
  --eval_result_dir $EVAL_RESULT_DIR \
  --schema_file ./data/local_tfdv_output/schema.pbtxt \
  --input_csv ./data/eval/data.csv

echo Done
echo Eval results written to $EVAL_RESULT_DIR
