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

echo Starting local TFT preprocessing...

# Preprocess the train files, keeping the transform functions
echo Preprocessing train data...
rm -R -f ./data/train/local_chicago_taxi_output
python preprocess.py \
  --input ./data/train/data.csv \
  --schema_file ./data/local_tfdv_output/schema.pbtxt \
  --output_dir ./data/train/local_chicago_taxi_output \
  --outfile_prefix train_transformed \
  --runner DirectRunner

# Preprocess the eval files
echo Preprocessing eval data...
rm -R -f ./data/eval/local_chicago_taxi_output
python preprocess.py \
  --input ./data/eval/data.csv \
  --schema_file ./data/local_tfdv_output/schema.pbtxt \
  --output_dir ./data/eval/local_chicago_taxi_output \
  --outfile_prefix eval_transformed \
  --transform_dir ./data/train/local_chicago_taxi_output \
  --runner DirectRunner
