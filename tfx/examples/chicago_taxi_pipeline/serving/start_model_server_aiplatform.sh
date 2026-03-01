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

echo Running cloud serving...

# Dir for model exported for serving, e.g., gs://<bucket>/serving_model/chicago_taxi_pipeline_kubeflow
CLOUD_MODEL_DIR=$1

gsutil ls $CLOUD_MODEL_DIR

# Pick out the directory containing the last trained model.
MODEL_BINARIES=$(gsutil ls $CLOUD_MODEL_DIR \
  | sort | grep '\/[0-9]*\/$' | tail -n1)

echo latest model: $MODEL_BINARIES

MODEL_NAME=chicago_taxi

gcloud ai-platform models create $MODEL_NAME --regions us-central1

TF_VERSION=1.14

gcloud ai-platform versions create v1 \
  --model $MODEL_NAME \
  --origin $MODEL_BINARIES \
  --runtime-version $TF_VERSION
