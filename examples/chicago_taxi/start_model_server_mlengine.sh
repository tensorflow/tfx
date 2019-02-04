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

if [ -z "$WORKING_DIR" ]; then
  echo WORKING_DIR was not set.  Typically this is set by the model trainer.
  echo You can run 'source train_mlengine.sh' to set it.
  echo Otherwise please set WORKING_DIR using:
  echo export WORKING_DIR=gs://bucket/output
  exit 1
fi

gcloud ml-engine models create chicago_taxi --regions us-central1
gsutil ls $WORKING_DIR/serving_model_dir/export/chicago-taxi/

# Pick out the directory containing the last trained model.
MODEL_BINARIES=$(gsutil ls $WORKING_DIR/serving_model_dir/export/chicago-taxi/ \
  | sort | grep '\/[0-9]*\/$' | tail -n1)

# LINT.IfChange
TF_VERSION=1.10
# LINT.ThenChange(setup.py)

gcloud ml-engine versions create v1 \
  --model chicago_taxi \
  --origin $MODEL_BINARIES \
  --runtime-version $TF_VERSION
