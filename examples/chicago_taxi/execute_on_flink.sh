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

echo Starting pipeline on Flink...

if [ "${JOB_ENDPOINT:-unset}" == "unset" ]; then
  JOB_ENDPOINT="localhost:8099"
fi

image="$(whoami)-docker-apache.bintray.io/beam/python"

python $SCRIPT \
  --setup_file ./setup.py \
  --experiments=beam_fn_api \
  --runner PortableRunner \
  --job_endpoint=$JOB_ENDPOINT \
  --experiments=worker_threads=100 \
  --environment_type=PROCESS \
  --parallelism=1 \
  --environment_config='{"command": "'"$(pwd)"'/process_worker.sh"}' \
  --execution_mode_for_batch=BATCH_FORCED\
  $EXTRA_ARGS
