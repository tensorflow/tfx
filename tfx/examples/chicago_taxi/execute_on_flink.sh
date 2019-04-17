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

echo Starting pipeline on Flink...

if [ "${JOB_ENDPOINT:-unset}" == "unset" ]; then
  JOB_ENDPOINT="localhost:8099"
fi

SCRIPT=$1
# Remove the script from the arguments.
shift 1

# TODO(BEAM-6754): Utilize multicore in LOOPBACK environment.
#
# Note; We use 100 worker threads to mitigate the issue with scheduling work
# between Flink and Beam SdkHarness. Flink can process unlimited work items
# concurrently in a TaskManager while SdkHarness can only process 1 work item
# per worker thread. Having 100 threads will let 100 tasks execute concurrently
# avoiding scheduling issue in most cases. In case the threads are exhausted,
# beam print the relevant message in the log.
# TODO(BEAM-5167): Simplify this.
BEAM_ARGUMENTS="--runner PortableRunner \
                --job_endpoint $JOB_ENDPOINT \
                --experiments worker_threads=100 \
                --environment_type LOOPBACK "

# TODO(b/126725506): Utilize multiple cores on a machine.
# TODO(FLINK-10672): Obviate setting BATCH_FORCED.
FLINK_ARGUMENTS="--execution_mode_for_batch BATCH_FORCED "

echo "Executing: python $SCRIPT $BEAM_ARGUMENTS $FLINK_ARGUMENTS $@"
python $SCRIPT $BEAM_ARGUMENTS $FLINK_ARGUMENTS $@
