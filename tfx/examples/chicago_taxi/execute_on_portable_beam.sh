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

echo Starting pipeline on Portable Beam Runner...

if [ "${JOB_ENDPOINT:-unset}" == "unset" ]; then
  JOB_ENDPOINT="localhost:8099"
fi

SCRIPT=$1
# Remove the script from the arguments.
shift 1

# LINT.IfChange
# TODO(BEAM-6754): Utilize multicore in LOOPBACK environment.  # pylint: disable=g-bad-todo
# TODO(BEAM-5167): Use concurrency information from SDK Harness.  # pylint: disable=g-bad-todo
# TODO(BEAM-7199): Obviate the need for setting pre_optimize=all.  # pylint: disable=g-bad-todo
# TODO(b/126725506): Set the task parallelism based on cpu cores.

# Note; We use 100 worker threads to mitigate the issue with scheduling work
# between the Beam runner and SDK harness. Flink and Spark can process unlimited
# work items concurrently while SdkHarness can only process 1 work item per
# worker thread. Having 100 threads will let 100 tasks execute concurrently
# avoiding scheduling issue in most cases. In case the threads are exhausted,
# beam print the relevant message in the log.
BEAM_ARGUMENTS="--runner PortableRunner \
                --job_endpoint $JOB_ENDPOINT \
                --experiments worker_threads=100 \
                --experiments pre_optimize=all \
                --environment_type LOOPBACK "

# TODO(FLINK-10672): Obviate setting BATCH_FORCED.
FLINK_ARGUMENTS="--execution_mode_for_batch BATCH_FORCED "
# LINT.ThenChange(tfx/examples/chicago_taxi_pipeline/experimental/taxi_pipeline_flink.py)

echo "Executing: python $SCRIPT $BEAM_ARGUMENTS $FLINK_ARGUMENTS $@"
python $SCRIPT $BEAM_ARGUMENTS $FLINK_ARGUMENTS $@
