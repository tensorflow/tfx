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
EXTRA_ARGS=$2

# TODO: Utilize multicore in LOOPBACK environment https://issues.apache.org/jira/browse/BEAM-6754
# Note; We use 100 worker threads to mitigate the issue with scheduling work between Flink and Beam
# SdkHarness. Flink can process unlimited work items concurrently in a TaskManager while SdkHarness
# can only process 1 work item per worker thread. Having 100 threads will let 100 tasks execute
# concurrently avoiding scheduling issue in most cases. In case the threads are exhausted, beam
# print the relevant message in the log.
# Reference Jira https://issues.apache.org/jira/browse/BEAM-5167
BEAM_ARGUMENTS="--runner PortableRunner \
                --job_endpoint $JOB_ENDPOINT \
                --experiments worker_threads=100 \
                --environment_type LOOPBACK "

# TODO: Utilize multiple cores on a machine. b/126725506
# TODO: Using BATCH_FORCED because Flink scheduling has bug
# tracked in https://issues.apache.org/jira/browse/FLINK-10672 .
FLINK_ARGUMENTS="--execution_mode_for_batch BATCH_FORCED "

echo "Executing: python $SCRIPT $BEAM_ARGUMENTS $FLINK_ARGUMENTS $EXTRA_ARGS"
python $SCRIPT $BEAM_ARGUMENTS $FLINK_ARGUMENTS $EXTRA_ARGS
