# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main entrypoint for orchestrator container on Kubernetes."""

import argparse
import logging
import sys

from tfx.orchestration.experimental.kubernetes import kubernetes_dag_runner
from tfx.orchestration.experimental.kubernetes import kubernetes_remote_runner


def main():
  # Log to the container's stdout so it can be streamed by the client.
  logging.basicConfig(stream=sys.stdout, level=logging.INFO)
  logging.getLogger().setLevel(logging.INFO)

  parser = argparse.ArgumentParser()

  # Pipeline is serialized via a json format.
  # See kubernetes_remote_runner._serialize_pipeline for details.
  parser.add_argument('--serialized_pipeline', type=str, required=True)
  parser.add_argument('--tfx_image', type=str, required=True)
  args = parser.parse_args()

  kubernetes_dag_runner.KubernetesDagRunner(
      config=kubernetes_dag_runner.KubernetesDagRunnerConfig(
          tfx_image=args.tfx_image)).run(
              kubernetes_remote_runner.deserialize_pipeline(
                  args.serialized_pipeline))


if __name__ == '__main__':
  main()
