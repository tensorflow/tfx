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
"""Main entrypoint for driver containers with Kubernetes TFX pipeline executors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import sys
import absl

from google.protobuf import json_format
from ml_metadata.proto import metadata_store_pb2
from tfx.orchestration import pipeline
from tfx.orchestration.experimental.kubernetes import kubernetes_dag_runner
from tfx.utils import json_utils



def main():
  # Log to the container's stdout so Kubeflow Pipelines UI can display logs to
  # the user.
  logging.basicConfig(stream=sys.stdout, level=logging.INFO)
  logging.getLogger().setLevel(logging.INFO)

  parser = argparse.ArgumentParser()

  # pipeline is serialized via a json format 
  parser.add_argument('--serialized_pipeline', type=str, required=True)

  args = parser.parse_args()

  _tfx_pipeline = json.loads(args.serialized_pipeline)
  _components = [json_utils.loads(component) for component in _tfx_pipeline['components']]
  _metadata_connection_config = metadata_store_pb2.ConnectionConfig()
  json_format.Parse(_tfx_pipeline['metadata_connection_config'], _metadata_connection_config)

  absl.logging.set_verbosity(absl.logging.INFO)
  kubernetes_dag_runner.KubernetesDagRunner().run(
    pipeline.Pipeline(
        pipeline_name=_tfx_pipeline['pipeline_name'],
        pipeline_root=_tfx_pipeline['pipeline_root'],
        components=_components,
        sort_components=False,
        enable_cache=_tfx_pipeline['enable_cache'],
        metadata_connection_config=_metadata_connection_config,
        beam_pipeline_args=_tfx_pipeline['beam_pipeline_args'],
    )
  )


if __name__ == '__main__':
  main()
