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
  # Log to the container's stdout so it can be streamed by the client.
  logging.basicConfig(stream=sys.stdout, level=logging.INFO)
  logging.getLogger().setLevel(logging.INFO)

  parser = argparse.ArgumentParser()

  # Pipeline is serialized via a json format.
  # See kubernetes_remote_runner._serialize_pipeline for details.
  parser.add_argument('--serialized_pipeline', type=str, required=True)
  parser.add_argument('--downstream_ids', type=str, required=True)
  parser.add_argument('--tfx_image', type=str, required=True)
  args = parser.parse_args()

  tfx_pipeline = json.loads(args.serialized_pipeline)
  components = [
      json_utils.loads(component) for component in tfx_pipeline['components']
    ]
  metadata_connection_config = metadata_store_pb2.ConnectionConfig()
  json_format.Parse(tfx_pipeline['metadata_connection_config'],
                    metadata_connection_config)

  # Restore component dependencies.
  downstream_ids = json.loads(args.downstream_ids)
  if not isinstance(downstream_ids, list):
    raise RuntimeError("downstream_ids needs to be a 'dict'.")
  if len(downstream_ids) != len(components):
    raise RuntimeError(
        'Wrong number of elements in downstream_ids. Expected: %s. Actual: %s' %
        len(components), len(downstream_ids))

  id_to_component = {component.id: component for component in components}
  for component in components:
    # Since downstream and upstream node attributes are discarded during the
    # serialization process, we initialize them here.
    component._upstream_nodes = set() # pylint: disable=protected-access
    component._downstream_nodes = set() # pylint: disable=protected-access

  for ind, component in enumerate(components):
    for downstream_id in downstream_ids[ind]:
      component.add_downstream_node(id_to_component[downstream_id])
      id_to_component[downstream_id].add_upstream_node(component)

  absl.logging.set_verbosity(absl.logging.INFO)
  kubernetes_dag_runner.KubernetesDagRunner(
      config=kubernetes_dag_runner.KubernetesDagRunnerConfig(
          tfx_image=args.tfx_image)
  ).run(
      pipeline.Pipeline(
          pipeline_name=tfx_pipeline['pipeline_name'],
          pipeline_root=tfx_pipeline['pipeline_root'],
          components=components,
          enable_cache=tfx_pipeline['enable_cache'],
          metadata_connection_config=metadata_connection_config,
          beam_pipeline_args=tfx_pipeline['beam_pipeline_args'],
      )
  )


if __name__ == '__main__':
  main()
