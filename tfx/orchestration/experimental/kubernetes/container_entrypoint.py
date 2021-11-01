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
"""Main entrypoint for containers with Kubernetes TFX component executors."""

import argparse
import json
import logging
import sys

from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration.launcher import base_component_launcher
from tfx.utils import import_utils
from tfx.utils import json_utils
from tfx.utils import telemetry_utils

from google.protobuf import json_format
from ml_metadata.proto import metadata_store_pb2


def main():
  # Log to the container's stdout so it can be streamed by the orchestrator.
  logging.basicConfig(stream=sys.stdout, level=logging.INFO)
  logging.getLogger().setLevel(logging.INFO)

  parser = argparse.ArgumentParser()
  parser.add_argument('--pipeline_name', type=str, required=True)
  parser.add_argument('--pipeline_root', type=str, required=True)
  parser.add_argument('--run_id', type=str, required=True)
  parser.add_argument('--metadata_config', type=str, required=True)
  parser.add_argument('--beam_pipeline_args', type=str, required=True)
  parser.add_argument('--additional_pipeline_args', type=str, required=True)
  parser.add_argument(
      '--component_launcher_class_path', type=str, required=True)
  parser.add_argument('--enable_cache', action='store_true')
  parser.add_argument('--serialized_component', type=str, required=True)
  parser.add_argument('--component_config', type=str, required=True)

  args = parser.parse_args()

  component = json_utils.loads(args.serialized_component)
  component_config = json_utils.loads(args.component_config)
  component_launcher_class = import_utils.import_class_by_path(
      args.component_launcher_class_path)
  if not issubclass(component_launcher_class,
                    base_component_launcher.BaseComponentLauncher):
    raise TypeError(
        'component_launcher_class "%s" is not subclass of base_component_launcher.BaseComponentLauncher'
        % component_launcher_class)

  metadata_config = metadata_store_pb2.ConnectionConfig()
  json_format.Parse(args.metadata_config, metadata_config)
  driver_args = data_types.DriverArgs(enable_cache=args.enable_cache)
  beam_pipeline_args = json.loads(args.beam_pipeline_args)
  additional_pipeline_args = json.loads(args.additional_pipeline_args)

  launcher = component_launcher_class.create(
      component=component,
      pipeline_info=data_types.PipelineInfo(
          pipeline_name=args.pipeline_name,
          pipeline_root=args.pipeline_root,
          run_id=args.run_id,
      ),
      driver_args=driver_args,
      metadata_connection=metadata.Metadata(connection_config=metadata_config),
      beam_pipeline_args=beam_pipeline_args,
      additional_pipeline_args=additional_pipeline_args,
      component_config=component_config)

  # Attach necessary labels to distinguish different runner and DSL.
  with telemetry_utils.scoped_labels({
      telemetry_utils.LABEL_TFX_RUNNER: 'kubernetes',
  }):
    launcher.launch()


if __name__ == '__main__':
  main()
