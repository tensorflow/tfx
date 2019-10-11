# Copyright 2019 Google LLC. All Rights Reserved.
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
"""Main entrypoint for containers with Kubeflow TFX component executors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import sys

import absl
from typing import List, Text

from ml_metadata.proto import metadata_store_pb2
from tfx.orchestration import data_types
from tfx.orchestration.kubeflow.proto import kubeflow_pb2
from tfx.orchestration.launcher import base_component_launcher
from tfx.utils import import_utils
from tfx.utils import json_utils
from google.protobuf import json_format


def _get_config_value(config_value: kubeflow_pb2.ConfigValue) -> Text:
  value_from = config_value.WhichOneof('value_from')

  if value_from is None:
    raise ValueError('No value set in config value: {}'.format(config_value))

  if value_from == 'value':
    return config_value.value

  return os.getenv(config_value.environment_variable)


# TODO(ajaygopinathan): Add unit tests for these helper functions.
def _get_metadata_connection_config(
    kubeflow_metadata_config: kubeflow_pb2.KubeflowMetadataConfig
) -> metadata_store_pb2.ConnectionConfig:
  """Constructs a metadata connection config.

  Args:
    kubeflow_metadata_config: Configuration parameters to use for constructing a
      valid metadata connection config in a Kubeflow cluster.

  Returns:
    A metadata_store_pb2.ConnectionConfig object.
  """
  connection_config = metadata_store_pb2.ConnectionConfig()

  connection_config.mysql.host = _get_config_value(
      kubeflow_metadata_config.mysql_db_service_host)
  connection_config.mysql.port = int(
      _get_config_value(kubeflow_metadata_config.mysql_db_service_port))
  connection_config.mysql.database = _get_config_value(
      kubeflow_metadata_config.mysql_db_name)
  connection_config.mysql.user = _get_config_value(
      kubeflow_metadata_config.mysql_db_user)
  connection_config.mysql.password = _get_config_value(
      kubeflow_metadata_config.mysql_db_password)

  return connection_config


def _make_beam_pipeline_args(json_beam_pipeline_args: Text) -> List[Text]:
  """Constructs beam_pipeline_args for ComponentLauncher.

  Args:
    json_beam_pipeline_args: JSON serialized list of beam pipeline args.

  Returns:
    List containing `beam_pipeline_args`.
  """
  beam_pipeline_args = json.loads(json_beam_pipeline_args)

  # Ensure beam pipelines args has a setup.py file so we can use
  # DataflowRunner.
  module_dir = os.environ['TFX_SRC_DIR']
  setup_file = os.path.join(module_dir, 'setup.py')
  absl.logging.info('Using setup_file \'%s\' to capture TFX dependencies',
                    setup_file)
  beam_pipeline_args.append('--setup_file={}'.format(setup_file))

  return beam_pipeline_args


def main():
  # Log to the container's stdout so Kubeflow Pipelines UI can display logs to
  # the user.
  logging.basicConfig(stream=sys.stdout, level=logging.INFO)
  logging.getLogger().setLevel(logging.INFO)

  parser = argparse.ArgumentParser()
  parser.add_argument('--pipeline_name', type=str, required=True)
  parser.add_argument('--pipeline_root', type=str, required=True)
  parser.add_argument('--kubeflow_metadata_config', type=str, required=True)
  parser.add_argument('--beam_pipeline_args', type=str, required=True)
  parser.add_argument('--additional_pipeline_args', type=str, required=True)
  parser.add_argument(
      '--component_launcher_class_path', type=str, required=True)
  parser.add_argument('--enable_cache', action='store_true')
  parser.add_argument('--serialized_component', type=str, required=True)

  args = parser.parse_args()

  component = json_utils.loads(args.serialized_component)
  component_launcher_class = import_utils.import_class_by_path(
      args.component_launcher_class_path)
  if not issubclass(component_launcher_class,
                    base_component_launcher.BaseComponentLauncher):
    raise TypeError(
        'component_launcher_class "%s" is not subclass of base_component_launcher.BaseComponentLauncher'
        % component_launcher_class)

  kubeflow_metadata_config = kubeflow_pb2.KubeflowMetadataConfig()
  json_format.Parse(args.kubeflow_metadata_config, kubeflow_metadata_config)
  connection_config = _get_metadata_connection_config(kubeflow_metadata_config)
  driver_args = data_types.DriverArgs(enable_cache=args.enable_cache)

  beam_pipeline_args = _make_beam_pipeline_args(args.beam_pipeline_args)

  additional_pipeline_args = json.loads(args.additional_pipeline_args)

  launcher = component_launcher_class.create(
      component=component,
      pipeline_info=data_types.PipelineInfo(
          pipeline_name=args.pipeline_name,
          pipeline_root=args.pipeline_root,
          run_id=os.environ['WORKFLOW_ID']),
      driver_args=driver_args,
      metadata_connection_config=connection_config,
      beam_pipeline_args=beam_pipeline_args,
      additional_pipeline_args=additional_pipeline_args)

  launcher.launch()


if __name__ == '__main__':
  main()
