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

import tensorflow as tf
from typing import Any, Dict, Text

from tfx.orchestration import component_launcher
from tfx.orchestration import data_types
from tfx.orchestration.kubeflow import config_utils
from tfx.orchestration.kubeflow.proto import kubeflow_pb2
from tfx.types import artifact_utils
from tfx.types import channel
from tfx.utils import import_utils
from google.protobuf import json_format


def _make_channel_dict(artifact_dict: Dict[Text, Text]
                      ) -> Dict[Text, channel.Channel]:
  """Makes a dictionary of artifact channels from a dictionary of artifacts.

  Args:
    artifact_dict: Dictionary of artifacts.

  Returns:
    Dictionary of artifact channels.

  Raises:
    RuntimeError: If list of artifacts is malformed.
  """
  channel_dict = {}
  for name, artifact_list in artifact_dict.items():
    if not artifact_list:
      raise RuntimeError(
          'Found empty list of artifacts for input/output named {}: {}'.format(
              name, artifact_list))
    type_name = artifact_list[0].type_name
    channel_dict[name] = channel.Channel(
        type_name=type_name, artifacts=artifact_list)

  return channel_dict


def _make_additional_pipeline_args(json_additional_pipeline_args: Text
                                  ) -> Dict[Text, Any]:
  """Constructs additional_pipeline_args for ComponentLauncher.

  Currently, this mainly involves parsing and constructing `beam_pipeline_args`.

  Args:
    json_additional_pipeline_args: JSON serialized dictionary of additional
      pipeline args.

  Returns:
    Dictionary containing `additional_pipeline_args`.
  """
  additional_pipeline_args = json.loads(json_additional_pipeline_args)

  # Ensure beam pipelines args has a setup.py file so we can use
  # DataflowRunner.
  beam_pipeline_args = additional_pipeline_args.get('beam_pipeline_args', [])

  module_dir = os.environ['TFX_SRC_DIR']
  setup_file = os.path.join(module_dir, 'setup.py')
  tf.logging.info('Using setup_file \'%s\' to capture TFX dependencies',
                  setup_file)
  beam_pipeline_args.append('--setup_file={}'.format(setup_file))

  additional_pipeline_args['beam_pipeline_args'] = beam_pipeline_args
  return additional_pipeline_args


def main():
  # Log to the container's stdout so Kubeflow Pipelines UI can display logs to
  # the user.
  logging.basicConfig(stream=sys.stdout, level=logging.INFO)
  logging.getLogger().setLevel(logging.INFO)

  parser = argparse.ArgumentParser()
  parser.add_argument('--pipeline_name', type=str, required=True)
  parser.add_argument('--pipeline_root', type=str, required=True)
  parser.add_argument('--kubeflow_metadata_config', type=str, required=True)
  parser.add_argument('--additional_pipeline_args', type=str, required=True)
  parser.add_argument('--component_id', type=str, required=True)
  parser.add_argument('--component_name', type=str, required=True)
  parser.add_argument('--component_type', type=str, required=True)
  parser.add_argument('--name', type=str, required=True)
  parser.add_argument('--driver_class_path', type=str, required=True)
  parser.add_argument('--executor_class_path', type=str, required=True)
  parser.add_argument('--inputs', type=str, required=True)
  parser.add_argument('--outputs', type=str, required=True)
  parser.add_argument('--exec_properties', type=str, required=True)
  parser.add_argument('--enable_cache', action='store_true')

  args = parser.parse_args()

  inputs = artifact_utils.parse_artifact_dict(args.inputs)
  input_dict = _make_channel_dict(inputs)

  outputs = artifact_utils.parse_artifact_dict(args.outputs)
  output_dict = _make_channel_dict(outputs)

  exec_properties = json.loads(args.exec_properties)

  driver_class = import_utils.import_class_by_path(args.driver_class_path)
  executor_class = import_utils.import_class_by_path(args.executor_class_path)

  kubeflow_metadata_config = kubeflow_pb2.KubeflowMetadataConfig()
  json_format.Parse(args.kubeflow_metadata_config, kubeflow_metadata_config)
  connection_config = config_utils.get_metadata_connection_config(
      kubeflow_metadata_config)

  component_info = data_types.ComponentInfo(
      component_type=args.component_type, component_id=args.component_id)

  driver_args = data_types.DriverArgs(enable_cache=args.enable_cache)

  additional_pipeline_args = _make_additional_pipeline_args(
      args.additional_pipeline_args)

  launcher = component_launcher.BaseComponentLauncher(
      component_info=component_info,
      driver_class=driver_class,
      executor_class=executor_class,
      input_dict=input_dict,
      output_dict=output_dict,
      exec_properties=exec_properties,
      pipeline_info=data_types.PipelineInfo(
          pipeline_name=args.pipeline_name,
          pipeline_root=args.pipeline_root,
          run_id=os.environ['WORKFLOW_ID']),
      driver_args=driver_args,
      metadata_connection_config=connection_config,
      additional_pipeline_args=additional_pipeline_args)

  launcher.launch()


if __name__ == '__main__':
  main()
