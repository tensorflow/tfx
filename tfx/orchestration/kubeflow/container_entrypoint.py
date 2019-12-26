# Lint as: python2, python3
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
import textwrap
from typing import Dict, List, Text

import absl

from google.protobuf import json_format
from ml_metadata.proto import metadata_store_pb2
from tfx.components.base import base_component
from tfx.components.trainer import component as trainer_component
from tfx.orchestration import data_types
from tfx.orchestration.kubeflow.proto import kubeflow_pb2
from tfx.orchestration.launcher import base_component_launcher
from tfx.types import artifact
from tfx.types import channel
from tfx.utils import import_utils
from tfx.utils import json_utils


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


def _render_channel_as_mdstr(input_channel: channel.Channel) -> Text:
  """Render a Channel as markdown string with the following format.

  **Type**: input_channel.type_name
  **Artifact: artifact1**
  **Properties**:
  **key1**: value1
  **key2**: value2
  ......

  Args:
    input_channel: the channel to be rendered.

  Returns:
    a md-formatted string representation of the channel.
  """

  md_str = '**Type**: {}\n\n'.format(input_channel.type_name)
  rendered_artifacts = []
  # List all artifacts in the channel.
  for single_artifact in input_channel.get():
    rendered_artifacts.append(_render_artifact_as_mdstr(single_artifact))

  return md_str + '\n\n'.join(rendered_artifacts)


def _render_artifact_as_mdstr(single_artifact: artifact.Artifact) -> Text:
  """Render an artifact as markdown string with the following format.

  **Artifact: artifact1**
  **Properties**:
  **key1**: value1
  **key2**: value2
  ......

  Args:
    single_artifact: the artifact to be rendered.

  Returns:
    a md-formatted string representation of the artifact.
  """
  return textwrap.dedent("""\
      **Artifact: {name}**

      **Properties**:

      **uri**: {uri}

      **id**: {id}

      **span**: {span}

      **type_id**: {type_id}

      **type_name**: {type_name}

      **state**: {state}

      **split_names**: {split_names}

      **producer_component**: {producer_component}

      """.format(
          name=single_artifact.name or 'None',
          uri=single_artifact.uri or 'None',
          id=str(single_artifact.id),
          span=single_artifact.span or 'None',
          type_id=str(single_artifact.type_id),
          type_name=single_artifact.type_name,
          state=single_artifact.state or 'None',
          split_names=single_artifact.split_names or 'None',
          producer_component=single_artifact.producer_component or 'None'))


def _dump_ui_metadata(component: base_component.BaseComponent,
                      execution_info: data_types.ExecutionInfo) -> None:
  """Dump KFP UI metadata json file for visualization purpose.

  For general components we just render a simple Markdown file for
    exec_properties/inputs/outputs.

  Args:
    component: associated TFX component.
    execution_info: runtime execution info for this component, including
      materialized inputs/outputs/execution properties and id.
  """
  exec_properties_list = [
      '**{}**: {}'.format(name, exec_property)
      for name, exec_property in execution_info.exec_properties.items()
  ]
  src_str_exec_properties = '# Execution properties:\n{}'.format(
      '\n\n'.join(exec_properties_list) or 'No execution property.')

  def _dump_populated_artifacts(
      name_to_channel: Dict[Text, channel.Channel],
      name_to_artifacts: Dict[Text, List[artifact.Artifact]]) -> List[Text]:
    """Dump artifacts markdown string.

    Args:
      name_to_channel: maps from channel name to channel object.
      name_to_artifacts: maps from channel name to list of populated artifacts.

    Returns:
      A list of dumped markdown string, each of which represents a channel.
    """
    rendered_list = []
    for name, chnl in name_to_channel.items():
      # Need to look for materialized artifacts in the execution decision.
      rendered_artifacts = ''.join([
          _render_artifact_as_mdstr(single_artifact)
          for single_artifact in name_to_artifacts[name]
      ])
      rendered_list.append(
          '## {name}\n\n**Type**: {channel_type}\n\n{artifacts}'.format(
              name=name,
              channel_type=chnl.type_name,
              artifacts=rendered_artifacts))

    return rendered_list

  src_str_inputs = '# Inputs:\n{}'.format(''.join(
      _dump_populated_artifacts(
          name_to_channel=component.inputs.get_all(),
          name_to_artifacts=execution_info.input_dict)) or 'No input.')

  src_str_outputs = '# Outputs:\n{}'.format(''.join(
      _dump_populated_artifacts(
          name_to_channel=component.outputs.get_all(),
          name_to_artifacts=execution_info.output_dict)) or 'No output.')

  outputs = [{
      'storage':
          'inline',
      'source':
          '{exec_properties}\n{inputs}\n{outputs}'.format(
              exec_properties=src_str_exec_properties,
              inputs=src_str_inputs,
              outputs=src_str_outputs),
      'type':
          'markdown',
  }]
  # Add Tensorboard view for Trainer.
  # TODO(b/142804764): Visualization based on component type seems a bit of
  # arbitrary. We need a better way to improve this.
  if isinstance(component, trainer_component.Trainer):
    output_model = component.outputs['model'].get()[0]

    # Add Tensorboard view.
    tensorboard_output = {'type': 'tensorboard', 'source': output_model.uri}
    outputs.append(tensorboard_output)

  metadata = {'outputs': outputs}

  with open('/mlpipeline-ui-metadata.json', 'w') as f:
    json.dump(metadata, f)


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
      additional_pipeline_args=additional_pipeline_args,
      component_config=component_config)

  execution_info = launcher.launch()

  # Dump the UI metadata.
  _dump_ui_metadata(component, execution_info)


if __name__ == '__main__':
  main()
