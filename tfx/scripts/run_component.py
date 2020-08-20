# Lint as: python2, python3
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
"""Script to invoke TFX components using simple command-line.

No backwards compatibility guarantees.
"""

# TODO(b/149535307): Remove __future__ imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
from typing import List, Text

from google.protobuf import json_format
from google.protobuf import message
from tfx.components.base import base_executor
from tfx.types import channel_utils
from tfx.utils import import_utils


def run_component(
    full_component_class_name: Text,
    temp_directory_path: Text = None,
    beam_pipeline_args: List[Text] = None,
    **arguments
):
  r"""Loads a component, instantiates it with arguments and runs its executor.

  The component class is instantiated, so the component code is executed,
  not just the executor code.

  To pass artifact URI, use <input_name>_uri argument name.
  To pass artifact property, use <input_name>_<property> argument name.
  Protobuf property values can be passed as JSON-serialized protobufs.

  # pylint: disable=line-too-long

  Example::

    # When run as a script:
    python3 scripts/run_component.py \
      --full-component-class-name tfx.components.StatisticsGen \
      --examples-uri gs://my_bucket/chicago_taxi_simple/CsvExamplesGen/examples/1/ \
      --examples-split-names '["train", "eval"]' \
      --output-uri gs://my_bucket/chicago_taxi_simple/StatisticsGen/output/1/

    # When run as a function:
    run_component(
      full_component_class_name='tfx.components.StatisticsGen',
      examples_uri='gs://my_bucket/chicago_taxi_simple/CsvExamplesGen/sxamples/1/',
      examples_split_names='["train", "eval"]',
      output_uri='gs://my_bucket/chicago_taxi_simple/StatisticsGen/output/1/',
    )

  Args:
    full_component_class_name: The component class name including module name.
    temp_directory_path: Optional. Temporary directory path for the executor.
    beam_pipeline_args: Optional. Arguments to pass to the Beam pipeline.
    **arguments: Key-value pairs with component arguments.
  """
  component_class = import_utils.import_class_by_path(full_component_class_name)

  component_arguments = {}

  for name, execution_param in component_class.SPEC_CLASS.PARAMETERS.items():
    argument_value = arguments.get(name, None)
    if argument_value is None:
      continue
    param_type = execution_param.type
    if (isinstance(param_type, type) and
        issubclass(param_type, message.Message)):
      argument_value_obj = param_type()
      json_format.Parse(argument_value, argument_value_obj)
    else:
      argument_value_obj = argument_value
    component_arguments[name] = argument_value_obj

  for input_name, channel_param in component_class.SPEC_CLASS.INPUTS.items():
    uri = (arguments.get(input_name + '_uri') or
           arguments.get(input_name + '_path'))
    if uri:
      artifact = channel_param.type()
      artifact.uri = uri
      # Setting the artifact properties
      for property_name in channel_param.type.PROPERTIES:
        property_arg_name = input_name + '_' + property_name
        if property_arg_name in arguments:
          setattr(artifact, property_name, arguments[property_arg_name])
      component_arguments[input_name] = channel_utils.as_channel([artifact])

  component_instance = component_class(**component_arguments)

  input_dict = channel_utils.unwrap_channel_dict(
      component_instance.inputs.get_all())
  output_dict = channel_utils.unwrap_channel_dict(
      component_instance.outputs.get_all())
  exec_properties = component_instance.exec_properties

  # Generating paths for output artifacts
  for output_name, artifacts in output_dict.items():
    uri = (arguments.get('output_' + output_name + '_uri') or
           arguments.get(output_name + '_uri') or
           arguments.get(output_name + '_path'))
    if uri:
      for artifact in artifacts:
        artifact.uri = uri

  executor_context = base_executor.BaseExecutor.Context(
      beam_pipeline_args=beam_pipeline_args,
      tmp_dir=temp_directory_path,
      unique_id='',
  )
  executor = component_instance.executor_spec.executor_class(executor_context)
  executor.Do(
      input_dict=input_dict,
      output_dict=output_dict,
      exec_properties=exec_properties,
  )


if __name__ == '__main__':
  params = sys.argv[1::2]
  values = sys.argv[2::2]
  args = {
      param.lstrip('-').replace('-', '_'): value
      for param, value in zip(params, values)
  }
  run_component(**args)
