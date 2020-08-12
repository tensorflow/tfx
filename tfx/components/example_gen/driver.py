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
"""Generic TFX ExampleGen custom driver."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Any, Dict, List, Text
from absl import logging

from google.protobuf import json_format
from tfx import types
from tfx.components.base import base_driver
from tfx.components.example_gen import utils
from tfx.orchestration import data_types
from tfx.proto import example_gen_pb2
from tfx.types import artifact_utils
from tfx.types import channel_utils


class Driver(base_driver.BaseDriver):
  """Custom driver for ExampleGen.

  This driver supports file based ExampleGen, e.g., for CsvExampleGen and
  ImportExampleGen.
  """

  def resolve_exec_properties(
      self,
      exec_properties: Dict[Text, Any],
      pipeline_info: data_types.PipelineInfo,
      component_info: data_types.ComponentInfo,
  ) -> Dict[Text, Any]:
    """Overrides BaseDriver.resolve_exec_properties()."""
    del pipeline_info, component_info

    input_config = example_gen_pb2.Input()
    json_format.Parse(exec_properties[utils.INPUT_CONFIG_KEY], input_config)

    input_base = exec_properties[utils.INPUT_BASE_KEY]
    logging.debug('Processing input %s.', input_base)

    # Note that this function updates the input_config.splits.pattern.
    fingerprint, span, version = utils.calculate_splits_fingerprint_span_and_version(
        input_base, input_config.splits)

    exec_properties[utils.INPUT_CONFIG_KEY] = json_format.MessageToJson(
        input_config, sort_keys=True, preserving_proto_field_name=True)
    exec_properties[utils.SPAN_PROPERTY_NAME] = span
    exec_properties[utils.VERSION_PROPERTY_NAME] = version
    exec_properties[utils.FINGERPRINT_PROPERTY_NAME] = fingerprint

    return exec_properties

  def _prepare_output_artifacts(
      self,
      input_artifacts: Dict[Text, List[types.Artifact]],
      output_dict: Dict[Text, types.Channel],
      exec_properties: Dict[Text, Any],
      execution_id: int,
      pipeline_info: data_types.PipelineInfo,
      component_info: data_types.ComponentInfo,
  ) -> Dict[Text, List[types.Artifact]]:
    """Overrides BaseDriver._prepare_output_artifacts()."""
    del input_artifacts

    result = channel_utils.unwrap_channel_dict(output_dict)
    if len(result) != 1:
      raise RuntimeError('Multiple output artifacts are not supported.')

    base_output_dir = os.path.join(pipeline_info.pipeline_root,
                                   component_info.component_id)

    example_artifact = artifact_utils.get_single_instance(
        result[utils.EXAMPLES_KEY])
    example_artifact.uri = base_driver._generate_output_uri(  # pylint: disable=protected-access
        base_output_dir, utils.EXAMPLES_KEY, execution_id)
    example_artifact.set_string_custom_property(
        utils.FINGERPRINT_PROPERTY_NAME,
        exec_properties[utils.FINGERPRINT_PROPERTY_NAME])
    example_artifact.set_string_custom_property(
        utils.SPAN_PROPERTY_NAME,
        str(exec_properties[utils.SPAN_PROPERTY_NAME]))
    # TODO(b/162622803): add default behavior for when version spec not present.
    if exec_properties[utils.VERSION_PROPERTY_NAME]:
      example_artifact.set_string_custom_property(
          utils.VERSION_PROPERTY_NAME,
          str(exec_properties[utils.VERSION_PROPERTY_NAME]))

    base_driver._prepare_output_paths(example_artifact)  # pylint: disable=protected-access

    return result
