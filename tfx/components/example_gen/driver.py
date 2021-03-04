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

import copy
import os
from typing import Any, Dict, List, Text

from absl import logging
from tfx import types
from tfx.components.example_gen import utils
from tfx.dsl.components.base import base_driver
from tfx.orchestration import data_types
from tfx.orchestration import data_types_utils
from tfx.orchestration import metadata
from tfx.orchestration.portable import base_driver as ir_base_driver
from tfx.orchestration.portable import data_types as portable_data_types
from tfx.proto import example_gen_pb2
from tfx.proto import range_config_pb2
from tfx.proto.orchestration import driver_output_pb2
from tfx.types import standard_component_specs
from tfx.utils import proto_utils

from ml_metadata.proto import metadata_store_pb2


def update_output_artifact(
    exec_properties: Dict[Text, Any],
    output_artifact: metadata_store_pb2.Artifact) -> None:
  """Updates output_artifact for FileBasedExampleGen.

  Updates output_artifact properties by updating existing entries or creating
  new entries if not already exists.

  Args:
    exec_properties: execution properties passed to the example gen.
    output_artifact: the example artifact to be output.
  """
  output_artifact.custom_properties[
      utils.FINGERPRINT_PROPERTY_NAME].string_value = (
          exec_properties[utils.FINGERPRINT_PROPERTY_NAME])
  output_artifact.custom_properties[
      utils.SPAN_PROPERTY_NAME].string_value = str(
          exec_properties[utils.SPAN_PROPERTY_NAME])
  # TODO(b/162622803): add default behavior for when version spec not present.
  if exec_properties[utils.VERSION_PROPERTY_NAME] is not None:
    output_artifact.custom_properties[
        utils.VERSION_PROPERTY_NAME].string_value = str(
            exec_properties[utils.VERSION_PROPERTY_NAME])


class Driver(base_driver.BaseDriver, ir_base_driver.BaseDriver):
  """Custom driver for ExampleGen.

  This driver supports file based ExampleGen, e.g., for CsvExampleGen and
  ImportExampleGen.
  """

  def __init__(self, metadata_handler: metadata.Metadata):
    base_driver.BaseDriver.__init__(self, metadata_handler)
    ir_base_driver.BaseDriver.__init__(self, metadata_handler)

  def resolve_exec_properties(
      self,
      exec_properties: Dict[Text, Any],
      pipeline_info: data_types.PipelineInfo,
      component_info: data_types.ComponentInfo,
  ) -> Dict[Text, Any]:
    """Overrides BaseDriver.resolve_exec_properties()."""
    del pipeline_info, component_info

    input_config = example_gen_pb2.Input()
    proto_utils.json_to_proto(
        exec_properties[standard_component_specs.INPUT_CONFIG_KEY],
        input_config)

    input_base = exec_properties[standard_component_specs.INPUT_BASE_KEY]
    logging.debug('Processing input %s.', input_base)

    range_config = None
    range_config_entry = exec_properties.get(
        standard_component_specs.RANGE_CONFIG_KEY)
    if range_config_entry:
      range_config = range_config_pb2.RangeConfig()
      proto_utils.json_to_proto(range_config_entry, range_config)

      if range_config.HasField('static_range'):
        # For ExampleGen, StaticRange must specify an exact span to look for,
        # since only one span is processed at a time.
        start_span_number = range_config.static_range.start_span_number
        end_span_number = range_config.static_range.end_span_number
        if start_span_number != end_span_number:
          raise ValueError(
              'Start and end span numbers for RangeConfig.static_range must '
              'be equal: (%s, %s)' % (start_span_number, end_span_number))

    # Note that this function updates the input_config.splits.pattern.
    fingerprint, span, version = utils.calculate_splits_fingerprint_span_and_version(
        input_base, input_config.splits, range_config)

    exec_properties[standard_component_specs
                    .INPUT_CONFIG_KEY] = proto_utils.proto_to_json(input_config)
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

    example_artifact = output_dict[standard_component_specs.EXAMPLES_KEY].type()
    base_output_dir = os.path.join(pipeline_info.pipeline_root,
                                   component_info.component_id)

    example_artifact.uri = base_driver._generate_output_uri(  # pylint: disable=protected-access
        base_output_dir, standard_component_specs.EXAMPLES_KEY, execution_id)
    update_output_artifact(exec_properties, example_artifact.mlmd_artifact)
    base_driver._prepare_output_paths(example_artifact)  # pylint: disable=protected-access

    return {standard_component_specs.EXAMPLES_KEY: [example_artifact]}

  def run(
      self, execution_info: portable_data_types.ExecutionInfo
  ) -> driver_output_pb2.DriverOutput:

    # Populate exec_properties
    result = driver_output_pb2.DriverOutput()
    # PipelineInfo and ComponentInfo are not actually used, two fake one are
    # created just to be compatible with the old API.
    pipeline_info = data_types.PipelineInfo('', '')
    component_info = data_types.ComponentInfo('', '', pipeline_info)
    exec_properties = self.resolve_exec_properties(
        execution_info.exec_properties, pipeline_info, component_info)
    for k, v in exec_properties.items():
      if v is not None:
        data_types_utils.set_metadata_value(result.exec_properties[k], v)

    # Populate output_dict
    output_example = copy.deepcopy(execution_info.output_dict[
        standard_component_specs.EXAMPLES_KEY][0].mlmd_artifact)
    update_output_artifact(exec_properties, output_example)
    result.output_artifacts[
        standard_component_specs.EXAMPLES_KEY].artifacts.append(output_example)
    return result
