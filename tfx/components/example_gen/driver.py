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

from typing import Any, Dict, List, Optional, Text

from absl import logging
from tfx import types
from tfx.components.example_gen import utils
from tfx.dsl.components.base import base_driver
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration.portable import base_driver as ir_base_driver
from tfx.orchestration.portable.mlmd import common_utils
from tfx.proto import example_gen_pb2
from tfx.proto import range_config_pb2
from tfx.proto.orchestration import driver_output_pb2
from tfx.proto.orchestration import pipeline_pb2

from google.protobuf import json_format


class Driver(base_driver.BaseDriver, ir_base_driver.BaseDriver):
  """Custom driver for ExampleGen.

  This driver supports file based ExampleGen, e.g., for CsvExampleGen and
  ImportExampleGen.
  """

  def __init__(self,
               metadata_handler: metadata.Metadata,
               pipeline_info: Optional[pipeline_pb2.PipelineInfo] = None,
               pipeline_node: Optional[pipeline_pb2.PipelineNode] = None):
    base_driver.BaseDriver.__init__(self, metadata_handler)
    ir_base_driver.BaseDriver.__init__(self, metadata_handler, pipeline_info,
                                       pipeline_node)

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

    range_config = None
    range_config_entry = exec_properties.get(utils.RANGE_CONFIG_KEY)
    if range_config_entry:
      range_config = range_config_pb2.RangeConfig()
      json_format.Parse(range_config_entry, range_config)

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

    exec_properties[utils.INPUT_CONFIG_KEY] = json_format.MessageToJson(
        input_config, sort_keys=True, preserving_proto_field_name=True)
    exec_properties[utils.SPAN_PROPERTY_NAME] = span
    exec_properties[utils.VERSION_PROPERTY_NAME] = version
    exec_properties[utils.FINGERPRINT_PROPERTY_NAME] = fingerprint

    return exec_properties

  def run(self, input_dict: Dict[Text, List[types.Artifact]],
          output_dict: Dict[Text, List[types.Artifact]],
          exec_properties: Dict[Text, Any]) -> driver_output_pb2.DriverOutput:

    # Populate exec_properties
    result = driver_output_pb2.DriverOutput()
    # PipelineInfo and ComponentInfo are not actually used, two fake one are
    # created just to be compatable with the old API.
    pipeline_info = data_types.PipelineInfo('', '')
    component_info = data_types.ComponentInfo('', '', pipeline_info)
    exec_properties = self.resolve_exec_properties(
        exec_properties, pipeline_info, component_info)
    for k, v in exec_properties.items():
      if v is not None:
        common_utils.set_metadata_value(result.exec_properties[k], v)
    return result
