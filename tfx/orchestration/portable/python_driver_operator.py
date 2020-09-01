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
"""A class to define how to operator an python based driver."""

from typing import Any, Dict, List, Text, cast

from tfx import types
from tfx.orchestration import metadata
from tfx.orchestration.portable import base_driver_operator
from tfx.proto.orchestration import driver_output_pb2
from tfx.proto.orchestration import executable_spec_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import import_utils

from google.protobuf import message


class PythonDriverOperator(base_driver_operator.BaseDriverOperator):
  """PythonDriverOperator handles python class based driver's init and execution."""

  SUPPORTED_EXECUTABLE_SPEC_TYPE = [
      executable_spec_pb2.PythonClassExecutableSpec
  ]

  def __init__(self, driver_spec: message.Message,
               mlmd_connection: metadata.Metadata,
               pipeline_info: pipeline_pb2.PipelineInfo,
               pipeline_node: pipeline_pb2.PipelineNode):
    """Constructor.

    Args:
      driver_spec: The specification of how to initialize the driver.
      mlmd_connection: ML metadata connection.
      pipeline_info: The information of the pipeline that this driver is in.
      pipeline_node: The specification of the node that this driver is in.

    Raises:
      RuntimeError: if the driver_spec is not supported.
    """
    super(PythonDriverOperator, self).__init__(driver_spec, mlmd_connection,
                                               pipeline_info, pipeline_node)

    python_class_driver_spec = cast(
        pipeline_pb2.ExecutorSpec.PythonClassExecutorSpec, driver_spec)
    self._driver = import_utils.import_class_by_path(
        python_class_driver_spec.class_path)(self._mlmd_connection,
                                             self._pipeline_info,
                                             self._pipeline_node)

  def run_driver(
      self, input_dict: Dict[Text, List[types.Artifact]],
      output_dict: Dict[Text, List[types.Artifact]],
      exec_properties: Dict[Text, Any]) -> driver_output_pb2.DriverOutput:
    """Invokes the driver with inputs provided by the Launcher.

    Args:
      input_dict: The defult input_dict resolved by the launcher.
      output_dict: The default output_dict resolved by the launcher.
      exec_properties: The default exec_properties resolved by the launcher.

    Returns:
      An DriverOutput instance.
    """
    return self._driver.run(input_dict, output_dict, exec_properties)
