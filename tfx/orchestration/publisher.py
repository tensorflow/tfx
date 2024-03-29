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
"""TFX publisher."""

from typing import Any, Dict, List, Optional

from absl import logging

from tfx import types
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration.portable import outputs_utils


class Publisher:
  """Publish execution to metadata.

  Attributes:
    _metadata_handle: An instance of Metadata.
  """

  def __init__(self, metadata_handle: metadata.Metadata):
    self._metadata_handle = metadata_handle

  def publish_execution(
      self,
      component_info: data_types.ComponentInfo,
      output_artifacts: Optional[Dict[str, List[types.Artifact]]] = None,
      exec_properties: Optional[Dict[str, Any]] = None):
    """Publishes a component execution to metadata.

    This function will do two things:
    1. update the execution that was previously registered before execution to
       complete or skipped state, depending on whether cached results are used.
    2. for each input and output artifact, publish an event that associate the
       artifact to the execution, with type INPUT or OUTPUT respectively

    Args:
      component_info: the information of the component
      output_artifacts: optional key -> Artifacts to be published as outputs
        of the execution
      exec_properties: optional execution properties to be published for the
        execution

    Returns:
      A dict containing output artifacts.
    """
    outputs_utils.tag_output_artifacts_with_version(output_artifacts)

    logging.debug('Outputs: %s', output_artifacts)
    logging.debug('Execution properties: %s', exec_properties)

    self._metadata_handle.publish_execution(
        component_info=component_info,
        output_artifacts=output_artifacts,
        exec_properties=exec_properties,
    )
