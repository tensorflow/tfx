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
"""Base Dummy Executor"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import absl
from distutils import dir_util
import os
from typing import Any, Dict, List, Text, Optional

from tfx import types
from tfx.components.base import base_executor

class BaseDummyExecutor(base_executor.BaseExecutor):
  """TFX base dummy executor."""
  def __init__(self,
               component_id: Text,
               test_data_dir: Text,
               context: Optional[base_executor.BaseExecutor.Context] = None):
    """Initializes a BaseDummyExecutor
    Args:
      component_id: component id of a component associated
        with the dummy executor
      test_data_dir: The directory to test data
        (pipeline_recorder.py)
    """
    super(BaseDummyExecutor, self).__init__(context)
    absl.logging.info("Running DummyExecutor, component_id %s", component_id)
    self._component_id = component_id
    self._test_data_dir = test_data_dir
    if not os.path.exists(self._test_data_dir):
      raise ValueError("Must record pipeline in {}".format(self._test_data_dir))

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Copies over recorded data to pipeline output uri"""
    for output_key, artifact_list in output_dict.items():
      for artifact in artifact_list:
        dest = artifact.uri
        component_id = artifact.producer_component
        src = os.path.join(self._test_data_dir, component_id, output_key)
        dir_util.copy_tree(src, dest)
        absl.logging.info('Finished copying from %s to %s', src, dest)
