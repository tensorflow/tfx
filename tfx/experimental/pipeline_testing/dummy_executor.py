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
"""Base Dummy Executor"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, List, Text
import absl
import os
from distutils.dir_util import copy_tree

from tfx import types
from tfx.components.base import base_executor

class BaseDummyExecutor(base_executor.BaseExecutor):
  """TFX base dummy executor."""
  def __init__(self, component_id, record_dir, context):
    super(BaseDummyExecutor, self).__init__(context)
    absl.logging.info("Running DummyExecutor, component_id %s", component_id)
    self._component_id = component_id
    self._record_dir = record_dir
    if not os.path.exists(self._record_dir):
      raise Exception("Must record input/output in {}".format(self._record_dir))

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    for _, artifact_list in output_dict.items():
      for artifact in artifact_list:
        dest = artifact.uri
        pipeline_path = os.path.join(os.environ['HOME'],
                                     "tfx/pipelines/",
                                     "chicago_taxi_beam/")
        src = dest.replace(pipeline_path, "")
        src = src[:src.rfind('/')] # remove trailing number
        src = os.path.join(self._record_dir, src)
        copy_tree(src, dest)
        absl.logging.info('from %s, copied to %s', src, dest)
