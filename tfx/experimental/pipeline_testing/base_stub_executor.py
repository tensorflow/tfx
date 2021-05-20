# Lint as: python3
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
"""Base Stub Executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Any, Dict, List, Optional, Text

from absl import logging
from tfx import types
from tfx.dsl.components.base import base_beam_executor
from tfx.dsl.components.base import base_executor
from tfx.dsl.io import fileio
from tfx.utils import io_utils

TEST_DATA_DIR_FLAG = "--test_data_dir"
COMPONENT_ID_FLAG = "--component_id"


class BaseStubExecutor(base_executor.BaseExecutor):
  """TFX base stub executor."""

  def __init__(self,
               component_id: Optional[Text] = None,
               test_data_dir: Optional[Text] = None,
               context: Optional[base_executor.BaseExecutor.Context] = None):
    """Initializes a BaseStubExecutor.

    Args:
      component_id: component id of a component associated with the stub
        executor.
      test_data_dir: The directory to test data (pipeline_recorder.py).
      context: context class for all executors.
        component_id and test_data_dir can be encoded in the context as well.

    Raises:
      ValueError: If the recorded pipeline data doesn't exist at test_data_dir.
    """
    # Fill parameters from beam_pipeline_args if empty.
    # TODO(b/156000550): Migrate beam_pipeline_args to extra_flags
    if context:
      extra_flags = []
      if isinstance(context, base_beam_executor.BaseBeamExecutor.Context):
        extra_flags = context.extra_flags
      elif context.beam_pipeline_args:
        extra_flags = context.beam_pipeline_args
      for extra_flag in extra_flags:
        if extra_flag.startswith(TEST_DATA_DIR_FLAG) and test_data_dir is None:
          test_data_dir = extra_flag[len(TEST_DATA_DIR_FLAG)+1:]  # skip '='.
        elif extra_flag.startswith(COMPONENT_ID_FLAG) and component_id is None:
          component_id = extra_flag[len(COMPONENT_ID_FLAG)+1:]  # skip '='.

    super(BaseStubExecutor, self).__init__(context)
    logging.info("Running StubExecutor, component_id %s", component_id)
    self._component_id = component_id
    self._test_data_dir = test_data_dir
    if not fileio.exists(self._test_data_dir):
      raise ValueError("Must record pipeline in {}".format(self._test_data_dir))

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Copies over recorded data to pipeline output uri.

    Args:
      input_dict: Input dict from input key to a list of Artifacts.
      output_dict: Output dict from output key to a list of Artifacts.
      exec_properties: A dict of execution properties.

    Returns:
      None

    Raises:
      FileNotFoundError: If the recorded test data dir doesn't exist any more.
    """
    for output_key, artifact_list in output_dict.items():
      for idx, artifact in enumerate(artifact_list):
        dest = artifact.uri
        src = os.path.join(self._test_data_dir, self._component_id, output_key,
                           str(idx))
        if not fileio.exists(src):
          raise FileNotFoundError("{} does not exist".format(src))
        io_utils.copy_dir(src, dest)
        logging.info("Finished copying from %s to %s", src, dest)
