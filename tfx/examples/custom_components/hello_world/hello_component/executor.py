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
"""Example of a Hello World TFX custom component.

This custom component simply passes examples through. This is meant to serve as
a kind of starting point example for creating custom components.

This component along with other custom component related code will only serve as
an example and will not be supported by TFX team.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from typing import Any, Dict, List, Text

import tensorflow as tf
from tfx import types
from tfx.components.base import base_executor
from tfx.types import artifact_utils
from tfx.utils import io_utils


class Executor(base_executor.BaseExecutor):
  """Executor for HelloComponent."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Copy the input_data to the output_data.

    For this example that is all that the Executor does.  For a different
    custom component, this is where the real functionality of the component
    would be included.

    This component both reads and writes Examples, but a different component
    might read and write artifacts of other types.

    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
        - input_data: A list of type `standard_artifacts.Examples` which will
          often contain two splits, 'train' and 'eval'.
      output_dict: Output dict from key to a list of artifacts, including:
        - output_data: A list of type `standard_artifacts.Examples` which will
          usually contain the same splits as input_data.
      exec_properties: A dict of execution properties, including:
        - name: Optional unique name. Necessary iff multiple Hello components
          are declared in the same pipeline.

    Returns:
      None

    Raises:
      OSError and its subclasses
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    split_to_instance = {}
    for artifact in input_dict['input_data']:
      for split in json.loads(artifact.split_names):
        uri = os.path.join(artifact.uri, split)
        split_to_instance[split] = uri

    for split, instance in split_to_instance.items():
      input_dir = instance
      output_dir = artifact_utils.get_split_uri(
          output_dict['output_data'], split)
      for filename in tf.io.gfile.listdir(input_dir):
        input_uri = os.path.join(input_dir, filename)
        output_uri = os.path.join(output_dir, filename)
        io_utils.copy_file(src=input_uri, dst=output_uri, overwrite=True)
