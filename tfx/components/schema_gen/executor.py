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
"""Generic TFX schema_gen executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Any, Dict, List, Text

import absl
import tensorflow_data_validation as tfdv

from tfx import types
from tfx.components.base import base_executor
from tfx.types import artifact_utils
from tfx.utils import io_utils

# Key for statistics in executor input_dict.
STATISTICS_KEY = 'statistics'

# Key for output schema in executor output_dict.
SCHEMA_KEY = 'schema'

# Default file name for generated schema file.
_DEFAULT_FILE_NAME = 'schema.pbtxt'


class Executor(base_executor.BaseExecutor):
  """Generic TFX schema_gen executor."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """TensorFlow SchemaGen executor entrypoint.

    This infers the schema using tensorflow_data_validation on the precomputed
    stats of 'train' split.

    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
        - 'stats': A list of 'ExampleStatistics' type which must contain
          split 'train'. Stats on other splits are ignored.
        - 'statistics': Synonym for 'stats'.
      output_dict: Output dict from key to a list of artifacts, including:
        - output: A list of 'Schema' artifact of size one.
      exec_properties: A dict of execution properties, includes:
        - infer_feature_shape: Whether or not to infer the shape of the feature.

    Returns:
      None
    """
    # TODO(zhitaoli): Move constants between this file and component.py to a
    # constants.py.
    train_stats_uri = io_utils.get_only_uri_in_dir(
        artifact_utils.get_split_uri(input_dict[STATISTICS_KEY], 'train'))
    output_uri = os.path.join(
        artifact_utils.get_single_uri(output_dict[SCHEMA_KEY]),
        _DEFAULT_FILE_NAME)

    infer_feature_shape = exec_properties['infer_feature_shape']
    absl.logging.info('Infering schema from statistics.')
    schema = tfdv.infer_schema(
        tfdv.load_statistics(train_stats_uri), infer_feature_shape)
    io_utils.write_pbtxt_file(output_uri, schema)
    absl.logging.info('Schema written to %s.' % output_uri)
