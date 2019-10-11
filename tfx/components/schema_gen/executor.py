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
import absl
import tensorflow as tf
import tensorflow_data_validation as tfdv
from typing import Any, Dict, List, Text
from tensorflow_metadata.proto.v0 import schema_pb2
from tfx import types
from tfx.components.base import base_executor
from tfx.types import artifact_utils
from tfx.utils import io_utils

# Default file name for generated schema file.
_DEFAULT_FILE_NAME = 'schema.pbtxt'


class Executor(base_executor.BaseExecutor):
  """Generic TFX schema_gen executor."""

  def _provide_schema(self, input_dict, exec_properties) -> schema_pb2.Schema:
    """Generates schema from either schema or statistics."""
    # TODO(zhitaoli): Move constants between this file and component.py to a
    # constants.py.
    stats = input_dict.get('stats') or input_dict.get('statistics')
    schema = input_dict.get('schema')

    if bool(stats) == bool(schema):
      raise ValueError(
          'Exactly only one of schema or stats must be provided')

    if schema:
      schema_uri = artifact_utils.get_single_uri(schema)
      absl.logging.info('Schema is provided. Reading from %s.' % schema_uri)
      schema_reader = io_utils.SchemaReader()
      try:
        return schema_reader.read(
            os.path.join(schema_uri, _DEFAULT_FILE_NAME))

      except tf.errors.NotFoundError:
        raise ValueError(
            'Schema is provided, but failed to read from %s.' % schema_uri)

    train_stats_uri = io_utils.get_only_uri_in_dir(
        artifact_utils.get_split_uri(stats, 'train'))
    infer_feature_shape = exec_properties['infer_feature_shape']
    return tfdv.infer_schema(
        tfdv.load_statistics(train_stats_uri), infer_feature_shape)

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
        - 'schema': A singleton list of 'Schema' type. If provided, pass
          it through as the output as fixed schema. If not provided, infer
          schema from stats.
        If both or neither 'stats/statistics' nor 'schema' is provided,
        an error is raised.
      output_dict: Output dict from key to a list of artifacts, including:
        - output: A list of 'Schema' artifact of size one.
      exec_properties: A dict of execution properties, includes:
        - infer_feature_shape: Whether or not to infer the shape of the feature.

    Returns:
      None
    """
    output_uri = os.path.join(
        artifact_utils.get_single_uri(output_dict['output']),
        _DEFAULT_FILE_NAME)

    # Materializing schema as an output artifact from SchemaGen, in order to log
    # metadata of it in the same way regardless of inferred or fixed.
    io_utils.write_pbtxt_file(output_uri,
                              self._provide_schema(input_dict, exec_properties))
    absl.logging.info('Schema written to {}.'.format(output_uri))
