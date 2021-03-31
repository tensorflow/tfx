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

from absl import logging
import tensorflow_data_validation as tfdv
from tfx import types
from tfx.dsl.components.base import base_executor
from tfx.types import artifact_utils
from tfx.types import standard_component_specs
from tfx.utils import io_utils
from tfx.utils import json_utils


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
        - 'statistics': A list of 'ExampleStatistics' type which must contain
          split 'train'.
      output_dict: Output dict from key to a list of artifacts, including:
        - schema: A list of 'Schema' artifact of size one.
      exec_properties: A dict of execution properties, includes:
        - infer_feature_shape: Whether or not to infer the shape of the feature.
        - exclude_splits: Names of splits that will not be taken into
          consideration when auto-generating a schema.

    Returns:
      None
    """
    # TODO(zhitaoli): Move constants between this file and component.py to a
    # constants.py.
    infer_feature_shape = bool(
        exec_properties.get(standard_component_specs.INFER_FEATURE_SHAPE_KEY,
                            True))

    # Load and deserialize exclude splits from execution properties.
    exclude_splits = json_utils.loads(
        exec_properties.get(standard_component_specs.EXCLUDE_SPLITS_KEY,
                            'null')) or []
    if not isinstance(exclude_splits, list):
      raise ValueError('exclude_splits in execution properties needs to be a '
                       'list. Got %s instead.' % type(exclude_splits))

    # Only one schema is generated for all splits.
    schema = None
    stats_artifact = artifact_utils.get_single_instance(
        input_dict[standard_component_specs.STATISTICS_KEY])
    for split in artifact_utils.decode_split_names(stats_artifact.split_names):
      if split in exclude_splits:
        continue

      logging.info('Processing schema from statistics for split %s.', split)
      stats_uri = io_utils.get_only_uri_in_dir(
          artifact_utils.get_split_uri([stats_artifact], split))
      if artifact_utils.is_artifact_version_older_than(
          stats_artifact, artifact_utils._ARTIFACT_VERSION_FOR_STATS_UPDATE):  # pylint: disable=protected-access
        stats = tfdv.load_statistics(stats_uri)
      else:
        stats = tfdv.load_stats_binary(stats_uri)
      if not schema:
        schema = tfdv.infer_schema(stats, infer_feature_shape)
      else:
        schema = tfdv.update_schema(schema, stats, infer_feature_shape)

    output_uri = os.path.join(
        artifact_utils.get_single_uri(
            output_dict[standard_component_specs.SCHEMA_KEY]),
        _DEFAULT_FILE_NAME)
    io_utils.write_pbtxt_file(output_uri, schema)
    logging.info('Schema written to %s.', output_uri)
