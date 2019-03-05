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
"""Generic TFX example_validator executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import tensorflow_data_validation as tfdv
from typing import Any, Dict, List, Text
from tfx.components.base import base_executor
from tfx.utils import io_utils
from tfx.utils import types

# Default file name for anomalies output.
DEFAULT_FILE_NAME = 'anomalies.pbtxt'


class Executor(base_executor.BaseExecutor):
  """TensorFlow ExampleValidator component executor."""

  def Do(self, input_dict,
         output_dict,
         exec_properties):
    """TensorFlow ExampleValidator executor entrypoint.

    This validates the statistics on the 'eval' split against the schema.

    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
        - stats: A list of 'ExampleStatisticsPath' type which should contain
          split 'eval'. Stats on other splits are ignored.
        - schema: A list of 'SchemaPath' type which should contain a single
          schema artifact.
      output_dict: Output dict from key to a list of artifacts, including:
        - output: A list of 'ExampleValidationPath' artifact of size one. It
          will include a single pbtxt file which contains all anomalies found.
      exec_properties: A dict of execution properties. Not used yet.

    Returns:
      None
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    tf.logging.info('Validating schema against the computed statistics.')
    schema = io_utils.SchemaReader().read(
        io_utils.get_only_uri_in_dir(
            types.get_single_uri(input_dict['schema'])))
    stats = tfdv.load_statistics(
        io_utils.get_only_uri_in_dir(
            types.get_split_uri(input_dict['stats'], 'eval')))
    output_uri = types.get_single_uri(output_dict['output'])
    anomalies = tfdv.validate_statistics(stats, schema)
    io_utils.write_pbtxt_file(
        os.path.join(output_uri, DEFAULT_FILE_NAME), anomalies)
    tf.logging.info(
        'Validation complete. Anomalies written to {}.'.format(output_uri))
