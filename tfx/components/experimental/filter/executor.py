# Copyright 2026 Google LLC. All Rights Reserved.
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
"""TFX experimental Filter component executor."""

import os
from typing import Any, Dict, List

from absl import logging
import apache_beam as beam
import tensorflow as tf
from tfx import types
from tfx.dsl.components.base import base_beam_executor
from tfx.types import artifact_utils
from tfx.utils import import_utils


class Executor(base_beam_executor.BaseBeamExecutor):
  """TFX experimental Filter component executor."""

  def Do(self, input_dict: Dict[str, List[types.Artifact]],
         output_dict: Dict[str, List[types.Artifact]],
         exec_properties: Dict[str, Any]) -> None:
    """Runs the filter Apache Beam pipeline.

    Args:
      input_dict: Input dict from input key to a list of Artifacts.
        - examples: A list of type `standard_artifacts.Examples` containing
          the splits to be filtered.
      output_dict: Output dict from output key to a list of Artifacts.
        - filtered_examples: A list of type `standard_artifacts.Examples`
          where the filtered splits will be written.
      exec_properties: A dict of execution properties.
        - filter_fn_path: The Python import path to the filter function.
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    examples = artifact_utils.get_single_instance(input_dict['examples'])
    filtered_examples = artifact_utils.get_single_instance(
        output_dict['filtered_examples'])

    # Setup output splits.
    split_names = artifact_utils.decode_split_names(examples.split_names)
    filtered_examples.split_names = artifact_utils.encode_split_names(
        split_names)
    filtered_examples.span = examples.span
    filtered_examples.version = examples.version

    # Import the user-defined filter function.
    filter_fn_path = exec_properties['filter_fn_path']
    logging.info('Importing user filter function from: %s', filter_fn_path)
    filter_fn = import_utils.import_class_by_path(filter_fn_path)

    with self._make_beam_pipeline() as pipeline:
      for split in split_names:
        input_split_uri = artifact_utils.get_split_uri([examples], split)
        output_split_uri = artifact_utils.get_split_uri([filtered_examples],
                                                        split)

        # Ensure output split directory exists.
        tf.io.gfile.makedirs(output_split_uri)

        input_pattern = os.path.join(input_split_uri, '*')
        output_prefix = os.path.join(output_split_uri, 'data_tfrecord')

        logging.info('Filtering split %s. Reading from %s, writing to prefix %s',
                     split, input_pattern, output_prefix)

        # Run the Beam pipeline to read, filter, and write the split.
        _ = (
            pipeline
            | f'ReadFromTFRecord[{split}]' >> beam.io.ReadFromTFRecord(
                input_pattern)
            | f'FilterExamples[{split}]' >> beam.Filter(filter_fn)
            | f'WriteToTFRecord[{split}]' >> beam.io.WriteToTFRecord(
                output_prefix,
                file_name_suffix='.gz')
        )

    logging.info('FilterComponent execution completed successfully.')
