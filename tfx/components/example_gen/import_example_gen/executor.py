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
"""Generic TFX ImportExampleGen executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Any, Dict, Text

from absl import logging
import apache_beam as beam
import tensorflow as tf

from tfx.components.example_gen import utils
from tfx.components.example_gen.base_example_gen_executor import BaseExampleGenExecutor


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(tf.train.Example)
def _ImportExample(  # pylint: disable=invalid-name
    pipeline: beam.Pipeline, exec_properties: Dict[Text, Any],
    split_pattern: Text) -> beam.pvalue.PCollection:
  """Read TFRecord files to PCollection of TF examples.

  Note that each input split will be transformed by this function separately.

  Args:
    pipeline: beam pipeline.
    exec_properties: A dict of execution properties.
      - input: input dir that contains tf example data.
    split_pattern: Split.pattern in Input config, glob relative file pattern
      that maps to input files with root directory given by input_base.

  Returns:
    PCollection of TF examples.
  """
  input_base_uri = exec_properties[utils.INPUT_BASE_KEY]
  input_split_pattern = os.path.join(input_base_uri, split_pattern)
  logging.info('Reading input TFExample data %s.', input_split_pattern)

  # TODO(jyzhao): profile input examples.
  return (pipeline
          # TODO(jyzhao): support multiple input format.
          | 'ReadFromTFRecord' >>
          beam.io.ReadFromTFRecord(file_pattern=input_split_pattern)
          # TODO(jyzhao): consider move serialization out of base example gen.
          | 'ToTFExample' >> beam.Map(tf.train.Example.FromString))


class Executor(BaseExampleGenExecutor):
  """Generic TFX import example gen executor."""

  def GetInputSourceToExamplePTransform(self) -> beam.PTransform:
    """Returns PTransform for importing TF examples."""
    return _ImportExample
