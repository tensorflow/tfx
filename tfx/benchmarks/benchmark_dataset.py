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
"""Base class for classes representing a dataset for the benchmark."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf


class BenchmarkDataset(object):
  """Base class for classes representing a dataset for the benchmark."""

  def __init__(self, base_dir=None):
    """Construct a dataset instance.

    Args:
      base_dir: The directory in which datasets artifacts are located. This will
        be used for reading during benchmark execution, as well as writing
        during benchmark regeneration. By default, the directory in which this
        file is located at runtime will be used to infer the location of
        `tfx/benchmarks/datasets`.
    """
    self._base_dir = (
        base_dir if base_dir else os.path.join(
            os.path.dirname(__file__), "datasets"))

  def datasets_dir(self, subdir=""):
    """Returns the path to the datasets directory.

    Args:
      subdir: Subdirectory to join at the end of the datasets directory.

    Returns:
      The path to the datasets directory, with the subdir joined at the end.
    """
    return os.path.join(self._base_dir, subdir)

  def dataset_path(self):
    """Returns the path to the dataset file."""
    raise NotImplementedError()

  def tf_metadata_schema_path(self):
    """Returns the path to the tf.Metadata schema file."""
    raise NotImplementedError()

  def trained_saved_model_path(self):
    """Returns the path to the inference format SavedModel."""
    raise NotImplementedError()

  def tft_saved_model_path(self):
    """Returns the path to the tf.Transform SavedModel."""
    raise NotImplementedError()

  def tfma_saved_model_path(self):
    """Returns the path to the tf.ModelAnalysis SavedModel."""
    raise NotImplementedError()

  def num_examples(self, limit=None):
    """Returns the number of examples in the dataset.

    Args:
      limit: If set, returns min(limit, number of examples in dataset).

    Returns:
      The number of examples in the dataset.
    """
    raise NotImplementedError()

  def read_raw_dataset(self, deserialize=True, limit=None):
    """Read the raw dataset of tf.train.Examples.

    Args:
      deserialize: If False, return the raw serialized bytes. If True, return
        the tf.train.Example parsed from the serialized bytes.
      limit: If set, yields no more than the given number of examples (might be
        less if the dataset has less examples than the limit).

    Yields:
      Serialized/unserialized (depending on deserialize) tf.train.Examples.
    """
    for count, example_bytes in enumerate(
        tf.compat.v1.io.tf_record_iterator(
            self.dataset_path(),
            tf.compat.v1.io.TFRecordOptions(
                tf.compat.v1.io.TFRecordCompressionType.GZIP))):
      if limit and count >= limit:
        break
      if not deserialize:
        yield example_bytes
      else:
        yield tf.train.Example().FromString(example_bytes)

  def generate_raw_dataset(self, args):
    """Generate the raw dataset.

    Args:
      args: String of extra arguments to use when generating the raw dataset.
    """
    raise NotImplementedError()

  def generate_models(self, args):
    """Generate the inference and tf.ModelAnalysis format SavedModels.

    This is usually done by running a Trainer on the raw dataset and exporting
    the inference and tf.ModelAnalysis format SavedModels.

    Args:
      args: String of extra arguments to use when generating the models.
    """
    raise NotImplementedError()
