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
"""Utility functions shared across the different benchmarks."""

import importlib
from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2


def read_schema(proto_path):
  """Reads a TF Metadata schema from the given text proto file."""
  result = schema_pb2.Schema()
  with open(proto_path) as fp:
    text_format.Parse(fp.read(), result)
  return result


def get_dataset(name, base_dir=None):
  """Imports the given dataset and returns an instance of it."""
  lib = importlib.import_module("..datasets.%s.dataset" % name, __name__)
  return lib.get_dataset(base_dir)


def batched_iterator(records, batch_size):
  """Groups elements in the given list into batches of the given size.

  Args:
    records: List of elements to batch.
    batch_size: Size of each batch.

  Yields:
    Lists with batch_size elements from records. Every list yielded except the
    last will contain exactly batch_size elements.
  """
  batch = []
  for i, x in enumerate(records):
    batch.append(x)
    if (i + 1) % batch_size == 0:
      yield batch
      batch = []
  if batch:
    yield batch
