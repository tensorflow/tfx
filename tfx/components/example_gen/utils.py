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
"""Utilities for ExampleGen components."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tfx.proto import example_gen_pb2


def get_default_output_config():
  """Default config contains 'train' and 'eval' splits with size 2:1."""
  return example_gen_pb2.Output(
      split_config=example_gen_pb2.SplitConfig(splits=[
          example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=2),
          example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1)
      ]))
