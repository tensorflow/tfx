# Lint as: python2, python3
# Copyright 2020 Google LLC. All Rights Reserved.
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
"""TFX BigQueryExampleGen component definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional

from tfx.components.example_gen import component
from tfx.components.example_gen import utils
from tfx.dsl.components.base import executor_spec
from tfx.extensions.google_cloud_big_query.example_gen import executor
from tfx.proto import example_gen_pb2


class BigQueryExampleGen(component.QueryBasedExampleGen):
  """Cloud BigQueryExampleGen component.

  The BigQuery examplegen component takes a query, and generates train
  and eval examples for downstream components.

  Component `outputs` contains:
   - `examples`: Channel of type `standard_artifacts.Examples` for output train
                 and eval examples.
  """

  EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(executor.Executor)

  def __init__(self,
               query: Optional[str] = None,
               input_config: Optional[example_gen_pb2.Input] = None,
               output_config: Optional[example_gen_pb2.Output] = None):
    """Constructs a BigQueryExampleGen component.

    Args:
      query: BigQuery sql string, query result will be treated as a single
        split, can be overwritten by input_config.
      input_config: An example_gen_pb2.Input instance with Split.pattern as
        BigQuery sql string. If set, it overwrites the 'query' arg, and allows
        different queries per split. If any field is provided as a
        RuntimeParameter, input_config should be constructed as a dict with the
        same field names as Input proto message.
      output_config: An example_gen_pb2.Output instance, providing output
        configuration. If unset, default splits will be 'train' and 'eval' with
        size 2:1. If any field is provided as a RuntimeParameter,
        input_config should be constructed as a dict with the same field names
        as Output proto message.

    Raises:
      RuntimeError: Only one of query and input_config should be set.
    """
    if bool(query) == bool(input_config):
      raise RuntimeError('Exactly one of query and input_config should be set.')
    input_config = input_config or utils.make_default_input_config(query)
    super(BigQueryExampleGen, self).__init__(
        input_config=input_config, output_config=output_config)
