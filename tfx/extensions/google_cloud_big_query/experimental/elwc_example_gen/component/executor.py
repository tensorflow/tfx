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
"""Generic TFX BigQueryToElwcExampleGen executor."""

from typing import Any, Dict, Iterable, List, Set, Tuple

import apache_beam as beam
from google.cloud import bigquery
import tensorflow as tf
from tfx.components.example_gen import base_example_gen_executor
from tfx.extensions.google_cloud_big_query import utils
from tfx.extensions.google_cloud_big_query.experimental.elwc_example_gen.proto import elwc_config_pb2
from tfx.proto import example_gen_pb2

from google.protobuf import json_format
from tensorflow_serving.apis import input_pb2


# TODO(b/158514307): Revisit when PGBKCVOperation can hold serialized keys.
@beam.typehints.with_input_types(Dict[str, Any])
@beam.typehints.with_output_types(Tuple[bytes, tf.train.Example])
class _RowToContextFeatureAndExample(beam.DoFn):
  """Convert bigquery result to context feature and example feature pair."""

  def __init__(self, type_map: Dict[str, str],
               context_feature_fields: Set[str]):
    self._type_map = type_map
    self._context_feature_fields = context_feature_fields

  def process(
      self, instance: Dict[str, Any]
  ) -> Iterable[Tuple[bytes, tf.train.Example]]:
    context_feature = dict((k, instance[k])
                           for k in instance.keys()
                           if k in self._context_feature_fields)
    context_feature_proto = utils.row_to_example(self._type_map,
                                                 context_feature)
    context_feature_key = context_feature_proto.SerializeToString(
        deterministic=True)
    example_feature = dict((k, instance[k])
                           for k in instance.keys()
                           if k not in self._context_feature_fields)
    example_feature_value = utils.row_to_example(self._type_map,
                                                 example_feature)
    yield (context_feature_key, example_feature_value)


def _ConvertContextAndExamplesToElwc(
    context_feature_and_examples: Tuple[bytes, List[tf.train.Example]]
) -> input_pb2.ExampleListWithContext:
  """Convert context feature and examples to ELWC."""
  context_feature, examples = context_feature_and_examples
  context_feature_proto = tf.train.Example()
  context_feature_proto.ParseFromString(context_feature)
  return input_pb2.ExampleListWithContext(
      context=context_feature_proto, examples=examples)


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(input_pb2.ExampleListWithContext)
def _BigQueryToElwc(pipeline: beam.Pipeline, exec_properties: Dict[str, Any],
                    split_pattern: str) -> beam.pvalue.PCollection:
  """Read from BigQuery and transform to ExampleListWithContext.

  When a field has no value in BigQuery, a feature with no value will be
  generated in the tf.train.Features. This behavior is consistent with
  BigQueryExampleGen.

  Args:
    pipeline: beam pipeline.
    exec_properties: A dict of execution properties.
    split_pattern: Split.pattern in Input config, a BigQuery sql string.

  Returns:
    PCollection of ExampleListWithContext.

  Raises:
    RuntimeError: Context features must be included in the queried result.
  """

  custom_config = example_gen_pb2.CustomConfig()
  json_format.Parse(exec_properties['custom_config'], custom_config)
  elwc_config = elwc_config_pb2.ElwcConfig()
  custom_config.custom_config.Unpack(elwc_config)

  client = bigquery.Client()
  # Dummy query to get the type information for each field.
  query_job = client.query('SELECT * FROM ({}) LIMIT 0'.format(split_pattern))
  results = query_job.result()
  type_map = {}
  context_feature_fields = set(elwc_config.context_feature_fields)
  field_names = set()
  for field in results.schema:
    type_map[field.name] = field.field_type
    field_names.add(field.name)
  # Check whether the query contains necessary context fields.
  if not field_names.issuperset(context_feature_fields):
    raise RuntimeError('Context feature fields are missing from the query.')

  return (
      pipeline
      | 'ReadFromBigQuery' >> utils.ReadFromBigQuery(query=split_pattern)
      | 'RowToContextFeatureAndExample' >> beam.ParDo(
          _RowToContextFeatureAndExample(type_map, context_feature_fields))
      |
      'CombineByContext' >> beam.CombinePerKey(beam.combiners.ToListCombineFn())
      | 'ConvertContextAndExamplesToElwc' >>
      beam.Map(_ConvertContextAndExamplesToElwc))


class Executor(base_example_gen_executor.BaseExampleGenExecutor):
  """Generic TFX BigQueryElwcExampleGen executor."""

  def GetInputSourceToExamplePTransform(self) -> beam.PTransform:
    """Returns PTransform for BigQuery to ExampleListWithContext."""
    return _BigQueryToElwc
