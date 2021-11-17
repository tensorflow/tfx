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
"""Generic TFX BigQueryExampleGen executor."""

from typing import Any, Dict, Optional

import apache_beam as beam

from apache_beam.options import value_provider
from google.cloud import bigquery
import tensorflow as tf

from tfx.components.example_gen import base_example_gen_executor
from tfx.extensions.google_cloud_big_query import utils


class _BigQueryConverter:
  """Help class for bigquery result row to tf example conversion."""

  def __init__(self, query: str, project_id: Optional[str] = None):
    """Instantiate a _BigQueryConverter object.

    Args:
      query: the query statement to get the type information.
      project_id: optional. The GCP project ID to run the query job. Default to
        the GCP project ID set by the gcloud environment on the machine.
    """
    client = bigquery.Client(project=project_id)
    # Dummy query to get the type information for each field.
    query_job = client.query('SELECT * FROM ({}) LIMIT 0'.format(query))
    results = query_job.result()
    self._type_map = {}
    for field in results.schema:
      self._type_map[field.name] = field.field_type

  def RowToExample(self, instance: Dict[str, Any]) -> tf.train.Example:
    """Convert bigquery result row to tf example."""
    return utils.row_to_example(self._type_map, instance)


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(tf.train.Example)
def _BigQueryToExample(pipeline: beam.Pipeline, exec_properties: Dict[str, Any],
                       split_pattern: str) -> beam.pvalue.PCollection:
  """Read from BigQuery and transform to TF examples.

  Args:
    pipeline: beam pipeline.
    exec_properties: A dict of execution properties.
    split_pattern: Split.pattern in Input config, a BigQuery sql string.

  Returns:
    PCollection of TF examples.
  """

  beam_pipeline_args = exec_properties['_beam_pipeline_args']
  pipeline_options = beam.options.pipeline_options.PipelineOptions(
      beam_pipeline_args)
  # Try to parse the GCP project ID from the beam pipeline options.
  project = pipeline_options.view_as(
      beam.options.pipeline_options.GoogleCloudOptions).project
  if isinstance(project, value_provider.ValueProvider):
    project = project.get()
  converter = _BigQueryConverter(split_pattern, project)

  return (pipeline
          | 'QueryTable' >> utils.ReadFromBigQuery(query=split_pattern)
          | 'ToTFExample' >> beam.Map(converter.RowToExample))


class Executor(base_example_gen_executor.BaseExampleGenExecutor):
  """Generic TFX BigQueryExampleGen executor."""

  def GetInputSourceToExamplePTransform(self) -> beam.PTransform:
    """Returns PTransform for BigQuery to TF examples."""
    return _BigQueryToExample
