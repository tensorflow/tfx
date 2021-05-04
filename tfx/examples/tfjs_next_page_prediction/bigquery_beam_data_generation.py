# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Beam pipeline for converting Google Analytics into training Examples."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, List, Union, Text

import apache_beam as beam
from apache_beam.io.gcp.internal.clients import bigquery
import tensorflow as tf


def _sanitize_page_path(page_path: Text):
  """Remove everything after the query."""
  return page_path.split('?')[0]


def create_tensorflow_example(features: Dict[Text, List[Union[int, float,
                                                              Text]]]):
  """Populate a Tensorflow Example with the given features."""
  result = tf.train.Example()
  for name, value in features.items():
    if not value:
      raise ValueError('each feature must have a populated value list.')
    if isinstance(value[0], int):
      result.features.feature[name].int64_list.value.extend(value)
    elif isinstance(value[0], float):
      result.features.feature[name].float_list.value.extend(value)
    else:
      result.features.feature[name].bytes_list.value.extend(
          [bytes(v, 'utf-8') for v in value])
  return result


def ga_session_to_tensorflow_examples(session: List[Any]):
  """Converts a Google Analytics Session to Tensorflow Examples."""
  examples = []
  for i in range(len(session) - 1):
    features = {
        # Add any additional desired training features here.
        'cur_page': [_sanitize_page_path(session[i]['page']['pagePath'])],
        'label': [_sanitize_page_path(session[i + 1]['page']['pagePath'])],
        'session_index': [i],
    }
    examples.append(create_tensorflow_example(features))
  return examples


def is_duplicate_event(first_event: Dict[Text, Any], second_event: Dict[Text,
                                                                        Any]):
  return (first_event['time'] == second_event['time'] or
          first_event['page']['pagePath'] == second_event['page']['pagePath'])


class ExampleGeneratingDoFn(beam.DoFn):
  """Creates Tensorflow Examples from the provided Google Analytics session."""

  def process(self, entry: Dict[Text, Any]):
    session = entry['hits']
    session.sort(key=lambda s: s['hitNumber'])
    filtered_session = []
    for s in session:
      if filtered_session and is_duplicate_event(filtered_session[-1], s):
        continue
      filtered_session.append(s)
    return ga_session_to_tensorflow_examples(filtered_session)


def run_beam_pipeline():
  """Run the apache beam pipeline with the specified flags."""

  # Params used for running the Beam pipeline. Update these based on your
  # requirements.
  params = {}
  # Specify the projectid for BigQuery
  params['projectId'] = 'my_project_id'
  # Specify the datasetid for BigQuery
  params['datasetId'] = 'my_dataset_id'
  # Specify the table for BigQuery
  params['tableId'] = 'my_table_id'
  # Specify the list of flags for the Beam pipeline
  params['flags'] = ['--temp_location=my_temp_location']
  # Specify the desination for the generated examples.
  params['destination'] = 'my_destination'

  table_spec = bigquery.TableReference(
      projectId=params['projectId'],
      datasetId=params['datasetId'],
      tableId=params['tableId'])
  with beam.Pipeline(
      options=beam.options.pipeline_options.PipelineOptions(
          flags=params['flags'])) as p:
    _ = (
        p
        | 'ReadTable' >> beam.io.ReadFromBigQuery(table=table_spec)
        | 'ConvertToTensorFlowExamples' >> beam.ParDo(ExampleGeneratingDoFn())
        | 'Write' >> beam.io.tfrecordio.WriteToTFRecord(
            'gs://tfxdata/data/output',
            coder=beam.coders.ProtoCoder(tf.train.Example),
            file_name_suffix='.tfrecord.gz'))


if __name__ == '__main__':
  # TODO(b/187088244): Demonstrate how this can be done with a custom ExampleGen
  # within TFX. Also update function / method naming.
  run_beam_pipeline()
