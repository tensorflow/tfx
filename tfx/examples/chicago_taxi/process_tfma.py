# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runs a batch job for performing Tensorflow Model Analysis."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tempfile

import apache_beam as beam
import tensorflow as tf

import tensorflow_model_analysis as tfma
from tfx.examples.chicago_taxi.trainer import taxi


def process_tfma(eval_result_dir,
                 schema_file,
                 input_csv=None,
                 big_query_table=None,
                 eval_model_dir=None,
                 max_eval_rows=None,
                 pipeline_args=None):
  """Runs a batch job to evaluate the eval_model against the given input.

  Args:
    eval_result_dir: A directory where the evaluation result should be written
      to.
    schema_file: A file containing a text-serialized Schema that describes the
      eval data.
    input_csv: A path to a csv file which should be the input for evaluation.
      This can only be set if big_query_table is None.
    big_query_table: A BigQuery table name specified as DATASET.TABLE which
      should be the input for evaluation. This can only be set if input_csv is
      None.
    eval_model_dir: A directory where the eval model is located.
    max_eval_rows: Number of rows to query from BigQuery.
    pipeline_args: additional DataflowRunner or DirectRunner args passed to the
      beam pipeline.

  Raises:
    ValueError: if input_csv and big_query_table are not specified correctly.
  """

  if input_csv == big_query_table and input_csv is None:
    raise ValueError(
        'one of --input_csv or --big_query_table should be provided.')

  slice_spec = [
      tfma.slicer.SingleSliceSpec(),
      tfma.slicer.SingleSliceSpec(columns=['trip_start_hour'])
  ]

  schema = taxi.read_schema(schema_file)

  eval_shared_model = tfma.default_eval_shared_model(
      eval_saved_model_path=eval_model_dir,
      add_metrics_callbacks=[
          tfma.post_export_metrics.calibration_plot_and_prediction_histogram(),
          tfma.post_export_metrics.auc_plots()
      ])

  with beam.Pipeline(argv=pipeline_args) as pipeline:
    if input_csv:
      csv_coder = taxi.make_csv_coder(schema)
      raw_data = (
          pipeline
          | 'ReadFromText' >> beam.io.ReadFromText(
              input_csv, skip_header_lines=1)
          | 'ParseCSV' >> beam.Map(csv_coder.decode))
    else:
      assert big_query_table
      query = taxi.make_sql(big_query_table, max_eval_rows, for_eval=True)
      raw_feature_spec = taxi.get_raw_feature_spec(schema)
      raw_data = (
          pipeline
          | 'ReadBigQuery' >> beam.io.Read(
              beam.io.BigQuerySource(query=query, use_standard_sql=True))
          | 'CleanData' >>
          beam.Map(lambda x: (taxi.clean_raw_data_dict(x, raw_feature_spec))))

    # Examples must be in clean tf-example format.
    coder = taxi.make_proto_coder(schema)

    _ = (
        raw_data
        | 'ToSerializedTFExample' >> beam.Map(coder.encode)
        |
        'ExtractEvaluateAndWriteResults' >> tfma.ExtractEvaluateAndWriteResults(
            eval_shared_model=eval_shared_model,
            slice_spec=slice_spec,
            output_path=eval_result_dir))


def main():
  tf.logging.set_verbosity(tf.logging.INFO)

  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--eval_model_dir',
      help='Input path to the model which will be evaluated.')
  parser.add_argument(
      '--eval_result_dir',
      help='Output directory in which the model analysis result is written.')
  parser.add_argument(
      '--big_query_table',
      help='BigQuery path to input examples which will be evaluated.')
  parser.add_argument(
      '--input_csv',
      help='CSV file containing raw data which will be evaluated.')
  parser.add_argument(
      '--max_eval_rows',
      help='Maximum number of rows to evaluate on.',
      default=None,
      type=int)
  parser.add_argument(
      '--schema_file', help='File holding the schema for the input data')

  known_args, pipeline_args = parser.parse_known_args()

  if known_args.eval_result_dir:
    eval_result_dir = known_args.eval_result_dir
  else:
    eval_result_dir = tempfile.mkdtemp()

  process_tfma(
      eval_result_dir,
      input_csv=known_args.input_csv,
      big_query_table=known_args.big_query_table,
      eval_model_dir=known_args.eval_model_dir,
      max_eval_rows=known_args.max_eval_rows,
      schema_file=known_args.schema_file,
      pipeline_args=pipeline_args)


if __name__ == '__main__':
  main()
