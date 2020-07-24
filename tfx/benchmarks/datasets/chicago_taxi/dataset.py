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
"""Chicago taxi dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile

from absl import logging
import apache_beam as beam

from tfx_bsl.coders import csv_decoder

from tfx import components
from tfx.benchmarks import benchmark_dataset
from tfx.components.example_gen.csv_example_gen import executor as csv_exgen
from tfx.examples.chicago_taxi_pipeline import taxi_utils

from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.proto import trainer_pb2
from tfx.utils.dsl_utils import external_input


class ChicagoTaxiDataset(benchmark_dataset.BenchmarkDataset):
  """Chicago taxi dataset."""

  def dataset_path(self):
    return self.datasets_dir("chicago_taxi/data/taxi_1M.tfrecords.gz")

  def tf_metadata_schema_path(self):
    return self.datasets_dir(
        "../../examples/chicago_taxi_pipeline/data/user_provided_schema/"
        "schema.pbtxt")

  def trained_saved_model_path(self):
    return self.datasets_dir("chicago_taxi/model/trained_saved_model")

  def tft_saved_model_path(self):
    return self.datasets_dir("chicago_taxi/model/tft_saved_model")

  def tfma_saved_model_path(self):
    return self.datasets_dir("chicago_taxi/model/tfma_saved_model")

  def tft_preprocessing_fn(self):
    return taxi_utils.preprocessing_fn

  def num_examples(self, limit=None):
    result = 1000000
    if limit:
      result = min(result, limit)
    return result

  def convert_csv_to_tf_examples(self, csv_path, tfrecords_output_path):
    """Runs a Beam pipeline to convert the CSV file into a TFRecords file.

    This is needed because the conversion is orders of magnitude more
    time-consuming than the functions we want to benchmark, so instead of
    doing the conversion each time, we do it once to generate a converted
    dataset and use that for the benchmark instead.

    Args:
      csv_path: Path to CSV file containing examples.
      tfrecords_output_path: Path to output TFRecords file containing parsed
        examples.
    """
    # Copied from CSV example gen.
    fp = open(csv_path, "r")
    column_names = next(fp).strip().split(",")
    fp.close()

    with beam.Pipeline() as p:
      parsed_csv_lines = (
          p
          | "ReadFromText" >> beam.io.ReadFromText(
              file_pattern=csv_path, skip_header_lines=1)
          |
          "ParseCSVLine" >> beam.ParDo(csv_decoder.ParseCSVLine(delimiter=",")))
      # TODO(b/155997704) clean this up once tfx_bsl makes a release.
      if getattr(csv_decoder, "PARSE_CSV_LINE_YIELDS_RAW_RECORDS", False):
        # parsed_csv_lines is the following tuple (parsed_lines, raw_records)
        # we only want the parsed_lines.
        parsed_csv_lines |= "ExtractParsedCSVLines" >> beam.Keys()

      column_infos = beam.pvalue.AsSingleton(
          parsed_csv_lines
          | "InferColumnTypes" >> beam.CombineGlobally(
              csv_decoder.ColumnTypeInferrer(
                  column_names, skip_blank_lines=True)))
      _ = (
          parsed_csv_lines
          | "ToTFExample" >> beam.ParDo(
              csv_exgen._ParsedCsvToTfExample(),  # pylint: disable=protected-access
              column_infos)
          | "Serialize" >> beam.Map(lambda x: x.SerializeToString())
          | "WriteToTFRecord" >> beam.io.tfrecordio.WriteToTFRecord(
              file_path_prefix=tfrecords_output_path,
              shard_name_template="",
              compression_type=beam.io.filesystem.CompressionTypes.GZIP))

  def generate_raw_dataset(self, args):
    logging.warn(
        "Not actually regenerating the raw dataset.\n"
        "To regenerate the raw CSV dataset, see the TFX Chicago Taxi example "
        "for details as to how to do so. "
        "tfx/examples/chicago_taxi_pipeline/taxi_pipeline_kubeflow_gcp.py "
        "has the BigQuery query used to generate the dataset.\n"
        "After regenerating the raw CSV dataset, you should also regenerate "
        "the derived TFRecords dataset. You can do so by passing "
        "--generate_dataset_args=/path/to/csv_dataset.csv to "
        "regenerate_datasets.py.")

    if args:
      logging.info("Converting CSV at %s to TFRecords", args)
      self.convert_csv_to_tf_examples(args, self.dataset_path())
      logging.info("TFRecords written to %s", self.dataset_path())

  def generate_models(self, args):
    # Modified version of Chicago Taxi Example pipeline
    # tfx/examples/chicago_taxi_pipeline/taxi_pipeline_beam.py

    root = tempfile.mkdtemp()
    pipeline_root = os.path.join(root, "pipeline")
    metadata_path = os.path.join(root, "metadata/metadata.db")
    module_file = os.path.join(
        os.path.dirname(__file__),
        "../../../examples/chicago_taxi_pipeline/taxi_utils.py")

    examples = external_input(os.path.dirname(self.dataset_path()))
    example_gen = components.ImportExampleGen(input=examples)
    statistics_gen = components.StatisticsGen(
        examples=example_gen.outputs["examples"])
    schema_gen = components.SchemaGen(
        statistics=statistics_gen.outputs["statistics"],
        infer_feature_shape=False)
    transform = components.Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=module_file)
    trainer = components.Trainer(
        module_file=module_file,
        transformed_examples=transform.outputs["transformed_examples"],
        schema=schema_gen.outputs["schema"],
        transform_graph=transform.outputs["transform_graph"],
        train_args=trainer_pb2.TrainArgs(num_steps=100),
        eval_args=trainer_pb2.EvalArgs(num_steps=50))
    p = pipeline.Pipeline(
        pipeline_name="chicago_taxi_beam",
        pipeline_root=pipeline_root,
        components=[
            example_gen, statistics_gen, schema_gen, transform, trainer
        ],
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path))
    BeamDagRunner().run(p)

    def join_unique_subdir(path):
      dirs = os.listdir(path)
      if len(dirs) != 1:
        raise ValueError(
            "expecting there to be only one subdirectory in %s, but "
            "subdirectories were: %s" % (path, dirs))
      return os.path.join(path, dirs[0])

    trainer_output_dir = join_unique_subdir(
        os.path.join(pipeline_root, "Trainer/model"))
    eval_model_dir = join_unique_subdir(
        os.path.join(trainer_output_dir, "eval_model_dir"))
    serving_model_dir = join_unique_subdir(
        os.path.join(trainer_output_dir,
                     "serving_model_dir/export/chicago-taxi"))

    shutil.rmtree(self.trained_saved_model_path(), ignore_errors=True)
    shutil.rmtree(self.tfma_saved_model_path(), ignore_errors=True)
    shutil.copytree(serving_model_dir, self.trained_saved_model_path())
    shutil.copytree(eval_model_dir, self.tfma_saved_model_path())


def get_dataset(base_dir=None):
  return ChicagoTaxiDataset(base_dir)
