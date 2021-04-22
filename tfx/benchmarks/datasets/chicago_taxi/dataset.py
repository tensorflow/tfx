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

import itertools
import math
import os
import shutil
import tempfile
from typing import Optional, Text

from absl import logging
import apache_beam as beam
import tensorflow_transform as tft
from tfx import components
from tfx.benchmarks import benchmark_dataset
from tfx.components.example_gen.csv_example_gen import executor as csv_exgen
from tfx.examples.chicago_taxi_pipeline import taxi_utils
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.proto import trainer_pb2
from tfx_bsl.coders import csv_decoder


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

  def tft_saved_model_path(self, force_tf_compat_v1):
    if force_tf_compat_v1:
      return self.datasets_dir("chicago_taxi/model/tft_saved_model")
    else:
      return self.datasets_dir("chicago_taxi/model/tft_tf2_saved_model")

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
          "ParseCSVLine" >> beam.ParDo(csv_decoder.ParseCSVLine(delimiter=","))
          | "ExtractParsedCSVLines" >> beam.Keys())

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

  def generate_models(self, args, force_tf_compat_v1=True):
    # Modified version of Chicago Taxi Example pipeline
    # tfx/examples/chicago_taxi_pipeline/taxi_pipeline_beam.py

    root = tempfile.mkdtemp()
    pipeline_root = os.path.join(root, "pipeline")
    metadata_path = os.path.join(root, "metadata/metadata.db")
    module_file = os.path.join(
        os.path.dirname(__file__),
        "../../../examples/chicago_taxi_pipeline/taxi_utils.py")

    example_gen = components.ImportExampleGen(
        input_base=os.path.dirname(self.dataset_path()))
    statistics_gen = components.StatisticsGen(
        examples=example_gen.outputs["examples"])
    schema_gen = components.SchemaGen(
        statistics=statistics_gen.outputs["statistics"],
        infer_feature_shape=False)
    transform = components.Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=module_file,
        force_tf_compat_v1=force_tf_compat_v1)
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
    transform_output_dir = join_unique_subdir(
        os.path.join(pipeline_root, "Transform/transform_graph"))
    transform_model_dir = os.path.join(transform_output_dir, "transform_fn")
    tft_saved_model_path = self.tft_saved_model_path(force_tf_compat_v1)

    shutil.rmtree(self.trained_saved_model_path(), ignore_errors=True)
    shutil.rmtree(self.tfma_saved_model_path(), ignore_errors=True)
    shutil.rmtree(tft_saved_model_path, ignore_errors=True)
    shutil.copytree(serving_model_dir, self.trained_saved_model_path())
    shutil.copytree(eval_model_dir, self.tfma_saved_model_path())
    shutil.copytree(transform_model_dir, tft_saved_model_path)


class WideChicagoTaxiDataset(ChicagoTaxiDataset):
  """Chicago taxi dataset with a TFT preprocessing_fn containing specified number of analyzers.

  Note that the analyzers are called within the corresponding mappers. Half of
  the mappers will be `tft.compute_and_apply_vocabulary`. Another half is split
  between `tft.bucketize` and `tft.scale_to_z_score`.
  """
  # Percentage of mappers in the preprocessing function of the given type. The
  # remaining mappers will be `tft.scale_to_z_score`.
  _VOCABS_SHARE = 0.5
  _BUCKETIZE_SHARE = 0.25

  def __init__(self, base_dir: Optional[Text] = None, num_analyzers: int = 10):
    super(WideChicagoTaxiDataset, self).__init__(base_dir)
    self._num_vocabs = math.ceil(num_analyzers * self._VOCABS_SHARE)
    self._num_bucketize = math.ceil(num_analyzers * self._BUCKETIZE_SHARE)
    self._num_scale = num_analyzers - self._num_vocabs - self._num_bucketize

  def tft_preprocessing_fn(self):

    def wide_preprocessing_fn(inputs):
      """TFT preprocessing function.

      Args:
        inputs: Map from feature keys to raw not-yet-transformed features.

      Returns:
        Map from string feature key to transformed feature operations.
      """
      outputs = {}
      # pylint: disable=protected-access
      for idx, key in enumerate(
          itertools.islice(
              itertools.cycle(taxi_utils._BUCKET_FEATURE_KEYS),
              self._num_bucketize)):
        outputs["bucketized" + str(idx)] = tft.bucketize(
            taxi_utils._fill_in_missing(inputs[key]),
            taxi_utils._FEATURE_BUCKET_COUNT)

      for idx, key in enumerate(
          itertools.islice(
              itertools.cycle(taxi_utils._DENSE_FLOAT_FEATURE_KEYS),
              self._num_scale)):
        # Preserve this feature as a dense float, setting nan's to the mean.
        outputs["scaled" + str(idx)] = tft.scale_to_z_score(
            taxi_utils._fill_in_missing(inputs[key]))

      for idx, key in enumerate(
          itertools.islice(
              itertools.cycle(taxi_utils._VOCAB_FEATURE_KEYS),
              self._num_vocabs)):
        outputs["vocab" + str(idx)] = tft.compute_and_apply_vocabulary(
            taxi_utils._fill_in_missing(inputs[key]),
            top_k=taxi_utils._VOCAB_SIZE,
            num_oov_buckets=taxi_utils._OOV_SIZE)

      # Pass-through features.
      for key in taxi_utils._CATEGORICAL_FEATURE_KEYS + [taxi_utils._LABEL_KEY]:
        outputs[key] = inputs[key]

      return outputs

    return wide_preprocessing_fn


def get_dataset(base_dir=None):
  return ChicagoTaxiDataset(base_dir)


def get_wide_dataset(base_dir=None, num_analyzers=10):
  return WideChicagoTaxiDataset(base_dir, num_analyzers)
