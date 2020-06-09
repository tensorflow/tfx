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
"""TFT benchmark base."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import shutil
import tempfile
import time

# Standard Imports

from absl import logging
import apache_beam as beam
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform import graph_tools
from tensorflow_transform import impl_helper
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.beam import impl as tft_beam_impl
from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils
from tfx_bsl.beam import shared

import tfx
from tfx.benchmarks import benchmark_utils
from tfx.benchmarks import benchmark_base


class _CopySavedModel(beam.PTransform):
  """Copies the TFT SavedModel to another directory."""

  def __init__(self, dest_path):
    self._dest_path = dest_path

  def expand(self, transform_fn):

    def copy_saved_model(unused_element, source_path, dest_path):
      shutil.rmtree(dest_path, ignore_errors=True)
      shutil.copytree(source_path, dest_path)
      logging.info("Copied SavedModel from %s to %s", source_path, dest_path)

    return (transform_fn.pipeline
            | "CreateSole" >> beam.Create([None])
            | "CopySavedModel" >> beam.Map(
                copy_saved_model,
                source_path=beam.pvalue.AsSingleton(transform_fn),
                dest_path=self._dest_path))


class _AnalyzeAndTransformDataset(beam.PTransform):
  """PTransform to run AnalyzeAndTransformDataset."""

  def __init__(self,
               dataset,
               tf_metadata_schema,
               preprocessing_fn,
               transform_input_dataset_metadata,
               generate_dataset=False):
    """Constructor.

    Args:
      dataset: BenchmarkDataset object.
      tf_metadata_schema: tf.Metadata schema.
      preprocessing_fn: preprocessing_fn.
      transform_input_dataset_metadata: dataset_metadata.DatasetMetadata.
      generate_dataset: If True, generates the raw dataset and appropriate
        intermediate outputs (just the TFT SavedModel for now) necessary for
        other benchmarks.
    """
    self._dataset = dataset
    self._tf_metadata_schema = tf_metadata_schema
    self._preprocessing_fn = preprocessing_fn
    self._transform_input_dataset_metadata = transform_input_dataset_metadata
    self._generate_dataset = generate_dataset

  def expand(self, pipeline):
    # TODO(b/147620802): Consider making this (and other parameters)
    # configurable to test more variants (e.g. with and without deep-copy
    # optimisation, with and without cache, etc).
    with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
      converter = tft.coders.ExampleProtoCoder(
          self._tf_metadata_schema, serialized=False)
      raw_data = (
          pipeline
          | "ReadDataset" >> beam.Create(self._dataset.read_raw_dataset())
          | "Decode" >> beam.Map(converter.decode))
      transform_fn, output_metadata = (
          (raw_data, self._transform_input_dataset_metadata)
          | "AnalyzeDataset" >> tft_beam.AnalyzeDataset(self._preprocessing_fn))

      if self._generate_dataset:
        _ = transform_fn | "CopySavedModel" >> _CopySavedModel(
            dest_path=self._dataset.tft_saved_model_path())

      (transformed_dataset, transformed_metadata) = (
          ((raw_data, self._transform_input_dataset_metadata),
           (transform_fn, output_metadata))
          | "TransformDataset" >> tft_beam.TransformDataset())
      return transformed_dataset, transformed_metadata


# Tuple for variables common to all benchmarks.
CommonVariablesTuple = collections.namedtuple("CommonVariablesTuple", [
    "tf_metadata_schema", "preprocessing_fn", "transform_input_dataset_metadata"
])


def _get_common_variables(dataset):
  """Returns metadata schema, preprocessing fn, input dataset metadata."""

  tf_metadata_schema = benchmark_utils.read_schema(
      dataset.tf_metadata_schema_path())

  preprocessing_fn = dataset.tft_preprocessing_fn()

  feature_spec = schema_utils.schema_as_feature_spec(
      tf_metadata_schema).feature_spec
  transform_input_columns = (
      tft.get_transform_input_columns(preprocessing_fn, feature_spec))
  transform_input_dataset_metadata = dataset_metadata.DatasetMetadata(
      schema_utils.schema_from_feature_spec({
          feature: feature_spec[feature] for feature in transform_input_columns
      }))

  return CommonVariablesTuple(
      tf_metadata_schema=tf_metadata_schema,
      preprocessing_fn=preprocessing_fn,
      transform_input_dataset_metadata=transform_input_dataset_metadata)


def regenerate_intermediates_for_dataset(dataset):
  """Regenerate intermediate outputs required for the benchmark."""

  common_variables = _get_common_variables(dataset)

  logging.info("Regenerating intermediate outputs required for benchmark.")
  with beam.Pipeline() as p:
    _ = p | _AnalyzeAndTransformDataset(
        dataset,
        common_variables.tf_metadata_schema,
        common_variables.preprocessing_fn,
        common_variables.transform_input_dataset_metadata,
        generate_dataset=True)
  logging.info("Intermediate outputs regenerated.")


def _get_batched_records(dataset):
  """Returns a (batch_size, iterator for batched records) tuple for the dataset.

  Args:
    dataset: BenchmarkDataset object.

  Returns:
    Tuple of (batch_size, iterator for batched records), where records are
    decoded tf.train.Examples.
  """
  batch_size = 1000
  common_variables = _get_common_variables(dataset)
  converter = tft.coders.ExampleProtoCoder(
      common_variables.tf_metadata_schema, serialized=False)
  records = [converter.decode(x) for x in dataset.read_raw_dataset()]
  return batch_size, benchmark_utils.batched_iterator(records, batch_size)


class TFTBenchmarkBase(benchmark_base.BenchmarkBase):
  """TFT benchmark base class."""

  def __init__(self, dataset, **kwargs):
    # Benchmark runners may pass extraneous arguments we don't care about.
    del kwargs
    super(TFTBenchmarkBase, self).__init__()
    self._dataset = dataset

  def report_benchmark(self, **kwargs):
    if "extras" not in kwargs:
      kwargs["extras"] = {}
    # Note that the GIT_COMMIT_ID is not included in the packages themselves:
    # it must be injected by an external script.
    kwargs["extras"]["commit_tfx"] = getattr(tfx, "GIT_COMMIT_ID",
                                             tfx.__version__)
    kwargs["extras"]["commit_tft"] = getattr(tft, "GIT_COMMIT_ID",
                                             tft.__version__)
    super(TFTBenchmarkBase, self).report_benchmark(**kwargs)

  def benchmarkAnalyzeAndTransformDataset(self):
    """Benchmark AnalyzeAndTransformDataset.

    Runs AnalyzeAndTransformDataset in a Beam pipeline. Records the wall time
    taken for the whole pipeline.
    """
    common_variables = _get_common_variables(self._dataset)

    pipeline = self._create_beam_pipeline()
    _ = pipeline | _AnalyzeAndTransformDataset(
        self._dataset, common_variables.tf_metadata_schema,
        common_variables.preprocessing_fn,
        common_variables.transform_input_dataset_metadata)
    start = time.time()
    result = pipeline.run()
    result.wait_until_finish()
    end = time.time()
    delta = end - start

    self.report_benchmark(
        iters=1,
        wall_time=delta,
        extras={"num_examples": self._dataset.num_examples()})

  def benchmarkRunMetaGraphDoFnManualActuation(self):
    """Benchmark RunMetaGraphDoFn "manually".

    Runs RunMetaGraphDoFn "manually" outside of a Beam pipeline. Records the
    wall time taken.
    """
    common_variables = _get_common_variables(self._dataset)
    batch_size, batched_records = _get_batched_records(self._dataset)

    fn = tft_beam_impl._RunMetaGraphDoFn(  # pylint: disable=protected-access
        input_schema=common_variables.transform_input_dataset_metadata.schema,
        tf_config=None,
        shared_graph_state_handle=shared.Shared(),
        passthrough_keys=set(),
        exclude_outputs=None,
        use_tfxio=False)

    start = time.time()
    for batch in batched_records:
      _ = list(
          fn.process(
              batch, saved_model_dir=self._dataset.tft_saved_model_path()))
    end = time.time()
    delta = end - start
    self.report_benchmark(
        iters=1,
        wall_time=delta,
        extras={
            "batch_size": batch_size,
            "num_examples": self._dataset.num_examples()
        })

  def benchmarkRunMetagraphDoFnAtTFLevel(self):
    """Benchmark RunMetaGraphDoFn at the TF level.

    Benchmarks the parts of RunMetaGraphDoFn that involve feeding and
    fetching from the TFT SavedModel. Records the wall time taken.

    Note that this benchmark necessarily duplicates code directly from TFT
    since it's benchmarking the low-level internals of TFT, which are not
    exposed for use in this way.
    """
    common_variables = _get_common_variables(self._dataset)
    tf_config = tft_beam_impl._FIXED_PARALLELISM_TF_CONFIG  # pylint: disable=protected-access
    input_schema = common_variables.transform_input_dataset_metadata.schema

    # This block copied from _GraphState.__init__
    with tf.compat.v1.Graph().as_default() as graph:
      session = tf.compat.v1.Session(graph=graph, config=tf_config)
      with session.as_default():
        # TODO(b/148082271): Revert back to unpacking the result directly once
        # TFX depends on TFT 0.22.
        apply_saved_model_result = (
            saved_transform_io.partially_apply_saved_transform_internal(
                self._dataset.tft_saved_model_path(), {}))
        inputs, outputs = apply_saved_model_result[:2]
        session.run(tf.compat.v1.global_variables_initializer())
        session.run(tf.compat.v1.tables_initializer())
        graph.finalize()
      # We ignore the schema, and assume there are no excluded outputs.
      outputs_tensor_keys = sorted(set(outputs.keys()))
      fetches = [outputs[key] for key in outputs_tensor_keys]
      tensor_inputs = graph_tools.get_dependent_inputs(graph, inputs, fetches)
      input_tensor_keys = sorted(tensor_inputs.keys())
      feed_list = [inputs[key] for key in input_tensor_keys]
      callable_get_outputs = session.make_callable(fetches, feed_list=feed_list)

    batch_size, batched_records = _get_batched_records(self._dataset)

    # This block copied from _RunMetaGraphDoFn._handle_batch
    start = time.time()
    for batch in batched_records:
      feed_list = impl_helper.make_feed_list(input_tensor_keys, input_schema,
                                             batch)
      outputs_list = callable_get_outputs(*feed_list)
      _ = {key: value for key, value in zip(outputs_tensor_keys, outputs_list)}
    end = time.time()
    delta = end - start

    self.report_benchmark(
        iters=1,
        wall_time=delta,
        extras={
            "batch_size": batch_size,
            "num_examples": self._dataset.num_examples()
        })
