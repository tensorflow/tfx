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

import collections
import shutil
import tempfile
import time


from absl import logging
import apache_beam as beam
from apache_beam.utils import shared
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform import graph_tools
from tensorflow_transform import impl_helper
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.beam import impl as tft_beam_impl
from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.saved import saved_transform_io_v2
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils
import tfx
from tfx.benchmarks import benchmark_utils
from tfx.benchmarks import benchmark_base
from tfx_bsl.coders import example_coder
from tfx_bsl.tfxio import tensor_adapter
from tfx_bsl.tfxio import tf_example_record


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
               tfxio,
               preprocessing_fn,
               transform_input_dataset_metadata,
               force_tf_compat_v1=True,
               max_num_examples=None,
               generate_dataset=False):
    """Constructor.

    Args:
      dataset: BenchmarkDataset object.
      tfxio: A `tfx_bsl.TFXIO` instance.
      preprocessing_fn: preprocessing_fn.
      transform_input_dataset_metadata: dataset_metadata.DatasetMetadata.
      force_tf_compat_v1: If False then Transform will use its native TF2
        version, if True then Transform will use its TF1 version.
      max_num_examples: Max number of examples to read from the dataset.
      generate_dataset: If True, generates the raw dataset and appropriate
        intermediate outputs (just the TFT SavedModel for now) necessary for
        other benchmarks.
    """
    self._dataset = dataset
    self._tfxio = tfxio
    self._preprocessing_fn = preprocessing_fn
    self._transform_input_dataset_metadata = transform_input_dataset_metadata
    self._force_tf_compat_v1 = force_tf_compat_v1
    self._max_num_examples = max_num_examples
    self._generate_dataset = generate_dataset

  def expand(self, pipeline):
    # TODO(b/147620802): Consider making this (and other parameters)
    # configurable to test more variants (e.g. with and without deep-copy
    # optimisation, with and without cache, etc).
    with tft_beam.Context(
        temp_dir=tempfile.mkdtemp(),
        force_tf_compat_v1=self._force_tf_compat_v1):
      raw_data = (
          pipeline
          | "ReadDataset" >> beam.Create(
              self._dataset.read_raw_dataset(
                  deserialize=False, limit=self._max_num_examples))
          | "Decode" >> self._tfxio.BeamSource())
      transform_fn, output_metadata = (
          (raw_data, self._tfxio.TensorAdapterConfig())
          | "AnalyzeDataset" >> tft_beam.AnalyzeDataset(self._preprocessing_fn))

      if self._generate_dataset:
        _ = transform_fn | "CopySavedModel" >> _CopySavedModel(
            dest_path=self._dataset.tft_saved_model_path(
                self._force_tf_compat_v1))

      (transformed_dataset, transformed_metadata) = (
          ((raw_data, self._tfxio.TensorAdapterConfig()),
           (transform_fn, output_metadata))
          | "TransformDataset" >>
          tft_beam.TransformDataset(output_record_batches=True))
      return transformed_dataset, transformed_metadata


# Tuple for variables common to all benchmarks.
CommonVariablesTuple = collections.namedtuple("CommonVariablesTuple", [
    "tf_metadata_schema",
    "preprocessing_fn",
    "transform_input_dataset_metadata",
    "tfxio",
])


def _get_common_variables(dataset, force_tf_compat_v1):
  """Returns metadata schema, preprocessing fn, input dataset metadata."""

  tf_metadata_schema = benchmark_utils.read_schema(
      dataset.tf_metadata_schema_path())

  preprocessing_fn = dataset.tft_preprocessing_fn()

  feature_spec = schema_utils.schema_as_feature_spec(
      tf_metadata_schema).feature_spec
  type_spec = impl_helper.get_type_specs_from_feature_specs(feature_spec)
  transform_input_columns = (
      tft.get_transform_input_columns(
          preprocessing_fn, type_spec, force_tf_compat_v1=force_tf_compat_v1))
  transform_input_dataset_metadata = dataset_metadata.DatasetMetadata(
      schema_utils.schema_from_feature_spec({
          feature: feature_spec[feature] for feature in transform_input_columns
      }))
  tfxio = tf_example_record.TFExampleBeamRecord(
      physical_format="tfexamples",
      schema=transform_input_dataset_metadata.schema,
      telemetry_descriptors=["TFTransformBenchmark"])

  return CommonVariablesTuple(
      tf_metadata_schema=tf_metadata_schema,
      preprocessing_fn=preprocessing_fn,
      transform_input_dataset_metadata=transform_input_dataset_metadata,
      tfxio=tfxio)


def regenerate_intermediates_for_dataset(dataset,
                                         force_tf_compat_v1=True,
                                         max_num_examples=None):
  """Regenerate intermediate outputs required for the benchmark."""

  common_variables = _get_common_variables(dataset, force_tf_compat_v1)

  logging.info("Regenerating intermediate outputs required for benchmark.")
  with beam.Pipeline() as p:
    _ = p | _AnalyzeAndTransformDataset(
        dataset,
        common_variables.tfxio,
        common_variables.preprocessing_fn,
        common_variables.transform_input_dataset_metadata,
        force_tf_compat_v1=force_tf_compat_v1,
        max_num_examples=max_num_examples,
        generate_dataset=True)
  logging.info("Intermediate outputs regenerated.")


def _get_batched_records(dataset, force_tf_compat_v1, max_num_examples=None):
  """Returns a (batch_size, iterator for batched records) tuple for the dataset.

  Args:
    dataset: BenchmarkDataset object.
    force_tf_compat_v1: If False then Transform will use its native TF2 version,
      if True then Transform will use its TF1 version.
    max_num_examples: Maximum number of examples to read from the dataset.

  Returns:
    Tuple of (batch_size, iterator for batched records), where records are
    decoded tf.train.Examples.
  """
  batch_size = 1000
  common_variables = _get_common_variables(dataset, force_tf_compat_v1)
  converter = example_coder.ExamplesToRecordBatchDecoder(
      common_variables.transform_input_dataset_metadata.schema
      .SerializeToString())
  serialized_records = benchmark_utils.batched_iterator(
      dataset.read_raw_dataset(deserialize=False, limit=max_num_examples),
      batch_size)
  records = [converter.DecodeBatch(x) for x in serialized_records]
  return batch_size, records


class TFTBenchmarkBase(benchmark_base.BenchmarkBase):
  """TFT benchmark base class."""

  def __init__(self, dataset, **kwargs):
    # Benchmark runners may pass extraneous arguments we don't care about.
    del kwargs
    super().__init__()
    self._dataset = dataset

  def report_benchmark(self, **kwargs):
    if "extras" not in kwargs:
      kwargs["extras"] = {}
    # Note that the GIT_COMMIT_ID is not included in the packages themselves:
    # it must be injected by an external script.
    kwargs["extras"]["commit_tfx"] = (getattr(tfx, "GIT_COMMIT_ID", None) or
                                      getattr(tfx, "__version__", None))
    kwargs["extras"]["commit_tft"] = (getattr(tft, "GIT_COMMIT_ID", None) or
                                      getattr(tft, "__version__", None))
    super().report_benchmark(**kwargs)

  def _benchmarkAnalyzeAndTransformDatasetCommon(self, force_tf_compat_v1):
    """Common implementation to benchmark AnalyzeAndTransformDataset."""
    common_variables = _get_common_variables(self._dataset, force_tf_compat_v1)

    pipeline = self._create_beam_pipeline()
    _ = pipeline | _AnalyzeAndTransformDataset(
        self._dataset,
        common_variables.tfxio,
        common_variables.preprocessing_fn,
        common_variables.transform_input_dataset_metadata,
        force_tf_compat_v1=force_tf_compat_v1,
        max_num_examples=self._max_num_examples())
    start = time.time()
    result = pipeline.run()
    result.wait_until_finish()
    end = time.time()
    delta = end - start

    self.report_benchmark(
        iters=1,
        wall_time=delta,
        extras={
            "num_examples":
                self._dataset.num_examples(limit=self._max_num_examples())
        })

  def benchmarkAnalyzeAndTransformDataset(self):
    """Benchmark AnalyzeAndTransformDataset for TFT's TF1 implementation.

    Runs AnalyzeAndTransformDataset in a Beam pipeline. Records the wall time
    taken for the whole pipeline.
    """
    self._benchmarkAnalyzeAndTransformDatasetCommon(force_tf_compat_v1=True)

  def benchmarkTF2AnalyzeAndTransformDataset(self):
    """Benchmark AnalyzeAndTransformDataset for TFT's TF2 implementation.

    Runs AnalyzeAndTransformDataset in a Beam pipeline. Records the wall time
    taken for the whole pipeline.
    """
    self._benchmarkAnalyzeAndTransformDatasetCommon(force_tf_compat_v1=False)

  def _benchmarkRunMetaGraphDoFnManualActuationCommon(self, force_tf_compat_v1):
    """Common implementation to benchmark RunMetaGraphDoFn "manually"."""
    common_variables = _get_common_variables(self._dataset, force_tf_compat_v1)
    batch_size, batched_records = _get_batched_records(self._dataset,
                                                       force_tf_compat_v1,
                                                       self._max_num_examples())
    fn = tft_beam_impl._RunMetaGraphDoFn(  # pylint: disable=protected-access
        tf_config=None,
        shared_graph_state_handle=shared.Shared(),
        passthrough_keys=set(),
        exclude_outputs=None,
        use_tf_compat_v1=force_tf_compat_v1,
        input_tensor_adapter_config=common_variables.tfxio.TensorAdapterConfig(
        ))
    fn.setup()

    start = time.time()
    for batch in batched_records:
      _ = list(
          fn.process(
              batch,
              saved_model_dir=self._dataset.tft_saved_model_path(
                  force_tf_compat_v1)))
    end = time.time()
    delta = end - start
    self.report_benchmark(
        iters=1,
        wall_time=delta,
        extras={
            "batch_size":
                batch_size,
            "num_examples":
                self._dataset.num_examples(limit=self._max_num_examples())
        })

  def benchmarkRunMetaGraphDoFnManualActuation(self):
    """Benchmark RunMetaGraphDoFn "manually" for TFT's TF1 implementation.

    Runs RunMetaGraphDoFn "manually" outside of a Beam pipeline. Records the
    wall time taken.
    """
    self._benchmarkRunMetaGraphDoFnManualActuationCommon(
        force_tf_compat_v1=True)

  def benchmarkTF2RunMetaGraphDoFnManualActuation(self):
    """Benchmark RunMetaGraphDoFn "manually" for TFT's TF2 implementation.

    Runs RunMetaGraphDoFn "manually" outside of a Beam pipeline. Records the
    wall time taken.
    """
    self._benchmarkRunMetaGraphDoFnManualActuationCommon(
        force_tf_compat_v1=False)

  def benchmarkRunMetagraphDoFnAtTFLevel(self):
    """Benchmark RunMetaGraphDoFn at the TF level for TFT's TF1 implementation.

    Benchmarks the parts of RunMetaGraphDoFn that involve feeding and
    fetching from the TFT SavedModel. Records the wall time taken.

    Note that this benchmark necessarily duplicates code directly from TFT
    since it's benchmarking the low-level internals of TFT, which are not
    exposed for use in this way.
    """
    common_variables = _get_common_variables(
        self._dataset, force_tf_compat_v1=True)
    tf_config = tft_beam_impl._FIXED_PARALLELISM_TF_CONFIG  # pylint: disable=protected-access

    # This block copied from _GraphStateCompatV1.__init__
    with tf.compat.v1.Graph().as_default() as graph:
      session = tf.compat.v1.Session(graph=graph, config=tf_config)
      with session.as_default():
        inputs, outputs = (
            saved_transform_io.partially_apply_saved_transform_internal(
                self._dataset.tft_saved_model_path(force_tf_compat_v1=True),
                {}))
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

    batch_size, batched_records = _get_batched_records(
        self._dataset,
        force_tf_compat_v1=True,
        max_num_examples=self._max_num_examples())

    input_tensor_adapter = tensor_adapter.TensorAdapter(
        common_variables.tfxio.TensorAdapterConfig())

    # This block copied from _RunMetaGraphDoFn._handle_batch
    start = time.time()
    for batch in batched_records:
      feed_by_name = input_tensor_adapter.ToBatchTensors(
          batch, produce_eager_tensors=False)
      feed_list = [feed_by_name[name] for name in input_tensor_keys]
      outputs_list = callable_get_outputs(*feed_list)
      _ = {key: value for key, value in zip(outputs_tensor_keys, outputs_list)}
    end = time.time()
    delta = end - start

    self.report_benchmark(
        iters=1,
        wall_time=delta,
        extras={
            "batch_size":
                batch_size,
            "num_examples":
                self._dataset.num_examples(limit=self._max_num_examples())
        })

  def benchmarkTF2RunMetagraphDoFnAtTFLevel(self):
    """Benchmark RunMetaGraphDoFn at the TF level for TFT's TF2 implementation.

    Benchmarks the parts of RunMetaGraphDoFn that involve feeding and
    fetching from the TFT SavedModel. Records the wall time taken.

    Note that this benchmark necessarily duplicates code directly from TFT
    since it's benchmarking the low-level internals of TFT, which are not
    exposed for use in this way.
    """
    common_variables = _get_common_variables(
        self._dataset, force_tf_compat_v1=False)
    tensor_adapter_config = common_variables.tfxio.TensorAdapterConfig()

    # This block copied from _GraphStateV2.__init__
    saved_model_loader = saved_transform_io_v2.SavedModelLoader(
        self._dataset.tft_saved_model_path(force_tf_compat_v1=False))
    callable_get_outputs = saved_model_loader.apply_transform_model
    # We ignore the schema, and assume there are no excluded outputs.
    outputs_tensor_keys = set(saved_model_loader.structured_outputs.keys())
    saved_model_loader.finalize(
        tensor_adapter_config.tensor_representations.keys(),
        outputs_tensor_keys)

    batch_size, batched_records = _get_batched_records(
        self._dataset,
        force_tf_compat_v1=False,
        max_num_examples=self._max_num_examples())

    input_tensor_adapter = tensor_adapter.TensorAdapter(tensor_adapter_config)

    # This block copied from _RunMetaGraphDoFn._handle_batch
    start = time.time()
    for batch in batched_records:
      feed_dict = input_tensor_adapter.ToBatchTensors(
          batch, produce_eager_tensors=True)
      _ = callable_get_outputs(feed_dict)
    end = time.time()
    delta = end - start

    self.report_benchmark(
        iters=1,
        wall_time=delta,
        extras={
            "batch_size":
                batch_size,
            "num_examples":
                self._dataset.num_examples(limit=self._max_num_examples())
        })
