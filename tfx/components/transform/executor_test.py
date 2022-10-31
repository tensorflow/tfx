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
"""Tests for tfx.components.transform.executor."""

import copy
import json
import os
import tempfile
from unittest import mock

from absl.testing import parameterized
import apache_beam as beam
import tensorflow as tf
import tensorflow_data_validation as tfdv
import tensorflow_transform as tft
from tensorflow_transform.beam import tft_unit
from tfx import types
from tfx.components.testdata.module_file import transform_module
from tfx.components.transform import executor
from tfx.dsl.io import fileio
from tfx.proto import example_gen_pb2
from tfx.proto import transform_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx.utils import io_utils
from tfx.utils import name_utils
from tfx.utils import proto_utils


class _TempPath(types.Artifact):
  TYPE_NAME = 'TempPath'


# TODO(b/122478841): Add more detailed tests.
class ExecutorTest(tft_unit.TransformTestCase):

  _TEMP_ARTIFACTS_DIR = tempfile.mkdtemp()
  _SOURCE_DATA_DIR = os.path.join(
      os.path.dirname(os.path.dirname(__file__)), 'testdata')
  _ARTIFACT1_URI = os.path.join(_TEMP_ARTIFACTS_DIR, 'example_gen1')
  _ARTIFACT2_URI = os.path.join(_TEMP_ARTIFACTS_DIR, 'example_gen2')

  _SOURCE_EXAMPLE_DIR = 'csv_example_gen'

  # File format is not specified, will default to 'tfrecords_gzip'.
  _FILE_FORMAT = None
  _PAYLOAD_FORMAT = example_gen_pb2.FORMAT_TF_EXAMPLE

  _PREPROCESSING_FN = transform_module.preprocessing_fn
  _STATS_OPTIONS_UPDATER_FN = transform_module.stats_options_updater_fn

  _SCHEMA_ARTIFACT_DIR = 'schema_gen'
  _MODULE_FILE = 'module_file/transform_module.py'

  _TEST_COUNTERS = {
      'num_instances': 24909,
      'total_columns_count': 18,
      'analyze_columns_count': 9,
      'transform_columns_count': 17,
      'metric_committed_sum': 90
  }

  _CACHE_TEST_METRICS = {
      'num_instances_tfx.Transform_1st_run': 24909,
      'num_instances_tfx.Transform_2nd_run': 15000,
      'num_instances_tfx.DataValidation_1st_run': 24909,
      'num_instances_tfx.DataValidation_2nd_run': 24909
  }

  # executor_v2_test.py overrides this to False.
  def _use_force_tf_compat_v1(self):
    return True

  def _get_dataset_size(self, files):
    if tf.executing_eagerly():
      return sum(
          1 for _ in tf.data.TFRecordDataset(files, compression_type='GZIP'))
    else:
      result = 0
      for file in files:
        result += sum(1 for _ in tf.compat.v1.io.tf_record_iterator(
            file, tf.io.TFRecordOptions(compression_type='GZIP')))
      return result

  @classmethod
  def setUpClass(cls):
    super(ExecutorTest, cls).setUpClass()

    source_example_dir = os.path.join(cls._SOURCE_DATA_DIR,
                                      cls._SOURCE_EXAMPLE_DIR)

    io_utils.copy_dir(source_example_dir, cls._ARTIFACT1_URI)
    io_utils.copy_dir(source_example_dir, cls._ARTIFACT2_URI)

    # Duplicate the number of train and eval records such that
    # second artifact has twice as many as first.
    artifact2_pattern = os.path.join(cls._ARTIFACT2_URI, '*', '*')
    artifact2_files = fileio.glob(artifact2_pattern)
    for filepath in artifact2_files:
      directory, filename = os.path.split(filepath)
      io_utils.copy_file(filepath, os.path.join(directory, 'dup_' + filename))

  def _set_up_input_artifacts(self):
    example1 = standard_artifacts.Examples()
    example1.uri = self._ARTIFACT1_URI
    example1.split_names = artifact_utils.encode_split_names(['train', 'eval'])
    if self._FILE_FORMAT is not None:
      example1.set_string_custom_property('file_format', self._FILE_FORMAT)
    example1.set_string_custom_property(
        key='payload_format',
        value=example_gen_pb2.PayloadFormat.Name(self._PAYLOAD_FORMAT))
    example2 = copy.deepcopy(example1)
    example2.uri = self._ARTIFACT2_URI
    self._example_artifacts = [example1, example2]

  def _get_output_data_dir(self, sub_dir=None):
    test_dir = self._testMethodName
    if sub_dir is not None:
      test_dir = os.path.join(test_dir, sub_dir)
    return os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        test_dir)

  def _make_base_do_params(self, source_data_dir, output_data_dir):
    self._set_up_input_artifacts()

    schema_artifact = standard_artifacts.Schema()
    schema_artifact.uri = os.path.join(source_data_dir,
                                       self._SCHEMA_ARTIFACT_DIR)

    self._input_dict = {
        standard_component_specs.EXAMPLES_KEY: self._example_artifacts[:1],
        standard_component_specs.SCHEMA_KEY: [schema_artifact],
    }

    # Create output dict.
    self._transformed_output = standard_artifacts.TransformGraph()
    self._transformed_output.uri = os.path.join(output_data_dir,
                                                'transformed_graph')
    transformed1 = standard_artifacts.Examples()
    transformed1.uri = os.path.join(output_data_dir, 'transformed_examples',
                                    '0')
    transformed2 = standard_artifacts.Examples()
    transformed2.uri = os.path.join(output_data_dir, 'transformed_examples',
                                    '1')

    self._transformed_example_artifacts = [transformed1, transformed2]

    temp_path_output = _TempPath()
    temp_path_output.uri = tempfile.mkdtemp()
    self._updated_analyzer_cache_artifact = standard_artifacts.TransformCache()
    self._updated_analyzer_cache_artifact.uri = os.path.join(
        self._output_data_dir, 'CACHE')

    self._pre_transform_schema = standard_artifacts.Schema()
    self._pre_transform_schema.uri = os.path.join(output_data_dir,
                                                  'pre_transform_schema', '0')
    self._pre_transform_stats = standard_artifacts.ExampleStatistics()
    self._pre_transform_stats.uri = os.path.join(output_data_dir,
                                                 'pre_transform_stats', '0')
    self._post_transform_schema = standard_artifacts.Schema()
    self._post_transform_schema.uri = os.path.join(output_data_dir,
                                                   'post_transform_schema', '0')
    self._post_transform_stats = standard_artifacts.ExampleStatistics()
    self._post_transform_stats.uri = os.path.join(output_data_dir,
                                                  'post_transform_stats', '0')
    self._post_transform_anomalies = standard_artifacts.ExampleAnomalies()
    self._post_transform_anomalies.uri = os.path.join(
        output_data_dir, 'post_transform_anomalies', '0')

    self._output_dict = {
        standard_component_specs.TRANSFORM_GRAPH_KEY: [
            self._transformed_output
        ],
        standard_component_specs.TRANSFORMED_EXAMPLES_KEY:
            self._transformed_example_artifacts[:1],
        executor.TEMP_PATH_KEY: [temp_path_output],
        standard_component_specs.UPDATED_ANALYZER_CACHE_KEY: [
            self._updated_analyzer_cache_artifact
        ],
        standard_component_specs.PRE_TRANSFORM_STATS_KEY: [
            self._pre_transform_stats
        ],
        standard_component_specs.PRE_TRANSFORM_SCHEMA_KEY: [
            self._pre_transform_schema
        ],
        standard_component_specs.POST_TRANSFORM_ANOMALIES_KEY: [
            self._post_transform_anomalies
        ],
        standard_component_specs.POST_TRANSFORM_STATS_KEY: [
            self._post_transform_stats
        ],
        standard_component_specs.POST_TRANSFORM_SCHEMA_KEY: [
            self._post_transform_schema
        ],
    }

    # Create exec properties skeleton.
    self._exec_properties = {}

  def setUp(self):
    super().setUp()

    self._output_data_dir = self._get_output_data_dir()
    self._make_base_do_params(self._SOURCE_DATA_DIR, self._output_data_dir)

    # Create exec properties skeleton.
    self._module_file = os.path.join(self._SOURCE_DATA_DIR, self._MODULE_FILE)
    self._preprocessing_fn = name_utils.get_full_name(self._PREPROCESSING_FN)
    self._stats_options_updater_fn = name_utils.get_full_name(
        self._STATS_OPTIONS_UPDATER_FN)
    self._exec_properties[standard_component_specs.SPLITS_CONFIG_KEY] = None
    self._exec_properties[
        standard_component_specs.FORCE_TF_COMPAT_V1_KEY] = int(
            self._use_force_tf_compat_v1())

    # Executor for test.
    self._transform_executor = executor.Executor()

  def _verify_transform_outputs(self,
                                materialize=True,
                                store_cache=True,
                                multiple_example_inputs=False,
                                disable_statistics=False,
                                verify_sharded_statistics=False):
    expected_outputs = ['transformed_graph']

    if store_cache:
      expected_outputs.append('CACHE')
      self.assertNotEqual(
          0, len(fileio.listdir(self._updated_analyzer_cache_artifact.uri)))

    example_artifacts = self._example_artifacts[:1]
    transformed_example_artifacts = self._transformed_example_artifacts[:1]
    if multiple_example_inputs:
      example_artifacts = self._example_artifacts
      transformed_example_artifacts = self._transformed_example_artifacts

    if materialize:
      expected_outputs.append('transformed_examples')

      assert len(example_artifacts) == len(transformed_example_artifacts)
      for example, transformed_example in zip(example_artifacts,
                                              transformed_example_artifacts):
        examples_train_files = fileio.glob(
            os.path.join(example.uri, 'Split-train', '*'))
        transformed_train_files = fileio.glob(
            os.path.join(transformed_example.uri, 'Split-train', '*'))
        self.assertGreater(len(transformed_train_files), 0)

        examples_eval_files = fileio.glob(
            os.path.join(example.uri, 'Split-eval', '*'))
        transformed_eval_files = fileio.glob(
            os.path.join(transformed_example.uri, 'Split-eval', '*'))
        self.assertGreater(len(transformed_eval_files), 0)

        # Construct datasets and count number of records in each split.
        examples_train_count = self._get_dataset_size(examples_train_files)
        transformed_train_count = self._get_dataset_size(
            transformed_train_files)
        examples_eval_count = self._get_dataset_size(examples_eval_files)
        transformed_eval_count = self._get_dataset_size(transformed_eval_files)

        # Check for each split that it contains the same number of records in
        # the input artifact as in the output artifact (i.e 1-to-1 mapping is
        # preserved).
        self.assertEqual(examples_train_count, transformed_train_count)
        self.assertEqual(examples_eval_count, transformed_eval_count)
        self.assertGreater(transformed_train_count, transformed_eval_count)

    if disable_statistics:
      self.assertFalse(
          fileio.exists(
              os.path.join(self._pre_transform_schema.uri,
                           executor._SCHEMA_FILE)))
      self.assertFalse(
          fileio.exists(
              os.path.join(self._post_transform_schema.uri,
                           executor._SCHEMA_FILE)))
      self.assertFalse(
          fileio.exists(
              os.path.join(self._pre_transform_stats.uri, executor.STATS_FILE)))
      self.assertFalse(
          fileio.exists(
              os.path.join(self._post_transform_stats.uri,
                           executor.STATS_FILE)))
      self.assertFalse(
          fileio.exists(
              os.path.join(self._post_transform_anomalies.uri,
                           executor._ANOMALIES_FILE)))

    else:
      expected_outputs.extend([
          'pre_transform_schema', 'pre_transform_stats',
          'post_transform_schema', 'post_transform_stats',
          'post_transform_anomalies'
      ])
      self.assertTrue(
          fileio.exists(
              os.path.join(self._pre_transform_schema.uri,
                           executor._SCHEMA_FILE)))
      self.assertTrue(
          fileio.exists(
              os.path.join(self._post_transform_schema.uri,
                           executor._SCHEMA_FILE)))
      self.assertTrue(
          fileio.exists(
              os.path.join(self._pre_transform_stats.uri, executor.STATS_FILE)))
      self.assertTrue(
          fileio.exists(
              os.path.join(self._post_transform_stats.uri,
                           executor.STATS_FILE)))
      self.assertTrue(
          fileio.exists(
              os.path.join(self._post_transform_anomalies.uri,
                           executor._ANOMALIES_FILE)))
    if verify_sharded_statistics:
      pre_transform_paths = fileio.glob(
          os.path.join(self._pre_transform_stats.uri,
                       executor.SHARDED_STATS_PREFIX + '*'))
      post_transform_paths = fileio.glob(
          os.path.join(self._post_transform_stats.uri,
                       executor.SHARDED_STATS_PREFIX + '*'))

      self.assertNotEmpty(pre_transform_paths)
      self.assertNotEmpty(post_transform_paths)

    # Depending on `materialize` and `store_cache`, check that
    # expected outputs are exactly correct. If either flag is False, its
    # respective output should not be present.
    self.assertCountEqual(expected_outputs,
                          fileio.listdir(self._output_data_dir))

    path_to_saved_model = os.path.join(self._transformed_output.uri,
                                       tft.TFTransformOutput.TRANSFORM_FN_DIR,
                                       tf.saved_model.SAVED_MODEL_FILENAME_PB)
    self.assertTrue(fileio.exists(path_to_saved_model))

  def _run_pipeline_get_metrics(self):
    pipelines = []

    def _create_pipeline_wrapper(*_):
      result = self._makeTestPipeline()
      pipelines.append(result)
      return result

    with tft_unit.mock.patch.object(
        executor.Executor,
        '_make_beam_pipeline',
        autospec=True,
        side_effect=_create_pipeline_wrapper):
      transform_executor = executor.Executor()
      transform_executor.Do(self._input_dict, self._output_dict,
                            self._exec_properties)
    assert len(pipelines) == 1
    return pipelines[0].metrics

  def test_do_with_module_file(self):
    self._exec_properties[
        standard_component_specs.MODULE_FILE_KEY] = self._module_file
    self.assertIsNotNone(
        self._transform_executor._GetStatsOptionsUpdaterFn(
            self._exec_properties))

    self._transform_executor.Do(self._input_dict, self._output_dict,
                                self._exec_properties)
    self._verify_transform_outputs()

  def test_do_with_preprocessing_fn(self):
    self._exec_properties[
        standard_component_specs.PREPROCESSING_FN_KEY] = self._preprocessing_fn
    self.assertIsNone(
        self._transform_executor._GetStatsOptionsUpdaterFn(
            self._exec_properties))

    self._transform_executor.Do(self._input_dict, self._output_dict,
                                self._exec_properties)
    self._verify_transform_outputs()

  def test_do_with_preprocessing_fn_and_stats_updater_fn(self):
    self._exec_properties[
        standard_component_specs.PREPROCESSING_FN_KEY] = self._preprocessing_fn
    self._exec_properties[
        standard_component_specs.STATS_OPTIONS_UPDATER_FN_KEY] = (
            self._stats_options_updater_fn)
    self.assertIsNotNone(
        self._transform_executor._GetStatsOptionsUpdaterFn(
            self._exec_properties))

    self._transform_executor.Do(self._input_dict, self._output_dict,
                                self._exec_properties)
    self._verify_transform_outputs()

  def test_do_with_materialization_disabled(self):
    self._exec_properties[
        standard_component_specs.PREPROCESSING_FN_KEY] = self._preprocessing_fn
    del self._output_dict[standard_component_specs.TRANSFORMED_EXAMPLES_KEY]
    self._transform_executor.Do(self._input_dict, self._output_dict,
                                self._exec_properties)
    self._verify_transform_outputs(materialize=False)

  def test_do_with_cache_materialization_disabled(self):
    self._exec_properties[
        standard_component_specs.PREPROCESSING_FN_KEY] = self._preprocessing_fn
    del self._output_dict[standard_component_specs.UPDATED_ANALYZER_CACHE_KEY]
    self._transform_executor.Do(self._input_dict, self._output_dict,
                                self._exec_properties)
    self._verify_transform_outputs(store_cache=False)

  def test_do_with_statistics_disabled(self):
    self._exec_properties[
        standard_component_specs.MODULE_FILE_KEY] = self._module_file
    self._exec_properties[
        standard_component_specs.DISABLE_STATISTICS_KEY] = True
    for key in [
        standard_component_specs.PRE_TRANSFORM_STATS_KEY,
        standard_component_specs.PRE_TRANSFORM_SCHEMA_KEY,
        standard_component_specs.POST_TRANSFORM_ANOMALIES_KEY,
        standard_component_specs.POST_TRANSFORM_STATS_KEY,
        standard_component_specs.POST_TRANSFORM_SCHEMA_KEY
    ]:
      self._output_dict.pop(key)
    self._transform_executor.Do(self._input_dict, self._output_dict,
                                self._exec_properties)
    self._verify_transform_outputs(disable_statistics=True)

  def test_do_with_sharded_stats(self):
    self._exec_properties[
        standard_component_specs.MODULE_FILE_KEY] = self._module_file
    with mock.patch.object(
        executor.TransformProcessor, '_TFDVWriteShardedOutput') as mock_fn:
      mock_fn.return_value = True
      self._transform_executor.Do(self._input_dict, self._output_dict,
                                  self._exec_properties)
    self._verify_transform_outputs(
        verify_sharded_statistics=tfdv.default_sharded_output_supported())

  def test_do_with_preprocessing_fn_custom_config(self):
    self._exec_properties[
        standard_component_specs.PREPROCESSING_FN_KEY] = self._preprocessing_fn
    self._exec_properties[
        standard_component_specs.CUSTOM_CONFIG_KEY] = json.dumps({
            'VOCAB_SIZE': 1000,
            'OOV_SIZE': 10
        })
    self._transform_executor.Do(self._input_dict, self._output_dict,
                                self._exec_properties)
    self._verify_transform_outputs()

  def test_do_with_preprocessing_fn_and_none_custom_config(self):
    self._exec_properties[
        standard_component_specs.PREPROCESSING_FN_KEY] = self._preprocessing_fn
    self._exec_properties['custom_config'] = json.dumps(None)
    self._transform_executor.Do(self._input_dict, self._output_dict,
                                self._exec_properties)
    self._verify_transform_outputs()

  def test_do_with_no_preprocessing_fn(self):
    with self.assertRaises(ValueError):
      self._transform_executor.Do(self._input_dict, self._output_dict,
                                  self._exec_properties)

  def test_do_with_duplicate_preprocessing_fn(self):
    self._exec_properties[
        standard_component_specs.MODULE_FILE_KEY] = self._module_file
    self._exec_properties[
        standard_component_specs.PREPROCESSING_FN_KEY] = self._preprocessing_fn
    with self.assertRaises(ValueError):
      self._transform_executor.Do(self._input_dict, self._output_dict,
                                  self._exec_properties)

  def test_do_with_multiple_artifacts(self):
    self._exec_properties[
        standard_component_specs.MODULE_FILE_KEY] = self._module_file
    self._input_dict[
        standard_component_specs.EXAMPLES_KEY] = self._example_artifacts
    self._output_dict[standard_component_specs.TRANSFORMED_EXAMPLES_KEY] = (
        self._transformed_example_artifacts)
    self._transform_executor.Do(self._input_dict, self._output_dict,
                                self._exec_properties)
    self._verify_transform_outputs(multiple_example_inputs=True)

  def test_do_with_multiple_artifacts_single_output_artifact(self):
    self._exec_properties[
        standard_component_specs.MODULE_FILE_KEY] = self._module_file
    self._input_dict[
        standard_component_specs.EXAMPLES_KEY] = self._example_artifacts
    transformed = standard_artifacts.Examples()
    transformed.uri = os.path.join(self._output_data_dir,
                                   'transformed_examples')
    self._output_dict[standard_component_specs.TRANSFORMED_EXAMPLES_KEY] = ([
        transformed
    ])
    self._transform_executor.Do(self._input_dict, self._output_dict,
                                self._exec_properties)
    self._verify_transform_outputs(multiple_example_inputs=True)

  def test_do_with_custom_splits(self):
    self._exec_properties[
        standard_component_specs.SPLITS_CONFIG_KEY] = proto_utils.proto_to_json(
            transform_pb2.SplitsConfig(
                analyze=['train'], transform=['train', 'eval']))
    self._exec_properties[
        standard_component_specs.MODULE_FILE_KEY] = self._module_file
    self._transform_executor.Do(self._input_dict, self._output_dict,
                                self._exec_properties)
    self._verify_transform_outputs()

  def test_do_with_empty_analyze_splits(self):
    self._exec_properties[
        standard_component_specs.SPLITS_CONFIG_KEY] = proto_utils.proto_to_json(
            transform_pb2.SplitsConfig(analyze=[], transform=['train', 'eval']))
    self._exec_properties[
        standard_component_specs.MODULE_FILE_KEY] = self._module_file
    with self.assertRaises(ValueError):
      self._transform_executor.Do(self._input_dict, self._output_dict,
                                  self._exec_properties)

  def test_do_with_empty_transform_splits(self):
    self._exec_properties['splits_config'] = proto_utils.proto_to_json(
        transform_pb2.SplitsConfig(analyze=['train'], transform=[]))
    self._exec_properties[
        standard_component_specs.MODULE_FILE_KEY] = self._module_file
    self._output_dict[standard_component_specs.TRANSFORMED_EXAMPLES_KEY] = (
        self._transformed_example_artifacts[:1])

    self._transform_executor.Do(self._input_dict, self._output_dict,
                                self._exec_properties)
    self.assertFalse(
        fileio.exists(
            os.path.join(self._transformed_example_artifacts[0].uri, 'train')))
    self.assertFalse(
        fileio.exists(
            os.path.join(self._transformed_example_artifacts[0].uri, 'eval')))
    path_to_saved_model = os.path.join(self._transformed_output.uri,
                                       tft.TFTransformOutput.TRANSFORM_FN_DIR,
                                       tf.saved_model.SAVED_MODEL_FILENAME_PB)
    self.assertTrue(fileio.exists(path_to_saved_model))

  def test_counters(self):
    self._exec_properties[
        standard_component_specs.PREPROCESSING_FN_KEY] = self._preprocessing_fn
    self._exec_properties[
        standard_component_specs.DISABLE_STATISTICS_KEY] = True
    metrics = self._run_pipeline_get_metrics()

    # The test data has 9909 instances in the train dataset, and 5091 instances
    # in the eval dataset (obtained by running:
    # gqui third_party/py/tfx/components/testdata/csv_example_gen/Split-train/ \
    # data* \ 'select count(*)'
    # )
    # Since the analysis dataset (train) is read twice (once for analysis and
    # once for transform), the expected value of the num_instances counter is:
    # 9909 * 2 + 5091 = 24909. (SE - 10500 * 2 + 4500 = 25500)
    self.assertMetricsCounterEqual(metrics, 'num_instances',
                                   self._TEST_COUNTERS['num_instances'],
                                   ['tfx.Transform'])

    # Since disable_statistics is True, TFDV should see 0 instances.
    self.assertMetricsCounterEqual(metrics, 'num_instances', 0,
                                   ['tfx.DataValidation'])

    # We expect 2 saved_models to be created because this is a 1 phase analysis
    # preprocessing_fn.
    self.assertMetricsCounterEqual(metrics, 'saved_models_created', 2)

    # This should be the size of the preprocessing_fn's inputs dictionary which
    # is 18 (SE - 3) according to the schema.
    self.assertMetricsCounterEqual(metrics, 'total_columns_count',
                                   self._TEST_COUNTERS['total_columns_count'])

    # There are 9 (SE - 2) features that are passed into tft analyzers in the
    # preprocessing_fn.
    self.assertMetricsCounterEqual(metrics, 'analyze_columns_count',
                                   self._TEST_COUNTERS['analyze_columns_count'])

    # In addition, 7 features go through a pure TF map, not including the label,
    # so we expect 9 + 7 + 1 = 17 transform columns.
    self.assertMetricsCounterEqual(
        metrics, 'transform_columns_count',
        self._TEST_COUNTERS['transform_columns_count'])

    # There should be 1 path used for analysis since that's what input_dict
    # specifies.
    self.assertMetricsCounterEqual(metrics, 'analyze_paths_count', 1)

    # Analysis cache optimization is enabled (cache writing).
    self.assertMetricsCounterEqual(metrics, 'analyzer_cache_enabled', 1)

    # StatsGen is disabled for the test.
    self.assertMetricsCounterEqual(metrics, 'disable_statistics', 1)

    # Output materialization is enabled.
    self.assertMetricsCounterEqual(metrics, 'materialize', 1)

    # Estimated stage count is 90 because there are 9 analyzers in the
    # preprocessing_fn and a single span input.
    metric = metrics.query(beam.metrics.MetricsFilter().with_name(
        'estimated_stage_count_with_cache'))['distributions']
    self.assertLen(metric, 1)
    self.assertEqual(metric[0].committed.sum,
                     self._TEST_COUNTERS['metric_committed_sum'])

  @parameterized.named_parameters([('no_1st_input_cache', False),
                                   ('empty_1st_input_cache', True)])
  def test_do_with_cache(self, provide_first_input_cache):
    # First run that creates cache.
    self._exec_properties[
        standard_component_specs.MODULE_FILE_KEY] = self._module_file
    if provide_first_input_cache:
      self._input_dict[standard_component_specs.ANALYZER_CACHE_KEY] = []
    metrics = self._run_pipeline_get_metrics()

    # The test data has 9909 instances in the train dataset, and 5091 instances
    # in the eval dataset. Since the analysis dataset (train) is read twice when
    # no input cache is present (once for analysis and once for transform), the
    # expected value of the num_instances counter is: 9909 * 2 + 5091 = 24909.
    # (SE - 10500 * 2 + 4500 = 25500)
    self.assertMetricsCounterEqual(
        metrics, 'num_instances',
        self._CACHE_TEST_METRICS['num_instances_tfx.Transform_1st_run'],
        ['tfx.Transform'])

    # Additionally we have 24909 (SE - 25500) instances due to generating
    # statistics.
    self.assertMetricsCounterEqual(
        metrics, 'num_instances',
        self._CACHE_TEST_METRICS['num_instances_tfx.DataValidation_1st_run'],
        ['tfx.DataValidation'])
    self._verify_transform_outputs(store_cache=True)

    # Second run from cache.
    self._output_data_dir = self._get_output_data_dir('2nd_run')
    analyzer_cache_artifact = standard_artifacts.TransformCache()
    analyzer_cache_artifact.uri = self._updated_analyzer_cache_artifact.uri

    self._make_base_do_params(self._SOURCE_DATA_DIR, self._output_data_dir)

    self._input_dict[standard_component_specs.ANALYZER_CACHE_KEY] = [
        analyzer_cache_artifact
    ]

    self._exec_properties[
        standard_component_specs.MODULE_FILE_KEY] = self._module_file
    metrics = self._run_pipeline_get_metrics()

    # Since input cache should now cover all analysis (train) paths, the train
    # and eval sets are each read exactly once for transform. Thus, the
    # expected value of the num_instances counter is: 9909 + 5091 = 15000.
    # (SE - 10500 + 4500 = 15000)
    self.assertMetricsCounterEqual(
        metrics, 'num_instances',
        self._CACHE_TEST_METRICS['num_instances_tfx.Transform_2nd_run'],
        ['tfx.Transform'])

    # Additionally we have 24909 (SE - 25500) instances due to generating
    # statistics.
    self.assertMetricsCounterEqual(
        metrics, 'num_instances',
        self._CACHE_TEST_METRICS['num_instances_tfx.DataValidation_2nd_run'],
        ['tfx.DataValidation'])
    self._verify_transform_outputs(store_cache=True)

  @tft_unit.mock.patch.object(executor, '_MAX_ESTIMATED_STAGES_COUNT', 10)
  def test_do_with_cache_disabled_too_many_stages(self):
    self._exec_properties[
        standard_component_specs.MODULE_FILE_KEY] = self._module_file
    self._transform_executor.Do(self._input_dict, self._output_dict,
                                self._exec_properties)
    self._verify_transform_outputs(store_cache=False)
    self.assertFalse(fileio.exists(self._updated_analyzer_cache_artifact.uri))


if __name__ == '__main__':
  tf.test.main()
