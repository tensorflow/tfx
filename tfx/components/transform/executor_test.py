# Lint as: python2, python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.beam import tft_unit
from tfx import types
from tfx.components.testdata.module_file import transform_module
from tfx.components.transform import executor
from tfx.types import artifact_utils
from tfx.types import standard_artifacts


class _TempPath(types.Artifact):
  TYPE_NAME = 'TempPath'


# TODO(b/122478841): Add more detailed tests.
class ExecutorTest(tft_unit.TransformTestCase):

  def _get_output_data_dir(self, sub_dir=None):
    test_dir = self._testMethodName
    if sub_dir is not None:
      test_dir = os.path.join(test_dir, sub_dir)
    return os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        test_dir)

  def _make_base_do_params(self, source_data_dir, output_data_dir):
    # Create input dict.
    examples = standard_artifacts.Examples()
    examples.uri = os.path.join(source_data_dir, 'csv_example_gen')
    examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])
    schema_artifact = standard_artifacts.Schema()
    schema_artifact.uri = os.path.join(source_data_dir, 'schema_gen')

    self._input_dict = {
        executor.EXAMPLES_KEY: [examples],
        executor.SCHEMA_KEY: [schema_artifact],
    }

    # Create output dict.
    self._transformed_output = standard_artifacts.TransformGraph()
    self._transformed_output.uri = os.path.join(output_data_dir,
                                                'transformed_output')
    self._transformed_examples = standard_artifacts.Examples()
    self._transformed_examples.uri = output_data_dir
    self._transformed_examples.split_names = artifact_utils.encode_split_names(
        ['train', 'eval'])
    temp_path_output = _TempPath()
    temp_path_output.uri = tempfile.mkdtemp()

    self._output_dict = {
        executor.TRANSFORM_GRAPH_KEY: [self._transformed_output],
        executor.TRANSFORMED_EXAMPLES_KEY: [self._transformed_examples],
        executor.TEMP_PATH_KEY: [temp_path_output],
    }

    # Create exec properties skeleton.
    self._exec_properties = {}

  def setUp(self):
    super(ExecutorTest, self).setUp()

    self._source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    self._output_data_dir = self._get_output_data_dir()

    self._make_base_do_params(self._source_data_dir, self._output_data_dir)

    # Create exec properties skeleton.
    self._module_file = os.path.join(self._source_data_dir,
                                     'module_file/transform_module.py')
    self._preprocessing_fn = '%s.%s' % (
        transform_module.preprocessing_fn.__module__,
        transform_module.preprocessing_fn.__name__)

    # Executor for test.
    self._transform_executor = executor.Executor()

  def _verify_transform_outputs(self):
    self.assertNotEqual(
        0,
        len(
            tf.io.gfile.listdir(
                os.path.join(self._transformed_examples.uri, 'train'))))
    self.assertNotEqual(
        0,
        len(
            tf.io.gfile.listdir(
                os.path.join(self._transformed_examples.uri, 'eval'))))
    path_to_saved_model = os.path.join(
        self._transformed_output.uri, tft.TFTransformOutput.TRANSFORM_FN_DIR,
        tf.saved_model.SAVED_MODEL_FILENAME_PB)
    self.assertTrue(tf.io.gfile.exists(path_to_saved_model))

  # TODO(b/143355786): Remove _makeTestPipeline once TFX depends on TFT 0.16 and
  # use self._makeTestPipeline instead.
  def _makeTestPipeline(self):

    class _TestPipeline(tft_unit.beam.Pipeline):
      """Test pipeline class that retains pipeline metrics."""

      @property
      def metrics(self):
        return self._run_result.metrics()

      def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_type:
          self._run_result = self.run()
          self._run_result.wait_until_finish()

    return _TestPipeline(
        **tft_unit.test_helpers.make_test_beam_pipeline_kwargs())

  # TODO(b/143355786): Remove _assertMetricsCounterEqual once TFX depends on TFT
  # 0.16 and use self.assertMetricsCounterEqual instead.
  def _assertMetricsCounterEqual(self, metrics, name, expected_count):
    metric = metrics.query(
        tft_unit.beam.metrics.metric.MetricsFilter().with_name(
            name))['counters']
    committed = sum([r.committed for r in metric])
    attempted = sum([r.attempted for r in metric])
    self.assertEqual(committed, attempted)
    self.assertEqual(committed, expected_count)

  def _runPipelineGetMetrics(self, inputs, outputs, exec_properties):
    pipelines = []

    def _create_pipeline_wrapper(*_):
      result = self._makeTestPipeline()
      pipelines.append(result)
      return result

    with tft_unit.mock.patch.object(
        executor.Executor,
        '_CreatePipeline',
        autospec=True,
        side_effect=_create_pipeline_wrapper):
      transform_executor = executor.Executor()
      transform_executor.Do(self._input_dict, self._output_dict,
                            self._exec_properties)
    assert len(pipelines) == 1
    return pipelines[0].metrics

  def testDoWithModuleFile(self):
    self._exec_properties['module_file'] = self._module_file
    self._transform_executor.Do(self._input_dict, self._output_dict,
                                self._exec_properties)
    self._verify_transform_outputs()

  def testDoWithPreprocessingFn(self):
    self._exec_properties['preprocessing_fn'] = self._preprocessing_fn
    self._transform_executor.Do(self._input_dict, self._output_dict,
                                self._exec_properties)
    self._verify_transform_outputs()

  def testDoWithNoPreprocessingFn(self):
    with self.assertRaises(ValueError):
      self._transform_executor.Do(self._input_dict, self._output_dict,
                                  self._exec_properties)

  def testDoWithDuplicatePreprocessingFn(self):
    self._exec_properties['module_file'] = self._module_file
    self._exec_properties['preprocessing_fn'] = self._preprocessing_fn
    with self.assertRaises(ValueError):
      self._transform_executor.Do(self._input_dict, self._output_dict,
                                  self._exec_properties)

  def testCounters(self):
    self._exec_properties['preprocessing_fn'] = self._preprocessing_fn
    metrics = self._runPipelineGetMetrics(self._input_dict, self._output_dict,
                                          self._exec_properties)

    # The test data has 10036 instances in the train dataset, and 4964 instances
    # in the eval dataset (obtained by running:
    #   gqui third_party/tfx/components/testdata/csv_example_gen/train/data* \
    #     'select count(*)'
    # )
    # Since the analysis dataset (train) is read twice (once for analysis and
    # once for transform), the expected value of the num_instances counter is:
    # 10036 * 2 + 4964 = 25036.
    self._assertMetricsCounterEqual(metrics, 'num_instances', 25036)

    # We expect 2 saved_models to be created because this is a 1 phase analysis
    # preprocessing_fn.
    self._assertMetricsCounterEqual(metrics, 'saved_models_created', 2)

    # This should be the size of the preprocessing_fn's inputs dictionary which
    # is 18 according to the schema.
    self._assertMetricsCounterEqual(metrics, 'total_columns_count', 18)

    # There are 9 features that are passed into tft analyzers in the
    # preprocessing_fn.
    self._assertMetricsCounterEqual(metrics, 'analyze_columns_count', 9)

    # In addition, 7 features go through a pure TF map, not including the label,
    # so we expect 9 + 7 + 1 = 17 transform columns.
    self._assertMetricsCounterEqual(metrics, 'transform_columns_count', 17)

  def testDoWithCache(self):

    class InputCache(types.Artifact):
      TYPE_NAME = 'InputCache'

    class OutputCache(types.Artifact):
      TYPE_NAME = 'OutputCache'

    # First run that creates cache.
    output_cache_artifact = OutputCache()
    output_cache_artifact.uri = os.path.join(self._output_data_dir, 'CACHE')

    self._output_dict['cache_output_path'] = [output_cache_artifact]

    self._exec_properties['module_file'] = self._module_file
    self._transform_executor.Do(self._input_dict, self._output_dict,
                                self._exec_properties)
    self._verify_transform_outputs()
    self.assertNotEqual(0,
                        len(tf.io.gfile.listdir(output_cache_artifact.uri)))

    # Second run from cache.
    self._output_data_dir = self._get_output_data_dir('2nd_run')
    input_cache_artifact = InputCache()
    input_cache_artifact.uri = output_cache_artifact.uri

    output_cache_artifact = OutputCache()
    output_cache_artifact.uri = os.path.join(self._output_data_dir, 'CACHE')

    self._make_base_do_params(self._source_data_dir, self._output_data_dir)

    self._input_dict['cache_input_path'] = [input_cache_artifact]
    self._output_dict['cache_output_path'] = [output_cache_artifact]

    self._exec_properties['module_file'] = self._module_file
    self._transform_executor.Do(self._input_dict, self._output_dict,
                                self._exec_properties)

    self._verify_transform_outputs()
    self.assertNotEqual(0,
                        len(tf.io.gfile.listdir(output_cache_artifact.uri)))


if __name__ == '__main__':
  tf.test.main()
