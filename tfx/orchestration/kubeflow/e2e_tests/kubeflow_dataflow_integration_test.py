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
"""Integration tests for Kubeflow-based orchestrator and Dataflow."""

import os

import absl
import tensorflow as tf
from tfx.components.evaluator.component import Evaluator
from tfx.components.example_gen.csv_example_gen.component import CsvExampleGen
from tfx.components.statistics_gen.component import StatisticsGen
from tfx.components.transform.component import Transform
from tfx.dsl.components.common import importer
from tfx.orchestration import test_utils
from tfx.orchestration.kubeflow import test_utils as kubeflow_test_utils
from tfx.proto import evaluator_pb2
from tfx.types import standard_artifacts


# TODO(b/202799145): Check whether dataflow jobs have actually been launched.
class KubeflowDataflowIntegrationTest(kubeflow_test_utils.BaseKubeflowTest):

  def setUp(self):
    super().setUp()

    # Example artifacts for testing.
    self.raw_examples_importer = importer.Importer(
        source_uri=os.path.join(self._test_data_dir, 'csv_example_gen'),
        artifact_type=standard_artifacts.Examples,
        reimport=True,
        properties={
            'split_names': '["train", "eval"]'
        }).with_id('raw_examples')

    # Schema artifact for testing.
    self.schema_importer = importer.Importer(
        source_uri=os.path.join(self._test_data_dir, 'schema_gen'),
        artifact_type=standard_artifacts.Schema,
        reimport=True).with_id('schema')

    # Model artifact for testing.
    self.model_1_importer = importer.Importer(
        source_uri=os.path.join(self._test_data_dir, 'trainer', 'previous'),
        artifact_type=standard_artifacts.Model,
        reimport=True).with_id('model_1')

  def testCsvExampleGenOnDataflowRunner(self):
    """CsvExampleGen-only test pipeline on DataflowRunner invocation."""
    pipeline_name = 'kubeflow-csv-example-gen-dataflow-test-{}'.format(
        test_utils.random_id())
    pipeline = self._create_dataflow_pipeline(pipeline_name, [
        CsvExampleGen(input_base=self._data_root),
    ])
    self._compile_and_run_pipeline(pipeline)

  def testStatisticsGenOnDataflowRunner(self):
    """StatisticsGen-only test pipeline on DataflowRunner."""
    pipeline_name = 'kubeflow-statistics-gen-dataflow-test-{}'.format(
        test_utils.random_id())
    pipeline = self._create_dataflow_pipeline(pipeline_name, [
        self.raw_examples_importer,
        StatisticsGen(examples=self.raw_examples_importer.outputs['result'])
    ])
    self._compile_and_run_pipeline(pipeline)

  def testTransformOnDataflowRunner(self):
    """Transform-only test pipeline on DataflowRunner."""
    pipeline_name = 'kubeflow-transform-dataflow-test-{}'.format(
        test_utils.random_id())
    pipeline = self._create_dataflow_pipeline(pipeline_name, [
        self.raw_examples_importer, self.schema_importer,
        Transform(
            examples=self.raw_examples_importer.outputs['result'],
            schema=self.schema_importer.outputs['result'],
            module_file=self._transform_module)
    ])
    self._compile_and_run_pipeline(pipeline)

  def testEvaluatorOnDataflowRunner(self):
    """Evaluator-only test pipeline on DataflowRunner."""
    pipeline_name = 'kubeflow-evaluator-dataflow-test-{}'.format(
        test_utils.random_id())
    pipeline = self._create_dataflow_pipeline(pipeline_name, [
        self.raw_examples_importer, self.model_1_importer,
        Evaluator(
            examples=self.raw_examples_importer.outputs['result'],
            model=self.model_1_importer.outputs['result'],
            feature_slicing_spec=evaluator_pb2.FeatureSlicingSpec(specs=[
                evaluator_pb2.SingleSlicingSpec(
                    column_for_slicing=['trip_start_hour'])
            ]))
    ])
    self._compile_and_run_pipeline(pipeline)


if __name__ == '__main__':
  absl.logging.set_verbosity(absl.logging.INFO)
  tf.test.main()
