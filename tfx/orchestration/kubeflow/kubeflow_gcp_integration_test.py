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
"""Integration tests for Kubeflow-based orchestrator and GCP backend."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import absl
import tensorflow as tf

from tfx.components.base import executor_spec
from tfx.components.common_nodes.importer_node import ImporterNode
from tfx.components.evaluator.component import Evaluator
from tfx.components.example_gen.csv_example_gen.component import CsvExampleGen
from tfx.components.pusher.component import Pusher
from tfx.components.statistics_gen.component import StatisticsGen
from tfx.components.trainer.component import Trainer
from tfx.components.transform.component import Transform
from tfx.extensions.google_cloud_ai_platform.pusher import executor as ai_platform_pusher_executor
from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor
from tfx.extensions.google_cloud_ai_platform.tuner import component as ai_platform_tuner_component
from tfx.orchestration import test_utils
from tfx.orchestration.kubeflow import test_utils as kubeflow_test_utils
from tfx.proto import evaluator_pb2
from tfx.proto import trainer_pb2
from tfx.proto import tuner_pb2
from tfx.types import standard_artifacts
from tfx.utils import path_utils


class KubeflowGCPIntegrationTest(kubeflow_test_utils.BaseKubeflowTest):

  def setUp(self):
    super(KubeflowGCPIntegrationTest, self).setUp()

    # Example artifacts for testing.
    self.raw_examples_importer = ImporterNode(
        instance_name='raw_examples',
        source_uri=os.path.join(self._testdata_root, 'csv_example_gen'),
        artifact_type=standard_artifacts.Examples,
        reimport=True,
        properties={'split_names': '["train", "eval"]'})

    # Transformed Example artifacts for testing.
    self.transformed_examples_importer = ImporterNode(
        instance_name='transformed_examples',
        source_uri=os.path.join(self._testdata_root, 'transform',
                                'transformed_examples'),
        artifact_type=standard_artifacts.Examples,
        reimport=True,
        properties={'split_names': '["train", "eval"]'})

    # Schema artifact for testing.
    self.schema_importer = ImporterNode(
        instance_name='schema',
        source_uri=os.path.join(self._testdata_root, 'schema_gen'),
        artifact_type=standard_artifacts.Schema,
        reimport=True)

    # TransformGraph artifact for testing.
    self.transform_graph_importer = ImporterNode(
        instance_name='transform_graph',
        source_uri=os.path.join(self._testdata_root, 'transform',
                                'transform_graph'),
        artifact_type=standard_artifacts.TransformGraph,
        reimport=True)

    # Model artifact for testing.
    self.model_1_importer = ImporterNode(
        instance_name='model_1',
        source_uri=os.path.join(self._testdata_root, 'trainer', 'previous'),
        artifact_type=standard_artifacts.Model,
        reimport=True)

    self.model_2_importer = ImporterNode(
        instance_name='model_2',
        source_uri=os.path.join(self._testdata_root, 'trainer', 'current'),
        artifact_type=standard_artifacts.Model,
        reimport=True)

    # ModelBlessing artifact for testing.
    self.model_blessing_importer = ImporterNode(
        instance_name='model_blessing',
        source_uri=os.path.join(self._testdata_root, 'model_validator',
                                'blessed'),
        artifact_type=standard_artifacts.ModelBlessing,
        reimport=True,
        custom_properties={'blessed': 1})

    ### Test data and modules for native Keras trainer and tuner.
    self._iris_tuner_module = os.path.join(self._MODULE_ROOT, 'tuner_module.py')
    self.iris_examples_importer = ImporterNode(
        instance_name='iris_examples',
        source_uri=os.path.join(self._testdata_root, 'iris', 'data'),
        artifact_type=standard_artifacts.Examples,
        reimport=True,
        properties={'split_names': '["train", "eval"]'})
    self.iris_schema_importer = ImporterNode(
        instance_name='iris_schema',
        source_uri=os.path.join(self._testdata_root, 'iris', 'schema'),
        artifact_type=standard_artifacts.Schema,
        reimport=True)

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

  def _getCaipTrainingArgs(self, pipeline_name):
    """Training args for Google CAIP Training."""
    return {
        'project': self._GCP_PROJECT_ID,
        'region': self._GCP_REGION,
        'jobDir': os.path.join(self._pipeline_root(pipeline_name), 'tmp'),
        'masterConfig': {
            'imageUri': self._CONTAINER_IMAGE,
        },
    }

  def _getCaipTrainingArgsForDistributed(self, pipeline_name):
    """Training args to test that distributed training is behaves properly."""
    args = self._getCaipTrainingArgs(pipeline_name)
    args.update({
        'scaleTier': 'CUSTOM',
        'masterType': 'large_model',
        'parameterServerType': 'standard',
        'parameterServerCount': 1,
        'workerType': 'standard',
        'workerCount': 2,
    })
    return args

  def _assertNumberOfTrainerOutputIsOne(self, pipeline_name):
    """Make sure the number of trainer executions and output models."""
    # There must be only one execution of Trainer.
    trainer_output_base_dir = os.path.join(
        self._pipeline_root(pipeline_name), 'Trainer', 'model')
    trainer_outputs = tf.io.gfile.listdir(trainer_output_base_dir)
    self.assertEqual(1, len(trainer_outputs))

    # There must be only one saved models each for serving and eval.
    model_uri = os.path.join(trainer_output_base_dir, trainer_outputs[0])
    eval_model_dir = path_utils.eval_model_dir(model_uri)
    serving_model_dir = path_utils.serving_model_dir(model_uri)
    self.assertEqual(
        1,
        tf.io.gfile.listdir(eval_model_dir).count('saved_model.pb'))
    self.assertEqual(
        1,
        tf.io.gfile.listdir(serving_model_dir).count('saved_model.pb'))

  def testAIPlatformTrainerPipeline(self):
    """Trainer-only test pipeline on AI Platform Training."""
    pipeline_name = 'kubeflow-aip-trainer-test-{}'.format(
        test_utils.random_id())
    pipeline = self._create_pipeline(pipeline_name, [
        self.schema_importer, self.transformed_examples_importer,
        self.transform_graph_importer,
        Trainer(
            custom_executor_spec=executor_spec.ExecutorClassSpec(
                ai_platform_trainer_executor.Executor),
            module_file=self._trainer_module,
            transformed_examples=self.transformed_examples_importer
            .outputs['result'],
            schema=self.schema_importer.outputs['result'],
            transform_graph=self.transform_graph_importer.outputs['result'],
            train_args=trainer_pb2.TrainArgs(num_steps=10),
            eval_args=trainer_pb2.EvalArgs(num_steps=5),
            custom_config={
                ai_platform_trainer_executor.TRAINING_ARGS_KEY:
                    self._getCaipTrainingArgsForDistributed(pipeline_name)
            })
    ])
    self._compile_and_run_pipeline(pipeline)
    self._assertNumberOfTrainerOutputIsOne(pipeline_name)

  def testAIPlatformGenericTrainerPipeline(self):
    """Trainer-only pipeline on AI Platform Training with GenericTrainer."""
    pipeline_name = 'kubeflow-aip-generic-trainer-test-{}'.format(
        test_utils.random_id())
    pipeline = self._create_pipeline(pipeline_name, [
        self.schema_importer,
        self.transformed_examples_importer,
        self.transform_graph_importer,
        Trainer(
            custom_executor_spec=executor_spec.ExecutorClassSpec(
                ai_platform_trainer_executor.GenericExecutor),
            module_file=self._trainer_module,
            transformed_examples=self.transformed_examples_importer
            .outputs['result'],
            schema=self.schema_importer.outputs['result'],
            transform_graph=self.transform_graph_importer.outputs['result'],
            train_args=trainer_pb2.TrainArgs(num_steps=10),
            eval_args=trainer_pb2.EvalArgs(num_steps=5),
            custom_config={
                ai_platform_trainer_executor.TRAINING_ARGS_KEY:
                    self._getCaipTrainingArgs(pipeline_name)
            })
    ])
    self._compile_and_run_pipeline(pipeline)
    self._assertNumberOfTrainerOutputIsOne(pipeline_name)
  # TODO(b/150661783): Add tests using distributed training with a generic
  #  trainer.
  # TODO(b/150576271): Add Trainer tests using Keras models.

  def _assertHyperparametersAreWritten(self, pipeline_name):
    """Make sure the tuner execution and hyperpearameters output."""
    # There must be only one execution of Tuner.
    tuner_output_base_dir = os.path.join(
        self._pipeline_root(pipeline_name), 'Tuner', 'best_hyperparameters')
    tuner_outputs = tf.io.gfile.listdir(tuner_output_base_dir)
    self.assertEqual(1, len(tuner_outputs))

    # There must be only one best hyperparameters.
    best_hyperparameters_uri = os.path.join(tuner_output_base_dir,
                                            tuner_outputs[0])
    self.assertTrue(tf.io.gfile.exists(best_hyperparameters_uri))

  def testAIPlatformDistributedTunerPipeline(self):
    """Tuner-only pipeline for distributed Tuner flock on AIP Training."""
    pipeline_name = 'kubeflow-aip-dist-tuner-test-{}'.format(
        test_utils.random_id())
    pipeline = self._create_pipeline(pipeline_name, [
        self.iris_examples_importer, self.iris_schema_importer,
        ai_platform_tuner_component.Tuner(
            examples=self.iris_examples_importer.outputs['result'],
            module_file=self._iris_tuner_module,
            schema=self.iris_schema_importer.outputs['result'],
            train_args=trainer_pb2.TrainArgs(num_steps=10),
            eval_args=trainer_pb2.EvalArgs(num_steps=5),
            # 3 worker parallel tuning.
            tune_args=tuner_pb2.TuneArgs(num_parallel_trials=3),
            custom_config={
                ai_platform_trainer_executor.TRAINING_ARGS_KEY:
                    self._getCaipTrainingArgs(pipeline_name)
            })
    ])
    self._compile_and_run_pipeline(pipeline)
    self._assertHyperparametersAreWritten(pipeline_name)

  def testAIPlatformPusherPipeline(self):
    """Pusher-only test pipeline to AI Platform Prediction."""
    pipeline_name_base = 'kubeflow-aip-pusher-test-{}'.format(
        test_utils.random_id())
    # AI Platform does not accept '-' in the model name.
    model_name = ('%s_model' % pipeline_name_base).replace('-', '_')
    self.addCleanup(kubeflow_test_utils.delete_ai_platform_model, model_name)

    def _pusher(model_importer, model_blessing_importer):
      return Pusher(
          custom_executor_spec=executor_spec.ExecutorClassSpec(
              ai_platform_pusher_executor.Executor),
          model=model_importer.outputs['result'],
          model_blessing=model_blessing_importer.outputs['result'],
          custom_config={
              ai_platform_pusher_executor.SERVING_ARGS_KEY: {
                  'model_name': model_name,
                  'project_id': self._GCP_PROJECT_ID,
              }
          },
      )

    # Test creation of multiple versions under the same model_name.
    pipeline_name_1 = '%s-1' % pipeline_name_base
    pipeline_1 = self._create_pipeline(pipeline_name_1, [
        self.model_1_importer,
        self.model_blessing_importer,
        _pusher(self.model_1_importer, self.model_blessing_importer),
    ])
    self._compile_and_run_pipeline(pipeline_1)

    pipeline_name_2 = '%s-2' % pipeline_name_base
    pipeline_2 = self._create_pipeline(pipeline_name_2, [
        self.model_2_importer,
        self.model_blessing_importer,
        _pusher(self.model_2_importer, self.model_blessing_importer),
    ])
    self._compile_and_run_pipeline(pipeline_2)


if __name__ == '__main__':
  absl.logging.set_verbosity(absl.logging.INFO)
  tf.test.main()
