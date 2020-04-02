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

import logging
import os
import subprocess
import sys

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
from tfx.orchestration.kubeflow import test_utils
from tfx.proto import evaluator_pb2
from tfx.proto import trainer_pb2
from tfx.types import standard_artifacts
from tfx.utils import dsl_utils
from tfx.utils import path_utils


class KubeflowGCPIntegrationTest(test_utils.BaseKubeflowTest):

  def _delete_ai_platform_model(self, model_name):
    """Delete pushed model in AI Platform."""
    # In order to delete model, all versions in the model must be deleted first.
    versions_command = [
        'gcloud', 'ai-platform', 'versions', 'list',
        '--model=%s' % model_name
    ]
    versions = subprocess.run(versions_command, stdout=subprocess.PIPE)

    if versions.returncode == 0:
      absl.logging.info('Model %s has versions %s' %
                        (model_name, versions.stdout))

      # First line of the output is the header: [NAME] [DEPLOYMENT_URI] [STATE]
      # By specification of test case, the latest version is the default model,
      # which needs to be deleted last.
      for version in versions.stdout.decode('utf-8').strip('\n').split(
          '\n')[1:]:
        version = version.split()[0]
        absl.logging.info('Deleting version %s of model %s' %
                          (version, model_name))
        version_delete_command = [
            'gcloud', '--quiet', 'ai-platform', 'versions', 'delete', version,
            '--model=%s' % model_name
        ]
        subprocess.run(version_delete_command, check=True)

    absl.logging.info('Deleting model %s' % model_name)
    subprocess.run(
        ['gcloud', '--quiet', 'ai-platform', 'models', 'delete', model_name],
        check=True)

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

  def testCsvExampleGenOnDataflowRunner(self):
    """CsvExampleGen-only test pipeline on DataflowRunner invocation."""
    pipeline_name = 'kubeflow-csv-example-gen-dataflow-test-{}'.format(
        self._random_id())
    pipeline = self._create_dataflow_pipeline(pipeline_name, [
        CsvExampleGen(input=dsl_utils.csv_input(self._data_root)),
    ])
    self._compile_and_run_pipeline(pipeline)

  def testStatisticsGenOnDataflowRunner(self):
    """StatisticsGen-only test pipeline on DataflowRunner."""
    pipeline_name = 'kubeflow-statistics-gen-dataflow-test-{}'.format(
        self._random_id())
    pipeline = self._create_dataflow_pipeline(pipeline_name, [
        self.raw_examples_importer,
        StatisticsGen(examples=self.raw_examples_importer.outputs['result'])
    ])
    self._compile_and_run_pipeline(pipeline)

  def testTransformOnDataflowRunner(self):
    """Transform-only test pipeline on DataflowRunner."""
    pipeline_name = 'kubeflow-transform-dataflow-test-{}'.format(
        self._random_id())
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
        self._random_id())
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

  def getCaipTrainingArgs(self, pipeline_name):
    """Training args for Google CAIP Training."""
    return {
        'project': self._gcp_project_id,
        'region': self._gcp_region,
        'jobDir': os.path.join(self._pipeline_root(pipeline_name), 'tmp'),
        'masterConfig': {
            'imageUri': self._container_image,
        },
    }

  def getCaipTrainingArgsForDistributed(self, pipeline_name):
    """Training args to test that distributed training is behaves properly."""
    args = self.getCaipTrainingArgs(pipeline_name)
    args.update({
        'scaleTier': 'CUSTOM',
        'masterType': 'large_model',
        'parameterServerType': 'standard',
        'parameterServerCount': 1,
        'workerType': 'standard',
        'workerCount': 2,
    })
    return args

  def assertNumberOfTrainerOutputIsOne(self, pipeline_name):
    """Make sure the number of trainer executions and output models."""
    # There must be only one execution of Trainer.
    trainer_output_base_dir = os.path.join(
        self._pipeline_root(pipeline_name), 'Trainer', 'model')
    trainer_outputs = tf.io.gfile.listdir(trainer_output_base_dir)
    self.assertEqual(1, len(trainer_outputs))

    # There must be only one saved models each for serving and eval.
    model_uri = os.path.join(trainer_output_base_dir, trainer_outputs[0])
    self.assertEqual(
        1, len(tf.io.gfile.listdir(path_utils.eval_model_dir(model_uri))))
    self.assertEqual(
        1,
        len(
            tf.io.gfile.listdir(
                os.path.join(
                    path_utils.serving_model_dir(model_uri), 'export',
                    'chicago-taxi'))))

  def testAIPlatformTrainerPipeline(self):
    """Trainer-only test pipeline on AI Platform Training."""
    pipeline_name = 'kubeflow-aip-trainer-test-{}'.format(self._random_id())
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
                    self.getCaipTrainingArgsForDistributed(pipeline_name)
            })
    ])
    self._compile_and_run_pipeline(pipeline)
    self.assertNumberOfTrainerOutputIsOne(pipeline_name)

  def testAIPlatformGenericTrainerPipeline(self):
    """Trainer-only pipeline on AI Platform Training with GenericTrainer."""
    pipeline_name = 'kubeflow-aip-generic-trainer-test-{}'.format(
        self._random_id())
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
                    self.getCaipTrainingArgs(pipeline_name)
            })
    ])
    self._compile_and_run_pipeline(pipeline)
    self.assertNumberOfTrainerOutputIsOne(pipeline_name)
  # TODO(b/150661783): Add tests using distributed training with a generic
  #  trainer.
  # TODO(b/150576271): Add Trainer tests using Keras models.

  # TODO(muchida): Identify more model types to ensure models trained in TF 2
  # works with CAIP prediction service.
  def testAIPlatformPusherPipeline(self):
    """Pusher-only test pipeline to AI Platform Prediction."""
    pipeline_name_base = 'kubeflow-aip-pusher-test-{}'.format(self._random_id())
    # AI Platform does not accept '-' in the model name.
    model_name = ('%s_model' % pipeline_name_base).replace('-', '_')
    self.addCleanup(self._delete_ai_platform_model, model_name)

    def _pusher(model_importer, model_blessing_importer):
      return Pusher(
          custom_executor_spec=executor_spec.ExecutorClassSpec(
              ai_platform_pusher_executor.Executor),
          model=model_importer.outputs['result'],
          model_blessing=model_blessing_importer.outputs['result'],
          custom_config={
              ai_platform_pusher_executor.SERVING_ARGS_KEY: {
                  'model_name': model_name,
                  'project_id': self._gcp_project_id,
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
  logging.basicConfig(stream=sys.stdout, level=logging.INFO)
  tf.test.main()
