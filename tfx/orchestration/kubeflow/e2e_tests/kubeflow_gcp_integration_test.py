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

import os

import absl
from googleapiclient import discovery
from googleapiclient import errors as googleapiclient_errors
import tensorflow as tf
from tfx import v1 as tfx
from tfx.components.pusher.component import Pusher
from tfx.components.trainer.component import Trainer
from tfx.dsl.components.base import executor_spec
from tfx.dsl.components.common import importer
from tfx.dsl.io import fileio
from tfx.extensions.google_cloud_ai_platform import constants
from tfx.extensions.google_cloud_ai_platform import runner
from tfx.extensions.google_cloud_ai_platform.pusher import executor as ai_platform_pusher_executor
from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor
from tfx.extensions.google_cloud_ai_platform.tuner import component as ai_platform_tuner_component
from tfx.extensions.google_cloud_ai_platform.tuner import executor as ai_platform_tuner_executor
from tfx.extensions.google_cloud_big_query.pusher import executor as bigquery_pusher_executor
from tfx.orchestration import test_utils
from tfx.orchestration.kubeflow import test_utils as kubeflow_test_utils
from tfx.proto import trainer_pb2
from tfx.proto import tuner_pb2
from tfx.types import standard_artifacts
from tfx.utils import path_utils
from tfx.utils import telemetry_utils


class KubeflowGCPIntegrationTest(kubeflow_test_utils.BaseKubeflowTest):

  def setUp(self):
    super().setUp()

    # Transformed Example artifacts for testing.
    self.transformed_examples_importer = importer.Importer(
        source_uri=os.path.join(self._test_data_dir, 'transform',
                                'transformed_examples'),
        artifact_type=standard_artifacts.Examples,
        reimport=True,
        properties={
            'split_names': '["train", "eval"]'
        }).with_id('transformed_examples')

    # Schema artifact for testing.
    self.schema_importer = importer.Importer(
        source_uri=os.path.join(self._test_data_dir, 'schema_gen'),
        artifact_type=standard_artifacts.Schema,
        reimport=True).with_id('schema')

    # TransformGraph artifact for testing.
    self.transform_graph_importer = importer.Importer(
        source_uri=os.path.join(self._test_data_dir, 'transform',
                                'transform_graph'),
        artifact_type=standard_artifacts.TransformGraph,
        reimport=True).with_id('transform_graph')

    # Model artifact for testing.
    self.model_1_importer = importer.Importer(
        source_uri=os.path.join(self._test_data_dir, 'trainer', 'previous'),
        artifact_type=standard_artifacts.Model,
        reimport=True).with_id('model_1')

    self.model_2_importer = importer.Importer(
        source_uri=os.path.join(self._test_data_dir, 'trainer', 'current'),
        artifact_type=standard_artifacts.Model,
        reimport=True).with_id('model_2')

    # ModelBlessing artifact for testing.
    self.model_blessing_1_importer = importer.Importer(
        source_uri=os.path.join(self._test_data_dir, 'model_validator',
                                'blessed'),
        artifact_type=standard_artifacts.ModelBlessing,
        reimport=True,
        custom_properties={
            'blessed': 1
        }).with_id('model_blessing_1')

    self.model_blessing_2_importer = importer.Importer(
        source_uri=os.path.join(self._test_data_dir, 'model_validator',
                                'blessed'),
        artifact_type=standard_artifacts.ModelBlessing,
        reimport=True,
        custom_properties={
            'blessed': 1
        }).with_id('model_blessing_2')

    ### Test data and modules for native Keras trainer and tuner.
    self._penguin_tuner_module = os.path.join(self._MODULE_ROOT,
                                              'tuner_module.py')
    self.penguin_examples_importer = importer.Importer(
        source_uri=os.path.join(self._test_data_dir, 'penguin', 'data'),
        artifact_type=standard_artifacts.Examples,
        reimport=True,
        properties={
            'split_names': '["train", "eval"]'
        }).with_id('penguin_examples')
    self.penguin_schema_importer = importer.Importer(
        source_uri=os.path.join(self._test_data_dir, 'penguin', 'schema'),
        artifact_type=standard_artifacts.Schema,
        reimport=True).with_id('penguin_schema')

  def _getCaipTrainingArgs(self, pipeline_name):
    """Training args for Google CAIP Training."""
    return {
        'project': self._GCP_PROJECT_ID,
        'region': self._GCP_REGION,
        'jobDir': os.path.join(self._pipeline_root(pipeline_name), 'tmp'),
        'masterConfig': {
            'imageUri': self.container_image,
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

  def _getVertexTrainingArgs(self, pipeline_name):
    """Training args for Google Vertex AI Training."""
    return {
        'project': self._GCP_PROJECT_ID,
        'job_spec': {
            'worker_pool_specs': [{
                'machine_spec': {
                    'machine_type': 'e2-standard-8'
                },
                'replica_count': 1,
                'container_spec': {
                    'image_uri': self.container_image
                }
            }]
        }
    }

  def _assertNumberOfTrainerOutputIsOne(self, pipeline_name):
    """Make sure the number of trainer executions and output models."""
    # There must be only one execution of Trainer.
    trainer_output_base_dir = os.path.join(
        self._pipeline_root(pipeline_name), 'Trainer', 'model')
    trainer_outputs = fileio.listdir(trainer_output_base_dir)
    self.assertEqual(1, len(trainer_outputs))

    # There must be only one saved models each for serving and eval.
    model_uri = os.path.join(trainer_output_base_dir, trainer_outputs[0])
    eval_model_dir = path_utils.eval_model_dir(model_uri)
    serving_model_dir = path_utils.serving_model_dir(model_uri)
    self.assertEqual(1, fileio.listdir(eval_model_dir).count('saved_model.pb'))
    self.assertEqual(1,
                     fileio.listdir(serving_model_dir).count('saved_model.pb'))

  def _make_unique_pipeline_name(self, prefix):
    return '-'.join([prefix, 'test', test_utils.random_id()])

  def testAIPlatformTrainerPipeline(self):
    """Trainer-only test pipeline on AI Platform Training."""
    pipeline_name = self._make_unique_pipeline_name('kubeflow-aip-trainer')
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
    pipeline_name = self._make_unique_pipeline_name(
        'kubeflow-aip-generic-trainer')
    pipeline = self._create_pipeline(pipeline_name, [
        self.schema_importer, self.transformed_examples_importer,
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
    tuner_outputs = fileio.listdir(tuner_output_base_dir)
    self.assertEqual(1, len(tuner_outputs))

    # There must be only one best hyperparameters.
    best_hyperparameters_uri = os.path.join(tuner_output_base_dir,
                                            tuner_outputs[0])
    self.assertTrue(fileio.exists(best_hyperparameters_uri))

  def testVertexSequentialTunerPipeline(self):
    """Tuner-only pipeline for sequential Tuner flock on Vertex AI Training."""
    pipeline_name = self._make_unique_pipeline_name(
        'kubeflow-vertex-seq-tuner')
    pipeline = self._create_pipeline(
        pipeline_name,
        [
            self.penguin_examples_importer,
            self.penguin_schema_importer,
            ai_platform_tuner_component.Tuner(
                examples=self.penguin_examples_importer.outputs['result'],
                module_file=self._penguin_tuner_module,
                schema=self.penguin_schema_importer.outputs['result'],
                train_args=trainer_pb2.TrainArgs(num_steps=1),
                eval_args=trainer_pb2.EvalArgs(num_steps=1),
                # Single worker sequential tuning.
                tune_args=tuner_pb2.TuneArgs(num_parallel_trials=1),
                custom_config={
                    ai_platform_tuner_executor.TUNING_ARGS_KEY:
                        self._getVertexTrainingArgs(pipeline_name),
                    constants.ENABLE_VERTEX_KEY:
                        True,
                    constants.VERTEX_REGION_KEY:
                        self._GCP_REGION
                })
        ])
    self._compile_and_run_pipeline(pipeline)
    self._assertHyperparametersAreWritten(pipeline_name)

  def testVertexDistributedTunerPipeline(self):
    """Tuner-only pipeline for distributed Tuner flock on Vertex AI Training."""
    pipeline_name = self._make_unique_pipeline_name(
        'kubeflow-vertex-dist-tuner')
    pipeline = self._create_pipeline(
        pipeline_name,
        [
            self.penguin_examples_importer,
            self.penguin_schema_importer,
            ai_platform_tuner_component.Tuner(
                examples=self.penguin_examples_importer.outputs['result'],
                module_file=self._penguin_tuner_module,
                schema=self.penguin_schema_importer.outputs['result'],
                train_args=trainer_pb2.TrainArgs(num_steps=10),
                eval_args=trainer_pb2.EvalArgs(num_steps=5),
                # 3 worker parallel tuning.
                tune_args=tuner_pb2.TuneArgs(num_parallel_trials=3),
                custom_config={
                    ai_platform_tuner_executor.TUNING_ARGS_KEY:
                        self._getVertexTrainingArgs(pipeline_name),
                    constants.ENABLE_VERTEX_KEY:
                        True,
                    constants.VERTEX_REGION_KEY:
                        self._GCP_REGION
                })
        ])
    self._compile_and_run_pipeline(pipeline)
    self._assertHyperparametersAreWritten(pipeline_name)

  def testAIPlatformDistributedTunerPipeline(self):
    """Tuner-only pipeline for distributed Tuner flock on AIP Training."""
    pipeline_name = self._make_unique_pipeline_name('kubeflow-aip-dist-tuner')
    pipeline = self._create_pipeline(
        pipeline_name,
        [
            self.penguin_examples_importer,
            self.penguin_schema_importer,
            ai_platform_tuner_component.Tuner(
                examples=self.penguin_examples_importer.outputs['result'],
                module_file=self._penguin_tuner_module,
                schema=self.penguin_schema_importer.outputs['result'],
                train_args=trainer_pb2.TrainArgs(num_steps=10),
                eval_args=trainer_pb2.EvalArgs(num_steps=5),
                # 3 worker parallel tuning.
                tune_args=tuner_pb2.TuneArgs(num_parallel_trials=3),
                custom_config={
                    ai_platform_tuner_executor.TUNING_ARGS_KEY:
                        self._getCaipTrainingArgs(pipeline_name)
                })
        ])
    self._compile_and_run_pipeline(pipeline)
    self._assertHyperparametersAreWritten(pipeline_name)

  def _get_list_bigqueryml_models(self, api, dataset_name):
    r = api.models().list(
        projectId=self._GCP_PROJECT_ID,
        datasetId=dataset_name).execute()
    if r:
      return [m['modelReference']['modelId'] for m in r['models']]
    else:
      return []

  def testBigQueryMlPusherPipeline(self):
    """BigQuery ML Pusher pipeline on CAIP."""
    pipeline_name = self._make_unique_pipeline_name(
        'kubeflow-aip-bqml-pusher')
    # Big Query does not accept '-' in the dataset name.
    dataset_name = ('%s_model' % pipeline_name).replace('-', '_')
    self.addCleanup(_delete_bigquery_dataset,
                    dataset_name, self._GCP_PROJECT_ID)

    api = discovery.build('bigquery', 'v2')
    api.datasets().insert(
        projectId=self._GCP_PROJECT_ID,
        body={'location': 'US',
              'projectId': self._GCP_PROJECT_ID,
              'datasetReference': {'datasetId': dataset_name,
                                   'projectId': self._GCP_PROJECT_ID}
              }).execute()

    def _pusher(model_importer, model_blessing_importer, bigquery_dataset_id):
      return Pusher(
          custom_executor_spec=executor_spec.ExecutorClassSpec(
              bigquery_pusher_executor.Executor),
          model=model_importer.outputs['result'],
          model_blessing=model_blessing_importer.outputs['result'],
          custom_config={
              bigquery_pusher_executor.SERVING_ARGS_KEY: {
                  'bq_dataset_id': bigquery_dataset_id,
                  'model_name': pipeline_name,
                  'project_id': self._GCP_PROJECT_ID,
              }
          },
      )

    # The model list should be empty
    self.assertEmpty(self._get_list_bigqueryml_models(
        api, dataset_name))

    # Test creation of multiple versions under the same model_name.
    pipeline = self._create_pipeline(pipeline_name, [
        self.model_1_importer,
        self.model_blessing_1_importer,
        _pusher(self.model_1_importer, self.model_blessing_1_importer,
                dataset_name),
    ])
    self._compile_and_run_pipeline(pipeline)
    self.assertIn(
        pipeline_name, self._get_list_bigqueryml_models(
            api, dataset_name))

  def _getNumberOfVersionsForModel(self, api, project, model_name):
    resource_name = f'projects/{project}/models/{model_name}'
    res = api.projects().models().versions().list(
        parent=resource_name).execute()
    return len(res['versions'])

  def _sendDummyRequestToModel(self, api, project, model_name):
    resource_name = f'projects/{project}/models/{model_name}'
    res = api.projects().predict(
        name=resource_name,
        body={
            'instances': {
                'inputs': ''  # Just use dummy input for basic check.
            }
        }).execute()
    absl.logging.info('Response from the pushed model: %s', res)

  def testAIPlatformPusherPipeline(self):
    """Pusher-only test pipeline to AI Platform Prediction."""
    pipeline_name_base = self._make_unique_pipeline_name('kubeflow-aip-pusher')
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
              tfx.extensions.google_cloud_ai_platform.experimental
              .PUSHER_SERVING_ARGS_KEY: {
                  'model_name': model_name,
                  'project_id': self._GCP_PROJECT_ID,
              }
          },
      )

    # Use default service_name / api_version.
    service_name, api_version = runner.get_service_name_and_api_version({})
    api = discovery.build(
        service_name,
        api_version,
        requestBuilder=telemetry_utils.TFXHttpRequest,
    )

    # The model should be NotFound yet.
    with self.assertRaisesRegex(googleapiclient_errors.HttpError,
                                'HttpError 404'):
      self._sendDummyRequestToModel(api, self._GCP_PROJECT_ID, model_name)

    # Test creation of multiple versions under the same model_name.
    pipeline_name_1 = '%s-1' % pipeline_name_base
    pipeline_1 = self._create_pipeline(pipeline_name_1, [
        self.model_1_importer,
        self.model_blessing_1_importer,
        _pusher(self.model_1_importer, self.model_blessing_1_importer),
    ])
    self._compile_and_run_pipeline(pipeline_1)
    self.assertEqual(
        1,
        self._getNumberOfVersionsForModel(api, self._GCP_PROJECT_ID,
                                          model_name))
    self._sendDummyRequestToModel(api, self._GCP_PROJECT_ID, model_name)

    pipeline_name_2 = '%s-2' % pipeline_name_base
    pipeline_2 = self._create_pipeline(pipeline_name_2, [
        self.model_2_importer,
        self.model_blessing_2_importer,
        _pusher(self.model_2_importer, self.model_blessing_2_importer),
    ])
    self._compile_and_run_pipeline(pipeline_2)
    self.assertEqual(
        2,
        self._getNumberOfVersionsForModel(api, self._GCP_PROJECT_ID,
                                          model_name))
    self._sendDummyRequestToModel(api, self._GCP_PROJECT_ID, model_name)


def _delete_bigquery_dataset(dataset_name, project_id):
  """Deletes Big Query dataset with all the content."""
  api = discovery.build('bigquery', 'v2')
  try:
    api.datasets().delete(
        projectId=project_id,
        datasetId=dataset_name,
        deleteContents=True).execute()
  except googleapiclient_errors.HttpError as err:
    err_descr = err._get_reson()  # pylint: disable=protected-access
    if err.args[0].status == 404 and err_descr.startswith('Not found'):
      absl.logging.info('Dataset %s not found at project %s!',
                        dataset_name, project_id)
      pass
    else:
      raise


if __name__ == '__main__':
  absl.logging.set_verbosity(absl.logging.INFO)
  tf.test.main()
