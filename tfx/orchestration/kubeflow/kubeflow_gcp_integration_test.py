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
import time

import tensorflow as tf

from typing import List, Text

from ml_metadata.proto import metadata_store_pb2
from tfx.components.base import executor_spec
from tfx.components.evaluator.component import Evaluator
from tfx.components.example_gen.csv_example_gen.component import CsvExampleGen
from tfx.components.model_validator.component import ModelValidator
from tfx.components.pusher.component import Pusher
from tfx.components.statistics_gen.component import StatisticsGen
from tfx.components.trainer.component import Trainer
from tfx.components.transform.component import Transform
from tfx.extensions.google_cloud_ai_platform.pusher import executor as ai_platform_pusher_executor
from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration.kubeflow import test_utils
from tfx.proto import evaluator_pb2
from tfx.proto import trainer_pb2
from tfx.types import Artifact
from tfx.types import Channel
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.utils import dsl_utils


class KubeflowGCPIntegrationTest(test_utils.BaseKubeflowTest):

  @classmethod
  def setUpClass(cls):
    super(KubeflowGCPIntegrationTest, cls).setUpClass()

    cls._mysql_portward_process = cls._setup_mysql_port_forward()

  @classmethod
  def tearDownClass(cls):
    super(KubeflowGCPIntegrationTest, cls).tearDownClass()

    cls._mysql_portward_process.kill()

  @classmethod
  def _setup_mysql_port_forward(cls):
    """Establishes port forward to MLMD database in the cluster."""
    pod_name = cls._get_mysql_pod_name()
    mysql_portforward_command = [
        'kubectl', '-n', 'kubeflow', 'port-forward', pod_name, '3306:3306'
    ]
    proc = subprocess.Popen(mysql_portforward_command, stdout=subprocess.PIPE)

    # Wait while port forward to cluster is being established.
    poll_mysql_port_command = ['lsof', '-i', ':3306']
    result = subprocess.run(poll_mysql_port_command)
    timeout = 10
    for _ in range(timeout):
      if result.returncode == 0:
        break
      tf.logging.info('Waiting while MySQL port-forward is established...')
      time.sleep(1)

      result = subprocess.run(poll_mysql_port_command)

    if result.returncode != 0:
      raise RuntimeError('Failed to establish MySQL port-forward to cluster.')

    return proc

  def _input_artifacts(self, pipeline_name: Text,
                       input_artifacts: List[Artifact]) -> Channel:
    """Publish input artifacts for test to MLMD and return channel to them."""
    connection_config = metadata_store_pb2.ConnectionConfig(
        mysql=metadata_store_pb2.MySQLDatabaseConfig(
            host='127.0.0.1',
            port=3306,
            database=self._get_mlmd_db_name(pipeline_name),
            user='root',
            password=''))

    dummy_artifact = (input_artifacts[0].type_name, self._random_id())
    output_key = 'dummy_output_%s_%s' % dummy_artifact
    producer_component_id = 'dummy_producer_id_%s_%s' % dummy_artifact
    producer_component_type = 'dummy_producer_type_%s_%s' % dummy_artifact

    # Input artifacts must have a unique name and producer in MLMD.
    for artifact in input_artifacts:
      artifact.name = output_key
      artifact.pipeline_name = pipeline_name
      artifact.producer_component = producer_component_id

    with metadata.Metadata(connection_config=connection_config) as m:
      # Register a dummy execution to metadata store as producer execution.
      execution_id = m.register_execution(
          exec_properties={},
          pipeline_info=data_types.PipelineInfo(
              pipeline_name=pipeline_name,
              pipeline_root='/dummy_pipeline_root',
              # test_utils uses pipeline_name as fixed WORKFLOW_ID.
              run_id=pipeline_name,
          ),
          component_info=data_types.ComponentInfo(
              component_type=producer_component_type,
              component_id=producer_component_id))

      # Publish the test input artifact from the dummy execution.
      published_artifacts = m.publish_execution(
          execution_id=execution_id,
          input_dict={},
          output_dict={output_key: input_artifacts})

    return channel_utils.as_channel(published_artifacts[output_key])

  def _delete_ai_platform_model(self, model_name):
    """Delete pushed model in AI Platform."""
    # In order to delete model, all versions in the model must be deleted first.
    versions_command = [
        'gcloud', 'ai-platform', 'versions', 'list',
        '--model=%s' % model_name
    ]
    versions = subprocess.run(versions_command, stdout=subprocess.PIPE)

    if versions.returncode == 0:
      tf.logging.info('Model %s has versions %s' % (model_name, versions))

      # First line of the output is the header: [NAME] [DEPLOYMENT_URI] [STATE]
      # Second line is the 'default' version, which needs to be deleted last,
      # as such, need to flip the order of result lines.
      for version in versions.stdout.decode('utf-8').strip('\n').split(
          '\n')[1:][::-1]:
        version = version.split()[0]
        tf.logging.info('Deleting version %s of model %s' %
                        (version, model_name))
        version_delete_command = [
            'gcloud', '--quiet', 'ai-platform', 'versions', 'delete', version,
            '--model=%s' % model_name
        ]
        subprocess.run(version_delete_command, check=True)

    tf.logging.info('Deleting model %s' % model_name)
    subprocess.run(
        ['gcloud', '--quiet', 'ai-platform', 'models', 'delete', model_name],
        check=True)

  def setUp(self):
    super(KubeflowGCPIntegrationTest, self).setUp()

    # Raw Example artifacts for testing.
    raw_train_examples = standard_artifacts.Examples(split='train')
    raw_train_examples.uri = os.path.join(
        self._intermediate_data_root,
        'csv_example_gen/examples/test-pipeline/train/')
    raw_eval_examples = standard_artifacts.Examples(split='eval')
    raw_eval_examples.uri = os.path.join(
        self._intermediate_data_root,
        'csv_example_gen/examples/test-pipeline/eval/')
    self._test_raw_examples = [raw_train_examples, raw_eval_examples]

    # Transformed Example artifacts for testing.
    transformed_train_examples = standard_artifacts.Examples(split='train')
    transformed_train_examples.uri = os.path.join(
        self._intermediate_data_root,
        'transform/transformed_examples/test-pipeline/train/')
    transformed_eval_examples = standard_artifacts.Examples(split='eval')
    transformed_eval_examples.uri = os.path.join(
        self._intermediate_data_root,
        'transform/transformed_examples/test-pipeline/eval/')
    self._test_transformed_examples = [
        transformed_train_examples, transformed_eval_examples
    ]

    # Schema artifact for testing.
    schema = standard_artifacts.Schema()
    schema.uri = os.path.join(self._intermediate_data_root,
                              'schema_gen/output/test-pipeline/')
    self._test_schema = [schema]

    # TransformGraph artifact for testing.
    transform_graph = standard_artifacts.TransformGraph()
    transform_graph.uri = os.path.join(
        self._intermediate_data_root,
        'transform/transform_output/test-pipeline/')
    self._test_transform_graph = [transform_graph]

    # Model artifact for testing.
    model_1 = standard_artifacts.Model()
    model_1.uri = os.path.join(self._intermediate_data_root,
                               'trainer/output/test-pipeline/1/')
    self._test_model_1 = [model_1]

    model_2 = standard_artifacts.Model()
    model_2.uri = os.path.join(self._intermediate_data_root,
                               'trainer/output/test-pipeline/2/')
    self._test_model_2 = [model_2]

    # ModelBlessing artifact for testing.
    model_blessing = standard_artifacts.ModelBlessing()
    model_blessing.uri = os.path.join(
        self._intermediate_data_root, 'model_validator/blessing/test-pipeline/')
    self._test_model_blessing = [model_blessing]

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
        StatisticsGen(
            examples=self._input_artifacts(pipeline_name,
                                           self._test_raw_examples)),
    ])
    self._compile_and_run_pipeline(pipeline)

  def testTransformOnDataflowRunner(self):
    """Transform-only test pipeline on DataflowRunner."""
    pipeline_name = 'kubeflow-transform-dataflow-test-{}'.format(
        self._random_id())
    pipeline = self._create_dataflow_pipeline(pipeline_name, [
        Transform(
            examples=self._input_artifacts(pipeline_name,
                                           self._test_raw_examples),
            schema=self._input_artifacts(pipeline_name, self._test_schema),
            module_file=self._taxi_module_file)
    ])
    self._compile_and_run_pipeline(pipeline)

  def testEvaluatorOnDataflowRunner(self):
    """Evaluator-only test pipeline on DataflowRunner."""
    pipeline_name = 'kubeflow-evaluator-dataflow-test-{}'.format(
        self._random_id())
    pipeline = self._create_dataflow_pipeline(pipeline_name, [
        Evaluator(
            examples=self._input_artifacts(pipeline_name,
                                           self._test_raw_examples),
            model_exports=self._input_artifacts(pipeline_name,
                                                self._test_model_1),
            feature_slicing_spec=evaluator_pb2.FeatureSlicingSpec(specs=[
                evaluator_pb2.SingleSlicingSpec(
                    column_for_slicing=['trip_start_hour'])
            ]))
    ])
    self._compile_and_run_pipeline(pipeline)

  def testModelValidatorOnDataflowRunner(self):
    """ModelValidator-only test pipeline on DataflowRunner."""
    pipeline_name = 'kubeflow-evaluator-dataflow-test-{}'.format(
        self._random_id())
    pipeline = self._create_dataflow_pipeline(pipeline_name, [
        ModelValidator(
            examples=self._input_artifacts(pipeline_name,
                                           self._test_raw_examples),
            model=self._input_artifacts(pipeline_name, self._test_model_1))
    ])
    self._compile_and_run_pipeline(pipeline)

  def testAIPlatformTrainerPipeline(self):
    """Trainer-only test pipeline on AI Platform Training."""
    pipeline_name = 'kubeflow-aip-trainer-test-{}'.format(self._random_id())
    pipeline = self._create_pipeline(pipeline_name, [
        Trainer(
            custom_executor_spec=executor_spec.ExecutorClassSpec(
                ai_platform_trainer_executor.Executor),
            module_file=self._taxi_module_file,
            transformed_examples=self._input_artifacts(
                pipeline_name, self._test_transformed_examples),
            schema=self._input_artifacts(pipeline_name, self._test_schema),
            transform_graph=self._input_artifacts(pipeline_name,
                                                  self._test_transform_graph),
            train_args=trainer_pb2.TrainArgs(num_steps=10000),
            eval_args=trainer_pb2.EvalArgs(num_steps=5000),
            custom_config={
                'ai_platform_training_args': {
                    'project':
                        self._gcp_project_id,
                    'region':
                        self._gcp_region,
                    'jobDir':
                        os.path.join(self._pipeline_root(pipeline_name), 'tmp'),
                    'masterConfig': {
                        'imageUri': self._container_image,
                    }
                }
            }),
    ])
    self._compile_and_run_pipeline(pipeline)

  def testAIPlatformPusherPipeline(self):
    """Pusher-only test pipeline to AI Platform Prediction."""
    pipeline_name = 'kubeflow-aip-pusher-test-{}'.format(self._random_id())
    # AI Platform does not accept '-' in the model name.
    model_name = ('%s_model' % pipeline_name).replace('-', '_')

    def _pusher(model, name):
      return Pusher(
          custom_executor_spec=executor_spec.ExecutorClassSpec(
              ai_platform_pusher_executor.Executor),
          model=self._input_artifacts(pipeline_name, model),
          model_blessing=self._input_artifacts(pipeline_name,
                                               self._test_model_blessing),
          custom_config={
              'ai_platform_serving_args': {
                  'model_name': model_name,
                  'project_id': self._gcp_project_id,
              }
          },
          instance_name=name,
      )

    pipeline = self._create_pipeline(pipeline_name, [
        # Test creation of multiple versions under the same model_name.
        _pusher(self._test_model_1, 'model_1'),
        _pusher(self._test_model_2, 'model_2'),
    ])
    self.addCleanup(self._delete_ai_platform_model, model_name)
    self._compile_and_run_pipeline(pipeline)


if __name__ == '__main__':
  logging.basicConfig(stream=sys.stdout, level=logging.INFO)
  tf.test.main()
