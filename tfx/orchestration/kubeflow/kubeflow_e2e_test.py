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
"""End to end tests for Kubeflow-based orchestrator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import logging
import os
import random
import re
import shutil
import string
import subprocess
import sys
import tarfile
import tempfile

import docker
import tensorflow as tf
from typing import List, Text

from google.cloud import storage
from tfx.components.base.base_component import BaseComponent
from tfx.components.evaluator.component import Evaluator
from tfx.components.example_gen.csv_example_gen.component import CsvExampleGen
from tfx.components.example_validator.component import ExampleValidator
from tfx.components.model_validator.component import ModelValidator
from tfx.components.pusher.component import Pusher
from tfx.components.schema_gen.component import SchemaGen
from tfx.components.statistics_gen.component import StatisticsGen
from tfx.components.trainer.component import Trainer
from tfx.components.transform.component import Transform
from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration.kubeflow.runner import KubeflowRunner
from tfx.proto import evaluator_pb2
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.utils import dsl_utils

# The following environment variables need to be set prior to calling the test
# in this file. All variables are required and do not have a default.

# The base container image name to use when building the image used in tests.
_BASE_CONTAINER_IMAGE = os.environ['KFP_E2E_BASE_CONTAINER_IMAGE']

# The project id to use to run tests.
_GCP_PROJECT_ID = os.environ['KFP_E2E_GCP_PROJECT_ID']

# The GCP region in which the end-to-end test is run.
_GCP_REGION = os.environ['KFP_E2E_GCP_REGION']

# The GCP bucket to use to write output artifacts.
_BUCKET_NAME = os.environ['KFP_E2E_BUCKET_NAME']

# The input data root location on GCS. The input files are never modified and
# are safe for concurrent reads.
_DATA_ROOT = os.environ['KFP_E2E_DATA_ROOT']

# The intermediate data root location on GCS. The intermediate test data files
# are never modified and are safe for concurrent reads.
_INTERMEDIATE_DATA_ROOT = os.environ['KFP_E2E_INTERMEDIATE_DATA_ROOT']

# Location of the input taxi module file to be used in the test pipeline.
_TAXI_MODULE_FILE = os.environ['KFP_E2E_TAXI_MODULE_FILE']


def _create_e2e_components(pipeline_root: Text, csv_input_location: Text,
                           taxi_module_file: Text) -> List[BaseComponent]:
  """Creates components for a simple Chicago Taxi TFX pipeline for testing.

  Args:
    pipeline_root: The root of the pipeline output.
    csv_input_location: The location of the input data directory.
    taxi_module_file: The location of the module file for Transform/Trainer.

  Returns:
    A list of TFX components that constitutes an end-to-end test pipeline.
  """
  examples = dsl_utils.csv_input(csv_input_location)

  example_gen = CsvExampleGen(input_base=examples)
  statistics_gen = StatisticsGen(input_data=example_gen.outputs.examples)
  infer_schema = SchemaGen(stats=statistics_gen.outputs.output)
  validate_stats = ExampleValidator(
      stats=statistics_gen.outputs.output, schema=infer_schema.outputs.output)
  transform = Transform(
      input_data=example_gen.outputs.examples,
      schema=infer_schema.outputs.output,
      module_file=taxi_module_file)
  trainer = Trainer(
      module_file=taxi_module_file,
      transformed_examples=transform.outputs.transformed_examples,
      schema=infer_schema.outputs.output,
      transform_output=transform.outputs.transform_output,
      train_args=trainer_pb2.TrainArgs(num_steps=10000),
      eval_args=trainer_pb2.EvalArgs(num_steps=5000))
  model_analyzer = Evaluator(
      examples=example_gen.outputs.examples,
      model_exports=trainer.outputs.output,
      feature_slicing_spec=evaluator_pb2.FeatureSlicingSpec(specs=[
          evaluator_pb2.SingleSlicingSpec(
              column_for_slicing=['trip_start_hour'])
      ]))
  model_validator = ModelValidator(
      examples=example_gen.outputs.examples, model=trainer.outputs.output)
  pusher = Pusher(
      model_export=trainer.outputs.output,
      model_blessing=model_validator.outputs.blessing,
      push_destination=pusher_pb2.PushDestination(
          filesystem=pusher_pb2.PushDestination.Filesystem(
              base_directory=os.path.join(pipeline_root, 'model_serving'))))

  return [
      example_gen, statistics_gen, infer_schema, validate_stats, transform,
      trainer, model_analyzer, model_validator, pusher
  ]


class KubeflowEndToEndTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    super(KubeflowEndToEndTest, cls).setUpClass()

    # Create a container image for use by test pipelines.
    base_container_image = _BASE_CONTAINER_IMAGE

    cls._container_image = '{}:{}'.format(base_container_image,
                                          cls._random_id())
    cls._build_and_push_docker_image(cls._container_image)

  @classmethod
  def tearDownClass(cls):
    super(KubeflowEndToEndTest, cls).tearDownClass()

    # Delete container image used in tests.
    tf.logging.info('Deleting image {}'.format(cls._container_image))
    subprocess.run(
        ['gcloud', 'container', 'images', 'delete', cls._container_image],
        check=True)

  @classmethod
  def _build_and_push_docker_image(cls, container_image: Text):
    client = docker.from_env()
    repo_base = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    tf.logging.info('Building image {}'.format(container_image))
    _ = client.images.build(
        path=repo_base,
        dockerfile='tfx/tools/docker/Dockerfile',
        tag=container_image,
        buildargs={
            # Skip license gathering for tests.
            'gather_third_party_licenses': 'false',
        },
    )
    tf.logging.info('Pushing image {}'.format(container_image))
    client.images.push(repository=container_image)

  def setUp(self):
    super(KubeflowEndToEndTest, self).setUp()
    self._test_dir = tempfile.mkdtemp()
    os.chdir(self._test_dir)

    self._gcp_project_id = _GCP_PROJECT_ID
    self._gcp_region = _GCP_REGION
    self._bucket_name = _BUCKET_NAME
    self._data_root = _DATA_ROOT
    self._intermediate_data_root = _INTERMEDIATE_DATA_ROOT
    self._taxi_module_file = _TAXI_MODULE_FILE

    self._test_output_dir = 'gs://{}/test_output'.format(self._bucket_name)

    # Channel of mock Example artifacts for testing.
    train_examples = standard_artifacts.Examples(split='train')
    train_examples.uri = os.path.join(
        self._intermediate_data_root,
        'transform/transformed_examples/test-pipeline/train/')
    eval_examples = standard_artifacts.Examples(split='eval')
    eval_examples.uri = os.path.join(
        self._intermediate_data_root,
        'transform/transformed_examples/test-pipeline/eval/')
    self._mock_examples = channel_utils.as_channel(
        [train_examples, eval_examples])

    # Channel of mock Schema artifact for testing.
    schema = standard_artifacts.Schema()
    schema.uri = os.path.join(self._intermediate_data_root,
                              'schema_gen/output/test-pipeline/')
    self._mock_schema = channel_utils.as_channel([schema])

    # Channel of mock TransformGraph artifact for testing.
    transform_graph = standard_artifacts.TransformGraph()
    transform_graph.uri = os.path.join(
        self._intermediate_data_root,
        'transform/test-pipeline/transform_output/')
    self._mock_transform_graph = channel_utils.as_channel([transform_graph])

    # Channel of mock Model artifact for testing.
    model = standard_artifacts.Model()
    model.uri = os.path.join(self._intermediate_data_root,
                             'trainer/output/test-pipeline/')
    self._mock_model = channel_utils.as_channel([model])

    # Channel of mock ModelBlessing artifact for testing.
    model_blessing = standard_artifacts.ModelBlessing()
    model_blessing.uri = os.path.join(
        self._intermediate_data_root, 'model_validator/blessing/test-pipeline/')
    self._mock_model_blessing = channel_utils.as_channel([model_blessing])

  def tearDown(self):
    super(KubeflowEndToEndTest, self).tearDown()
    shutil.rmtree(self._test_dir)

  @staticmethod
  def _random_id():
    """Generates a random string that is also a valid Kubernetes DNS name."""
    choices = string.ascii_lowercase + string.digits
    result = ''.join([random.choice(choices) for _ in range(10)])
    result = result + '-{}'.format(
        datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    return result

  def _delete_workflow(self, workflow_name: Text):
    """Deletes the specified Argo workflow."""
    tf.logging.info('Deleting workflow {}'.format(workflow_name))
    subprocess.run(['argo', '--namespace', 'kubeflow', 'delete', workflow_name],
                   check=True)

  def _run_workflow(self, workflow_file: Text, workflow_name: Text):
    """Runs the specified workflow with Argo.

    Blocks until the workflow has run (successfully or not) to completion.

    Args:
      workflow_file: YAML file with Argo workflow spec for the pipeline.
      workflow_name: Name to use for the workflow.
    """
    # TODO(ajaygopinathan): Consider using KFP cli instead.
    run_command = [
        'argo',
        'submit',
        '--name',
        workflow_name,
        '--watch',
        '--namespace',
        'kubeflow',
        '--serviceaccount',
        'pipeline-runner',
        workflow_file,
    ]
    tf.logging.info('Launching workflow {}'.format(workflow_name))
    subprocess.run(run_command, check=True)

  def _delete_pipeline_output(self, pipeline_name: Text):
    """Deletes output produced by the named pipeline.

    Args:
      pipeline_name: The name of the pipeline.
    """
    client = storage.Client(project=self._gcp_project_id)
    bucket = client.get_bucket(self._bucket_name)
    prefix = 'test_output/{}'.format(pipeline_name)
    tf.logging.info(
        'Deleting output under GCS bucket prefix: {}'.format(prefix))
    blobs = bucket.list_blobs(prefix=prefix)
    bucket.delete_blobs(blobs)

  def _pipeline_root(self, pipeline_name: Text):
    return os.path.join(self._test_output_dir, pipeline_name)

  def _create_pipeline(self, pipeline_name: Text,
                       components: List[BaseComponent]):
    """Creates a pipeline given name and list of components."""
    return tfx_pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=self._pipeline_root(pipeline_name),
        components=components,
        log_root='/var/tmp/tfx/logs',
        additional_pipeline_args={
            'tfx_image': self._container_image,
        },
    )

  def _create_dataflow_pipeline(self, pipeline_name: Text,
                                components: List[BaseComponent]):
    """Creates a pipeline with Beam DataflowRunner."""
    pipeline = self._create_pipeline(pipeline_name, components)
    pipeline.additional_pipeline_args['beam_pipeline_args'] = [
        '--runner=DataflowRunner',
        '--experiments=shuffle_mode=auto',
        '--project=' + self._gcp_project_id,
        '--temp_location=' +
        os.path.join(self._pipeline_root(pipeline_name), 'tmp'),
        '--region=' + self._gcp_region,
    ]
    return pipeline

  def _compile_and_run_pipeline(self, pipeline: tfx_pipeline.Pipeline):
    """Compiles and runs a KFP pipeline.

    Args:
      pipeline: The logical pipeline to run.
    """
    _ = KubeflowRunner().run(pipeline)
    pipeline_name = pipeline.pipeline_info.pipeline_name

    file_path = os.path.join(self._test_dir, '{}.tar.gz'.format(pipeline_name))
    self.assertTrue(tf.gfile.Exists(file_path))
    tarfile.TarFile.open(file_path).extract('pipeline.yaml')
    pipeline_file = os.path.join(self._test_dir, 'pipeline.yaml')
    self.assertIsNotNone(pipeline_file)

    # Ensure cleanup regardless of whether pipeline succeeds or fails.
    self.addCleanup(self._delete_workflow, pipeline_name)
    self.addCleanup(self._delete_pipeline_output, pipeline_name)

    # Run the pipeline to completion.
    self._run_workflow(pipeline_file, pipeline_name)

    # Check if pipeline completed successfully.
    get_workflow_command = [
        'argo', '--namespace', 'kubeflow', 'get', pipeline_name
    ]
    output = subprocess.check_output(get_workflow_command).decode('utf-8')

    self.assertIsNotNone(
        re.search(r'^Status:\s+Succeeded$', output, flags=re.MULTILINE),
        'Pipeline {} failed to complete successfully:\n{}'.format(
            pipeline_name, output))

#  def testSimpleEnd2EndPipeline(self):
#    """End-to-End test for simple pipeline."""
#    pipeline_name = 'kubeflow-e2e-test-{}'.format(self._random_id())
#    components = _create_e2e_components(
#        self._pipeline_root(pipeline_name), self._data_root,
#        self._taxi_module_file)
#    pipeline = self._create_pipeline(pipeline_name, components)
#
#    self._compile_and_run_pipeline(pipeline)

  def testDataflowRunner(self):
    """Test for DataflowRunner invocation."""
    pipeline_name = 'kubeflow-dataflow-test-{}'.format(self._random_id())

    pipeline = self._create_dataflow_pipeline(pipeline_name, [
        CsvExampleGen(input_base=dsl_utils.csv_input(self._data_root)),
        StatisticsGen(input_data=self._mock_examples),
        Transform(
            input_data=self._mock_examples,
            schema=self._mock_schema,
            module_file=self._taxi_module_file),
        Evaluator(
            examples=self._mock_examples,
            model_exports=self._mock_model,
            feature_slicing_spec=evaluator_pb2.FeatureSlicingSpec(specs=[
                evaluator_pb2.SingleSlicingSpec(
                    column_for_slicing=['trip_start_hour'])
            ])),
        ModelValidator(examples=self._mock_examples, model=self._mock_model),
    ])

    self._compile_and_run_pipeline(pipeline)


#  def testExampleGenOnDataflowRunner(self):
#    """ExampleGen-only test pipeline on DataflowRunner."""
#    pipeline_name = 'kubeflow-example-gen-dataflow-test-{}'.format(
#        self._random_id())
#    pipeline = self._create_dataflow_pipeline(
#        pipeline_name,
#        [CsvExampleGen(input_base=dsl_utils.csv_input(self._data_root))])
#    self._compile_and_run_pipeline(pipeline)
#
#  def testStatisticsGenOnDataflowRunner(self):
#    """StatisticsGen-only test pipeline on DataflowRunner."""
#    pipeline_name = 'kubeflow-statistics-gen-dataflow-test-{}'.format(
#        self._random_id())
#    pipeline = self._create_dataflow_pipeline(
#        pipeline_name, [StatisticsGen(input_data=self._mock_examples)])
#    self._compile_and_run_pipeline(pipeline)
#
#  def testTransformOnDataflowRunner(self):
#    """Transform-only test pipeline on DataflowRunner."""
#    pipeline_name = 'kubeflow-transform-dataflow-test-{}'.format(
#        self._random_id())
#    pipeline = self._create_dataflow_pipeline(pipeline_name, [
#        Transform(
#            input_data=self._mock_examples,
#            schema=self._mock_schema,
#            module_file=self._taxi_module_file)
#    ])
#    self._compile_and_run_pipeline(pipeline)
#
#  def testEvaluatorOnDataflowRunner(self):
#    """Evaluator-only test pipeline on DataflowRunner."""
#    pipeline_name = 'kubeflow-evaluator-dataflow-test-{}'.format(
#        self._random_id())
#    pipeline = self._create_dataflow_pipeline(pipeline_name, [
#        Evaluator(
#            examples=self._mock_examples,
#            model_exports=self._mock_model,
#            feature_slicing_spec=evaluator_pb2.FeatureSlicingSpec(specs=[
#                evaluator_pb2.SingleSlicingSpec(
#                    column_for_slicing=['trip_start_hour'])
#            ]))
#    ])
#    self._compile_and_run_pipeline(pipeline)
#
#  def testModelValidatorOnDataflowRunner(self):
#    """ModelValidator-only test pipeline on DataflowRunner."""
#    pipeline_name = 'kubeflow-evaluator-dataflow-test-{}'.format(
#        self._random_id())
#    pipeline = self._create_dataflow_pipeline(
#        pipeline_name,
#        [ModelValidator(examples=self._mock_examples, model=self._mock_model)])
#    self._compile_and_run_pipeline(pipeline)

#  def testAIPTrainerPipeline(self):
#    """Trainer-only test pipeline on AIP Training."""
#    pipeline_name = 'kubeflow-aip-trainer-test-{}'.format(self._random_id())
#    pipeline = self._create_pipeline(pipeline_name, [
#        Trainer(
#            executor_class=ai_platform_trainer_executor.Executor,
#            module_file=self._taxi_module_file,
#            transformed_examples=self._mock_examples,
#            schema=self._mock_schema,
#            transform_output=self._mock_transform_graph,
#            train_args=trainer_pb2.TrainArgs(num_steps=10000),
#            eval_args=trainer_pb2.EvalArgs(num_steps=5000),
#            custom_config={
#                'ai_platform_training_args': {
#                    'project':
#                        self._gcp_project_id,
#                    'region':
#                        self._gcp_region,
#                    'jobDir':
#                        os.path.join(self._pipeline_root(pipeline_name),
#                                     'tmp'),
#                }
#            })
#    ])
#
#    self._compile_and_run_pipeline(pipeline)
#
#  def testAIPPusherPipeline(self):
#    """Pusher-only test pipeline on AIP Prediction."""
#    pipeline_name = 'kubeflow-aip-pusher-test-{}'.format(self._random_id())
#    model_name = '%s_model' % pipeline_name
#    pipeline = self._create_pipeline(pipeline_name, [
#        Pusher(
#            executor_class=ai_platform_pusher_executor.Executor,
#            model_export=self._mock_model,
#            model_blessing=self._mock_model_blessing,
#            custom_config={
#                'ai_platform_serving_args': {
#                    'model_name': model_name,
#                    'project_id': self._gcp_project_id,
#                }
#            })
#    ])
#
#    self._compile_and_run_pipeline(pipeline)
#
#    # TODO(muchida): Implement cleanup.


if __name__ == '__main__':
  logging.basicConfig(stream=sys.stdout, level=logging.INFO)
  tf.test.main()
