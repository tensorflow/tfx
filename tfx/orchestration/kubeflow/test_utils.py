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
"""Common utility for testing Kubeflow-based orchestrator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import random
import re
import shutil
import string
import subprocess
import tarfile
import tempfile
import time
from typing import Any, Dict, List, Text

import absl
import docker
import tensorflow as tf

from google.cloud import storage
from ml_metadata.proto import metadata_store_pb2
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
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.orchestration.kubeflow.proto import kubeflow_pb2
from tfx.proto import evaluator_pb2
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.utils import dsl_utils


_POLLING_INTERVAL_IN_SECONDS = 10

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

# The location of test data. The input files are copied to a test-local
# location for each invocation, and cleaned up at the end of test.
_TEST_DATA_ROOT = os.environ['KFP_E2E_TEST_DATA_ROOT']

# The location of test user module
# It is retrieved from inside the container subject to testing.
_MODULE_ROOT = '/tfx-src/tfx/components/testdata/module_file'


def create_e2e_components(
    pipeline_root: Text,
    csv_input_location: Text,
    transform_module: Text,
    trainer_module: Text,
) -> List[BaseComponent]:
  """Creates components for a simple Chicago Taxi TFX pipeline for testing.

  Args:
    pipeline_root: The root of the pipeline output.
    csv_input_location: The location of the input data directory.
    transform_module: The location of the transform module file.
    trainer_module: The location of the trainer module file.

  Returns:
    A list of TFX components that constitutes an end-to-end test pipeline.
  """
  examples = dsl_utils.csv_input(csv_input_location)

  example_gen = CsvExampleGen(input=examples)
  statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
  infer_schema = SchemaGen(
      statistics=statistics_gen.outputs['statistics'],
      infer_feature_shape=False)
  validate_stats = ExampleValidator(
      statistics=statistics_gen.outputs['statistics'],
      schema=infer_schema.outputs['schema'])
  transform = Transform(
      examples=example_gen.outputs['examples'],
      schema=infer_schema.outputs['schema'],
      module_file=transform_module)
  trainer = Trainer(
      transformed_examples=transform.outputs['transformed_examples'],
      schema=infer_schema.outputs['schema'],
      transform_graph=transform.outputs['transform_graph'],
      train_args=trainer_pb2.TrainArgs(num_steps=10),
      eval_args=trainer_pb2.EvalArgs(num_steps=5),
      module_file=trainer_module,
  )
  model_analyzer = Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      feature_slicing_spec=evaluator_pb2.FeatureSlicingSpec(specs=[
          evaluator_pb2.SingleSlicingSpec(
              column_for_slicing=['trip_start_hour'])
      ]))
  model_validator = ModelValidator(
      examples=example_gen.outputs['examples'], model=trainer.outputs['model'])
  pusher = Pusher(
      model=trainer.outputs['model'],
      model_blessing=model_validator.outputs['blessing'],
      push_destination=pusher_pb2.PushDestination(
          filesystem=pusher_pb2.PushDestination.Filesystem(
              base_directory=os.path.join(pipeline_root, 'model_serving'))))

  return [
      example_gen, statistics_gen, infer_schema, validate_stats, transform,
      trainer, model_analyzer, model_validator, pusher
  ]


class _Timer(object):
  """Helper class to time operations in Kubeflow e2e tests."""

  def __init__(self, operation: Text):
    """Creates a context object to measure time taken.

    Args:
      operation: A description of the operation being measured.
    """
    self._operation = operation

  def __enter__(self):
    self._start = time.time()

  def __exit__(self, *unused_args):
    self._end = time.time()

    absl.logging.info(
        'Timing Info >> Operation: %s Elapsed time in seconds: %d' %
        (self._operation, self._end - self._start))


class BaseKubeflowTest(tf.test.TestCase):
  """Base class that defines testing harness for pipeline on KubeflowRunner."""

  @classmethod
  def setUpClass(cls):
    super(BaseKubeflowTest, cls).setUpClass()

    # Create a container image for use by test pipelines.
    base_container_image = _BASE_CONTAINER_IMAGE

    cls._container_image = '{}:{}'.format(base_container_image,
                                          cls._random_id())
    cls._build_and_push_docker_image(cls._container_image)

  @classmethod
  def tearDownClass(cls):
    super(BaseKubeflowTest, cls).tearDownClass()

    # Delete container image used in tests.
    absl.logging.info('Deleting image {}'.format(cls._container_image))
    subprocess.run(
        ['gcloud', 'container', 'images', 'delete', cls._container_image],
        check=True)

  @classmethod
  def _build_and_push_docker_image(cls, container_image: Text):
    client = docker.from_env()
    repo_base = os.environ['KFP_E2E_SRC']

    absl.logging.info('Building image {}'.format(container_image))
    with _Timer('BuildingTFXContainerImage'):
      _ = client.images.build(
          path=repo_base,
          dockerfile='tfx/tools/docker/Dockerfile',
          tag=container_image,
          buildargs={
              # Skip license gathering for tests.
              'gather_third_party_licenses': 'false',
          },
      )

    absl.logging.info('Pushing image {}'.format(container_image))
    with _Timer('PushingTFXContainerImage'):
      client.images.push(repository=container_image)

  @classmethod
  def _get_mysql_pod_name(cls):
    """Returns MySQL pod name in the cluster."""
    pod_name = subprocess.check_output([
        'kubectl',
        '-n',
        'kubeflow',
        'get',
        'pods',
        '-l',
        'app=mysql',
        '--no-headers',
        '-o',
        'custom-columns=:metadata.name',
    ]).decode('utf-8').strip('\n')
    absl.logging.info('MySQL pod name is: {}'.format(pod_name))
    return pod_name

  @classmethod
  def _get_mlmd_db_name(cls, pipeline_name: Text):
    # MySQL DB names must not contain '-' while k8s names must not contain '_'.
    # So we replace the dashes here for the DB name.
    valid_mysql_name = pipeline_name.replace('-', '_')
    # MySQL database name cannot exceed 64 characters.
    return 'mlmd_{}'.format(valid_mysql_name[-59:])

  def setUp(self):
    super(BaseKubeflowTest, self).setUp()
    self._old_cwd = os.getcwd()
    self._test_dir = tempfile.mkdtemp()
    os.chdir(self._test_dir)

    self._gcp_project_id = _GCP_PROJECT_ID
    self._gcp_region = _GCP_REGION
    self._bucket_name = _BUCKET_NAME
    self._testdata_root = _TEST_DATA_ROOT

    self._test_output_dir = 'gs://{}/test_output'.format(self._bucket_name)

    test_id = self._random_id()

    self._testdata_root = 'gs://{}/test_data/{}'.format(self._bucket_name,
                                                        test_id)
    subprocess.run(
        ['gsutil', 'cp', '-r', _TEST_DATA_ROOT, self._testdata_root],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    self._data_root = os.path.join(self._testdata_root, 'external', 'csv')
    self._transform_module = os.path.join(_MODULE_ROOT, 'transform_module.py')
    self._trainer_module = os.path.join(_MODULE_ROOT, 'trainer_module.py')

    self.addCleanup(self._delete_test_dir, test_id)

  def tearDown(self):
    super(BaseKubeflowTest, self).tearDown()
    os.chdir(self._old_cwd)
    shutil.rmtree(self._test_dir)

  @staticmethod
  def _random_id():
    """Generates a random string that is also a valid Kubernetes DNS name."""
    random.seed(datetime.datetime.now())

    choices = string.ascii_lowercase + string.digits
    return '{}-{}'.format(datetime.datetime.now().strftime('%s'),
                          ''.join([random.choice(choices) for _ in range(10)]))

  def _delete_test_dir(self, test_id: Text):
    """Deletes files for this test including the module file and data files.

    Args:
      test_id: Randomly generated id of the test.
    """
    self._delete_gcs_files('test_data/{}'.format(test_id))

  def _delete_workflow(self, workflow_name: Text):
    """Deletes the specified Argo workflow."""
    absl.logging.info('Deleting workflow {}'.format(workflow_name))
    subprocess.run(['argo', '--namespace', 'kubeflow', 'delete', workflow_name],
                   check=True)

  def _run_workflow(self,
                    workflow_file: Text,
                    workflow_name: Text,
                    parameter: Dict[Text, Text] = None):
    """Runs the specified workflow with Argo.

    Blocks until the workflow has run (successfully or not) to completion.

    Args:
      workflow_file: YAML file with Argo workflow spec for the pipeline.
      workflow_name: Name to use for the workflow.
      parameter: mapping from pipeline parameter name to its runtime value.
    """

    # TODO(ajaygopinathan): Consider using KFP cli instead.
    def _format_parameter(parameter: Dict[Text, Any]) -> List[Text]:
      """Format the pipeline parameter section of argo workflow."""
      if parameter:
        result = []
        for k, v in parameter.items():
          result.append('-p')
          result.append('%s=%s' % (k, v))
        return result
      else:
        return []

    run_command = [
        'argo',
        'submit',
        '--name',
        workflow_name,
        '--namespace',
        'kubeflow',
        '--serviceaccount',
        'pipeline-runner',
        workflow_file,
    ]
    run_command += _format_parameter(parameter)
    absl.logging.info('Launching workflow {} with parameter {}'.format(
        workflow_name, _format_parameter(parameter)))
    with _Timer('RunningPipelineToCompletion'):
      subprocess.run(run_command, check=True)
      # Wait in the loop while pipeline is running.
      status = 'Running'
      while status == 'Running':
        time.sleep(_POLLING_INTERVAL_IN_SECONDS)
        status = self._get_argo_pipeline_status(workflow_name)

  def _delete_gcs_files(self, path: Text):
    """Deletes files under specified path in the test bucket.

    Args:
      path: path(or prefix) of the file to delete.
    """
    client = storage.Client(project=self._gcp_project_id)
    bucket = client.get_bucket(self._bucket_name)
    absl.logging.info(
        'Deleting files under GCS bucket path: {}'.format(path))

    with _Timer('ListingAndDeletingFilesFromGCS'):
      blobs = bucket.list_blobs(prefix=path)
      bucket.delete_blobs(blobs)

  def _delete_pipeline_output(self, pipeline_name: Text):
    """Deletes output produced by the named pipeline.

    Args:
      pipeline_name: The name of the pipeline.
    """
    self._delete_gcs_files('test_output/{}'.format(pipeline_name))

  def _delete_pipeline_metadata(self, pipeline_name: Text):
    """Drops the database containing metadata produced by the pipeline.

    Args:
      pipeline_name: The name of the pipeline owning the database.
    """
    pod_name = self._get_mysql_pod_name()
    db_name = self._get_mlmd_db_name(pipeline_name)

    command = [
        'kubectl',
        '-n',
        'kubeflow',
        'exec',
        '-it',
        pod_name,
        '--',
        'mysql',
        '--user',
        'root',
        '--execute',
        'drop database if exists {};'.format(db_name),
    ]
    absl.logging.info('Dropping MLMD DB with name: {}'.format(db_name))

    with _Timer('DeletingMLMDDatabase'):
      subprocess.run(command, check=True)

  def _pipeline_root(self, pipeline_name: Text):
    return os.path.join(self._test_output_dir, pipeline_name)

  def _create_pipeline(self, pipeline_name: Text,
                       components: List[BaseComponent]):
    """Creates a pipeline given name and list of components."""
    return tfx_pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=self._pipeline_root(pipeline_name),
        metadata_connection_config=metadata_store_pb2.ConnectionConfig(),
        components=components,
        additional_pipeline_args={
            # Use a fixed WORKFLOW_ID (which is used as run id) for testing,
            # for the purpose of making debugging easier.
            'WORKFLOW_ID': pipeline_name,
        },
    )

  def _create_dataflow_pipeline(self, pipeline_name: Text,
                                components: List[BaseComponent]):
    """Creates a pipeline with Beam DataflowRunner."""
    pipeline = self._create_pipeline(pipeline_name, components)
    pipeline.beam_pipeline_args = [
        '--runner=DataflowRunner',
        '--experiments=shuffle_mode=auto',
        '--project=' + self._gcp_project_id,
        '--temp_location=' +
        os.path.join(self._pipeline_root(pipeline_name), 'tmp'),
        '--region=' + self._gcp_region,
    ]
    return pipeline

  def _get_kubeflow_metadata_config(
      self) -> kubeflow_pb2.KubeflowMetadataConfig:
    config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()
    return config

  def _get_argo_pipeline_status(self, pipeline_name: Text) -> Text:
    """Get Pipeline status.

    Args:
      pipeline_name: The name of the pipeline.

    Returns:
      Simple status string which is returned from `argo get` command.
    """
    get_workflow_command = [
        'argo', '--namespace', 'kubeflow', 'get', pipeline_name
    ]
    output = subprocess.check_output(get_workflow_command).decode('utf-8')
    absl.logging.info('Argo output ----\n%s', output)
    match = re.search(r'^Status:\s+(.+)$', output, flags=re.MULTILINE)
    self.assertIsNotNone(match)
    return match.group(1)

  def _compile_and_run_pipeline(self,
                                pipeline: tfx_pipeline.Pipeline,
                                parameters: Dict[Text, Any] = None):
    """Compiles and runs a KFP pipeline.

    Args:
      pipeline: The logical pipeline to run.
      parameters: Value of runtime paramters of the pipeline.
    """
    pipeline_name = pipeline.pipeline_info.pipeline_name
    config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
        kubeflow_metadata_config=self._get_kubeflow_metadata_config(),
        tfx_image=self._container_image)
    kubeflow_dag_runner.KubeflowDagRunner(config=config).run(pipeline)

    file_path = os.path.join(self._test_dir, '{}.tar.gz'.format(pipeline_name))
    self.assertTrue(tf.io.gfile.exists(file_path))
    tarfile.TarFile.open(file_path).extract('pipeline.yaml')
    pipeline_file = os.path.join(self._test_dir, 'pipeline.yaml')
    self.assertIsNotNone(pipeline_file)

    # Ensure cleanup regardless of whether pipeline succeeds or fails.
    self.addCleanup(self._delete_workflow, pipeline_name)
    self.addCleanup(self._delete_pipeline_output, pipeline_name)
    self.addCleanup(self._delete_pipeline_metadata, pipeline_name)

    # Run the pipeline to completion.
    self._run_workflow(pipeline_file, pipeline_name, parameters)

    # Obtain workflow logs.
    get_logs_command = [
        'argo', '--namespace', 'kubeflow', 'logs', '-w', pipeline_name
    ]
    logs_output = subprocess.check_output(get_logs_command).decode('utf-8')

    # Check if pipeline completed successfully.
    status = self._get_argo_pipeline_status(pipeline_name)
    self.assertEqual(
        'Succeeded', status, 'Pipeline {} failed to complete successfully: {}'
        '\nFailed workflow logs:\n{}'.format(pipeline_name, status,
                                             logs_output))
