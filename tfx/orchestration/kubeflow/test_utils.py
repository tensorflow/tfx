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
import json
import os
import re
import subprocess
import tarfile
import time
from typing import Any, Dict, List, Text

from absl import logging
import kfp
from kfp_server_api import rest
import tensorflow_model_analysis as tfma

from tfx.components import CsvExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import InfraValidator
from tfx.components import Pusher
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.dsl.components.base import executor_spec
from tfx.dsl.components.base.base_component import BaseComponent
from tfx.dsl.components.common import resolver
from tfx.dsl.input_resolution.strategies import latest_artifact_strategy
from tfx.dsl.io import fileio
from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration import test_utils
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.orchestration.kubeflow.proto import kubeflow_pb2
from tfx.proto import infra_validator_pb2
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.types import Channel
from tfx.types import channel_utils
from tfx.types import component_spec
from tfx.types import standard_artifacts
from tfx.types.standard_artifacts import Model
from tfx.utils import kube_utils
from tfx.utils import retry
from tfx.utils import test_case_utils


# TODO(jiyongjung): Merge with kube_utils.PodStatus
# Various execution status of a KFP pipeline.
KFP_RUNNING_STATUS = 'running'
KFP_SUCCESS_STATUS = 'succeeded'
KFP_FAIL_STATUS = 'failed'
KFP_SKIPPED_STATUS = 'skipped'
KFP_ERROR_STATUS = 'error'

KFP_FINAL_STATUS = frozenset(
    (KFP_SUCCESS_STATUS, KFP_FAIL_STATUS, KFP_SKIPPED_STATUS, KFP_ERROR_STATUS))


def poll_kfp_with_retry(host: Text, run_id: Text, retry_limit: int,
                        timeout: datetime.timedelta,
                        polling_interval: int) -> Text:
  """Gets the pipeline execution status by polling KFP at the specified host.

  Args:
    host: address of the KFP deployment.
    run_id: id of the execution of the pipeline.
    retry_limit: number of retries that will be performed before raise an error.
    timeout: timeout of this long-running operation, in timedelta.
    polling_interval: interval between two consecutive polls, in seconds.

  Returns:
    The final status of the execution. Possible value can be found at
    https://github.com/kubeflow/pipelines/blob/master/backend/api/run.proto#L254

  Raises:
    RuntimeError: if polling failed for retry_limit times consecutively.
  """

  start_time = datetime.datetime.now()
  retry_count = 0
  while True:
    # TODO(jxzheng): workaround for 1hr timeout limit in kfp.Client().
    # This should be changed after
    # https://github.com/kubeflow/pipelines/issues/3630 is fixed.
    # Currently gcloud authentication token has a 1-hour expiration by default
    # but kfp.Client() does not have a refreshing mechanism in place. This
    # causes failure when attempting to get running status for a long pipeline
    # execution (> 1 hour).
    # Instead of implementing a whole authentication refreshing mechanism
    # here, we chose re-creating kfp.Client() frequently to make sure the
    # authentication does not expire. This is based on the fact that
    # kfp.Client() is very light-weight.
    # See more details at
    # https://github.com/kubeflow/pipelines/issues/3630
    client = kfp.Client(host=host)
    # TODO(b/156784019): workaround the known issue at b/156784019 and
    # https://github.com/kubeflow/pipelines/issues/3669
    # by wait-and-retry when ApiException is hit.
    try:
      get_run_response = client.get_run(run_id=run_id)
    except rest.ApiException as api_err:
      # If get_run failed with ApiException, wait _POLLING_INTERVAL and retry.
      if retry_count < retry_limit:
        retry_count += 1
        logging.info('API error %s was hit. Retrying: %s / %s.', api_err,
                     retry_count, retry_limit)
        time.sleep(polling_interval)
        continue

      raise RuntimeError('Still hit remote error after %s retries: %s' %
                         (retry_limit, api_err))
    else:
      # If get_run succeeded, reset retry_count.
      retry_count = 0

    if (get_run_response and get_run_response.run and
        get_run_response.run.status and
        get_run_response.run.status.lower() in KFP_FINAL_STATUS):
      # Return because final status is reached.
      return get_run_response.run.status

    if datetime.datetime.now() - start_time > timeout:
      # Timeout.
      raise RuntimeError('Waiting for run timeout at %s' %
                         datetime.datetime.now().strftime('%H:%M:%S'))

    logging.info('Waiting for the job to complete...')
    time.sleep(polling_interval)


def print_failure_log_for_run(host: Text, run_id: Text, namespace: Text):
  """Prints logs of failed components of a run.

  Prints execution logs for failed componentsusing `logging.info`.
  This resembles the behavior of `argo logs` but uses K8s API directly.
  Don't print anything if the run was successful.

  Args:
    host: address of the KFP deployment.
    run_id: id of the execution of the pipeline.
    namespace: namespace of K8s cluster.
  """
  client = kfp.Client(host=host)
  run = client.get_run(run_id=run_id)
  workflow_manifest = json.loads(run.pipeline_runtime.workflow_manifest)
  if kube_utils.PodPhase(
      workflow_manifest['status']['phase']) != kube_utils.PodPhase.FAILED:
    return

  k8s_client = kube_utils.make_core_v1_api()
  pods = [i for i in workflow_manifest['status']['nodes'] if i['type'] == 'Pod']
  for pod in pods:
    if kube_utils.PodPhase(pod['phase']) != kube_utils.PodPhase.FAILED:
      continue
    display_name = pod['displayName']
    pod_id = pod['id']

    log = k8s_client.read_namespaced_pod_log(
        pod_id, namespace=namespace, container='main')
    for line in log.splitlines():
      logging.info('%s:%s', display_name, line)


# Custom component definitions for testing purpose.
class _HelloWorldSpec(component_spec.ComponentSpec):
  INPUTS = {}
  OUTPUTS = {
      'greeting':
          component_spec.ChannelParameter(type=standard_artifacts.String)
  }
  PARAMETERS = {
      'word': component_spec.ExecutionParameter(type=str),
  }


class _ByeWorldSpec(component_spec.ComponentSpec):
  INPUTS = {
      'hearing': component_spec.ChannelParameter(type=standard_artifacts.String)
  }
  OUTPUTS = {}
  PARAMETERS = {}


class HelloWorldComponent(BaseComponent):
  """Producer component."""

  SPEC_CLASS = _HelloWorldSpec
  EXECUTOR_SPEC = executor_spec.ExecutorContainerSpec(
      # TODO(b/143965964): move the image to private repo if the test is flaky
      # due to docker hub.
      image='google/cloud-sdk:latest',
      command=['sh', '-c'],
      args=[
          'echo "hello {{exec_properties.word}}" | gsutil cp - {{output_dict["greeting"][0].uri}}'
      ])

  def __init__(self, word, greeting=None):
    if not greeting:
      artifact = standard_artifacts.String()
      greeting = channel_utils.as_channel([artifact])
    super(HelloWorldComponent,
          self).__init__(_HelloWorldSpec(word=word, greeting=greeting))


class ByeWorldComponent(BaseComponent):
  """Consumer component."""

  SPEC_CLASS = _ByeWorldSpec
  EXECUTOR_SPEC = executor_spec.ExecutorContainerSpec(
      image='bash:latest',
      command=['echo'],
      args=['received {{input_dict["hearing"][0].value}}'])

  def __init__(self, hearing):
    super(ByeWorldComponent, self).__init__(_ByeWorldSpec(hearing=hearing))


def create_primitive_type_components(
    pipeline_name: Text) -> List[BaseComponent]:
  """Creates components for testing primitive type artifact passing.

  Args:
    pipeline_name: Name of this pipeline.

  Returns:
    A list of TFX custom container components.
  """
  hello_world = HelloWorldComponent(word=pipeline_name)
  bye_world = ByeWorldComponent(hearing=hello_world.outputs['greeting'])

  return [hello_world, bye_world]


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
  example_gen = CsvExampleGen(input_base=csv_input_location)
  statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
  schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'])
  example_validator = ExampleValidator(
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema'])
  transform = Transform(
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      module_file=transform_module)
  latest_model_resolver = resolver.Resolver(
      strategy_class=latest_artifact_strategy.LatestArtifactStrategy,
      latest_model=Channel(type=Model)).with_id('latest_model_resolver')
  trainer = Trainer(
      transformed_examples=transform.outputs['transformed_examples'],
      schema=schema_gen.outputs['schema'],
      base_model=latest_model_resolver.outputs['latest_model'],
      transform_graph=transform.outputs['transform_graph'],
      train_args=trainer_pb2.TrainArgs(num_steps=10),
      eval_args=trainer_pb2.EvalArgs(num_steps=5),
      module_file=trainer_module,
  )
  # Set the TFMA config for Model Evaluation and Validation.
  eval_config = tfma.EvalConfig(
      model_specs=[tfma.ModelSpec(signature_name='eval')],
      metrics_specs=[
          tfma.MetricsSpec(
              metrics=[tfma.MetricConfig(class_name='ExampleCount')],
              thresholds={
                  'accuracy':
                      tfma.MetricThreshold(
                          value_threshold=tfma.GenericValueThreshold(
                              lower_bound={'value': 0.5}),
                          change_threshold=tfma.GenericChangeThreshold(
                              direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                              absolute={'value': -1e-10}))
              })
      ],
      slicing_specs=[
          tfma.SlicingSpec(),
          tfma.SlicingSpec(feature_keys=['trip_start_hour'])
      ])
  evaluator = Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      eval_config=eval_config)

  infra_validator = InfraValidator(
      model=trainer.outputs['model'],
      examples=example_gen.outputs['examples'],
      serving_spec=infra_validator_pb2.ServingSpec(
          tensorflow_serving=infra_validator_pb2.TensorFlowServing(
              tags=['latest']),
          kubernetes=infra_validator_pb2.KubernetesConfig()),
      request_spec=infra_validator_pb2.RequestSpec(
          tensorflow_serving=infra_validator_pb2.TensorFlowServingRequestSpec())
  )

  pusher = Pusher(
      model=trainer.outputs['model'],
      model_blessing=evaluator.outputs['blessing'],
      push_destination=pusher_pb2.PushDestination(
          filesystem=pusher_pb2.PushDestination.Filesystem(
              base_directory=os.path.join(pipeline_root, 'model_serving'))))

  return [
      example_gen,
      statistics_gen,
      schema_gen,
      example_validator,
      transform,
      latest_model_resolver,
      trainer,
      evaluator,
      infra_validator,
      pusher,
  ]


@retry.retry(ignore_eventual_failure=True)
def delete_ai_platform_model(model_name):
  """Delete pushed model with the given name in AI Platform."""
  # In order to delete model, all versions in the model must be deleted first.
  versions_command = ('gcloud', 'ai-platform', 'versions', 'list',
                      '--model={}'.format(model_name), '--region=global')
  # The return code of the following subprocess call will be explicitly checked
  # using the logic below, so we don't need to call check_output().
  versions = subprocess.run(versions_command, stdout=subprocess.PIPE)  # pylint: disable=subprocess-run-check
  if versions.returncode == 0:
    logging.info('Model %s has versions %s', model_name, versions.stdout)
    # The first stdout line is headers, ignore. The columns are
    # [NAME] [DEPLOYMENT_URI] [STATE]
    #
    # By specification of test case, the last version in the output list is the
    # default version, which will be deleted last in the for loop, so there's no
    # special handling needed hear.
    # The operation setting default version is at
    # https://github.com/tensorflow/tfx/blob/65633c772f6446189e8be7c6332d32ea221ff836/tfx/extensions/google_cloud_ai_platform/runner.py#L309
    for version in versions.stdout.decode('utf-8').strip('\n').split('\n')[1:]:
      version = version.split()[0]
      logging.info('Deleting version %s of model %s', version, model_name)
      version_delete_command = ('gcloud', '--quiet', 'ai-platform', 'versions',
                                'delete', version,
                                '--model={}'.format(model_name),
                                '--region=global')
      subprocess.run(version_delete_command, check=True)

  logging.info('Deleting model %s', model_name)
  subprocess.run(('gcloud', '--quiet', 'ai-platform', 'models', 'delete',
                  model_name, '--region=global'),
                 check=True)


class BaseKubeflowTest(test_case_utils.TfxTest):
  """Base class that defines testing harness for pipeline on KubeflowRunner."""

  _POLLING_INTERVAL_IN_SECONDS = 10

  # The following environment variables need to be set prior to calling the test
  # in this file. All variables are required and do not have a default.

  # The base container image name to use when building the image used in tests.
  _BASE_CONTAINER_IMAGE = os.environ['KFP_E2E_BASE_CONTAINER_IMAGE']

  # The src path to use to build docker image
  _REPO_BASE = os.environ['KFP_E2E_SRC']

  # The project id to use to run tests.
  _GCP_PROJECT_ID = os.environ['KFP_E2E_GCP_PROJECT_ID']

  # The GCP region in which the end-to-end test is run.
  _GCP_REGION = os.environ['KFP_E2E_GCP_REGION']

  # The GCP bucket to use to write output artifacts.
  _BUCKET_NAME = os.environ['KFP_E2E_BUCKET_NAME']

  # The location of test data. The input files are copied to a test-local
  # location for each invocation, and cleaned up at the end of test.
  _TEST_DATA_ROOT = os.environ['KFP_E2E_TEST_DATA_ROOT']

  # The location of test user module. Will be packaged and copied to under the
  # pipeline root before pipeline execution.
  _MODULE_ROOT = os.path.join(
      os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
      'components/testdata/module_file')

  @classmethod
  def setUpClass(cls):
    super(BaseKubeflowTest, cls).setUpClass()

    if ':' not in cls._BASE_CONTAINER_IMAGE:
      # Generate base container image for the test if tag is not specified.
      cls.container_image = '{}:{}'.format(cls._BASE_CONTAINER_IMAGE,
                                           test_utils.random_id())

      # Create a container image for use by test pipelines.
      test_utils.build_and_push_docker_image(cls.container_image,
                                             cls._REPO_BASE)
    else:  # Use the given image as a base image.
      cls.container_image = cls._BASE_CONTAINER_IMAGE

  @classmethod
  def tearDownClass(cls):
    super(BaseKubeflowTest, cls).tearDownClass()

    if cls.container_image != cls._BASE_CONTAINER_IMAGE:
      # Delete container image used in tests.
      logging.info('Deleting image %s', cls.container_image)
      subprocess.run(
          ['gcloud', 'container', 'images', 'delete', cls.container_image],
          check=True)

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
    logging.info('MySQL pod name is: %s', pod_name)
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
    self._test_dir = self.tmp_dir
    self.enter_context(test_case_utils.change_working_dir(self.tmp_dir))

    self._test_output_dir = 'gs://{}/test_output'.format(self._BUCKET_NAME)

    test_id = test_utils.random_id()

    self._testdata_root = 'gs://{}/test_data/{}'.format(self._BUCKET_NAME,
                                                        test_id)
    subprocess.run(
        ['gsutil', 'cp', '-r', self._TEST_DATA_ROOT, self._testdata_root],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    self._data_root = os.path.join(self._testdata_root, 'external', 'csv')
    self._transform_module = os.path.join(self._MODULE_ROOT,
                                          'transform_module.py')
    self._trainer_module = os.path.join(self._MODULE_ROOT, 'trainer_module.py')

    self.addCleanup(self._delete_test_dir, test_id)

  def _delete_test_dir(self, test_id: Text):
    """Deletes files for this test including the module file and data files.

    Args:
      test_id: Randomly generated id of the test.
    """
    test_utils.delete_gcs_files(self._GCP_PROJECT_ID, self._BUCKET_NAME,
                                'test_data/{}'.format(test_id))

  def _delete_workflow(self, workflow_name: Text):
    """Deletes the specified Argo workflow."""
    logging.info('Deleting workflow %s', workflow_name)
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
          result.append('{}={}'.format(k, v))
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
    logging.info('Launching workflow %s with parameter %s', workflow_name,
                 _format_parameter(parameter))
    with test_utils.Timer('RunningPipelineToCompletion'):
      subprocess.run(run_command, check=True)
      # Wait in the loop while pipeline is pending or running state.
      status = 'Pending'
      while status in ('Pending', 'Running'):
        time.sleep(self._POLLING_INTERVAL_IN_SECONDS)
        status = self._get_argo_pipeline_status(workflow_name)

  def _delete_pipeline_output(self, pipeline_name: Text):
    """Deletes output produced by the named pipeline.

    Args:
      pipeline_name: The name of the pipeline.
    """
    test_utils.delete_gcs_files(self._GCP_PROJECT_ID, self._BUCKET_NAME,
                                'test_output/{}'.format(pipeline_name))

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
    logging.info('Dropping MLMD DB with name: %s', db_name)

    with test_utils.Timer('DeletingMLMDDatabase'):
      subprocess.run(command, check=True)

  def _pipeline_root(self, pipeline_name: Text):
    return os.path.join(self._test_output_dir, pipeline_name)

  def _create_pipeline(self, pipeline_name: Text,
                       components: List[BaseComponent]):
    """Creates a pipeline given name and list of components."""
    return tfx_pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=self._pipeline_root(pipeline_name),
        components=components,
        enable_cache=True,
    )

  def _create_dataflow_pipeline(self,
                                pipeline_name: Text,
                                components: List[BaseComponent],
                                wait_until_finish_ms: int = 1000 * 60 * 20):
    """Creates a pipeline with Beam DataflowRunner."""
    pipeline = self._create_pipeline(pipeline_name, components)
    pipeline.beam_pipeline_args = [
        '--runner=TestDataflowRunner',
        '--wait_until_finish_duration=%d' % wait_until_finish_ms,
        '--project=' + self._GCP_PROJECT_ID,
        '--temp_location=' +
        os.path.join(self._pipeline_root(pipeline_name), 'tmp'),
        '--region=' + self._GCP_REGION,

        # TODO(b/171733562): Remove `use_runner_v2` once it is the default for
        # Dataflow.
        '--experiments=use_runner_v2',
    ]
    return pipeline

  def _get_kubeflow_metadata_config(
      self) -> kubeflow_pb2.KubeflowMetadataConfig:
    config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()
    return config

  def _get_argo_pipeline_status(self, workflow_name: Text) -> Text:
    """Get Pipeline status.

    Args:
      workflow_name: The name of the workflow.

    Returns:
      Simple status string which is returned from `argo get` command.
    """
    get_workflow_command = [
        'argo', '--namespace', 'kubeflow', 'get', workflow_name
    ]
    output = subprocess.check_output(get_workflow_command).decode('utf-8')
    logging.info('Argo output ----\n%s', output)
    match = re.search(r'^Status:\s+(.+)$', output, flags=re.MULTILINE)
    self.assertIsNotNone(match)
    return match.group(1)

  def _compile_and_run_pipeline(self,
                                pipeline: tfx_pipeline.Pipeline,
                                workflow_name: Text = None,
                                parameters: Dict[Text, Any] = None):
    """Compiles and runs a KFP pipeline.

    Args:
      pipeline: The logical pipeline to run.
      workflow_name: The argo workflow name, default to pipeline name.
      parameters: Value of runtime paramters of the pipeline.
    """
    pipeline_name = pipeline.pipeline_info.pipeline_name
    config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
        kubeflow_metadata_config=self._get_kubeflow_metadata_config(),
        tfx_image=self.container_image)
    kubeflow_dag_runner.KubeflowDagRunner(config=config).run(pipeline)

    file_path = os.path.join(self._test_dir, '{}.tar.gz'.format(pipeline_name))
    self.assertTrue(fileio.exists(file_path))
    tarfile.TarFile.open(file_path).extract('pipeline.yaml')
    pipeline_file = os.path.join(self._test_dir, 'pipeline.yaml')
    self.assertIsNotNone(pipeline_file)

    workflow_name = workflow_name or pipeline_name
    # Ensure cleanup regardless of whether pipeline succeeds or fails.
    self.addCleanup(self._delete_workflow, workflow_name)
    self.addCleanup(self._delete_pipeline_metadata, pipeline_name)
    self.addCleanup(self._delete_pipeline_output, pipeline_name)

    # Run the pipeline to completion.
    self._run_workflow(pipeline_file, workflow_name, parameters)

    # Obtain workflow logs.
    get_logs_command = [
        'argo', '--namespace', 'kubeflow', 'logs', '-w', workflow_name
    ]
    logs_output = subprocess.check_output(get_logs_command).decode('utf-8')

    # Check if pipeline completed successfully.
    status = self._get_argo_pipeline_status(workflow_name)
    self.assertEqual(
        'Succeeded', status, 'Pipeline {} failed to complete successfully: {}'
        '\nFailed workflow logs:\n{}'.format(pipeline_name, status,
                                             logs_output))
