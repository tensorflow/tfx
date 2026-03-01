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

import datetime
import json
import os
import time
from typing import List

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
from tfx.dsl.component.experimental import executor_specs
from tfx.dsl.components.base.base_component import BaseComponent
from tfx.dsl.components.common import resolver
from tfx.dsl.input_resolution.strategies import latest_artifact_strategy
from tfx.dsl.placeholder import placeholder as ph
from tfx.proto import infra_validator_pb2
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.types import Channel
from tfx.types import channel_utils
from tfx.types import component_spec
from tfx.types import standard_artifacts
from tfx.types.standard_artifacts import Model
from tfx.utils import kube_utils


# TODO(jiyongjung): Merge with kube_utils.PodStatus
# Various execution status of a KFP pipeline.
KFP_RUNNING_STATUS = 'running'
KFP_SUCCESS_STATUS = 'succeeded'
KFP_FAIL_STATUS = 'failed'
KFP_SKIPPED_STATUS = 'skipped'
KFP_ERROR_STATUS = 'error'

KFP_FINAL_STATUS = frozenset(
    (KFP_SUCCESS_STATUS, KFP_FAIL_STATUS, KFP_SKIPPED_STATUS, KFP_ERROR_STATUS))


def poll_kfp_with_retry(host: str, run_id: str, retry_limit: int,
                        timeout: datetime.timedelta,
                        polling_interval: int) -> str:
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


def print_failure_log_for_run(host: str, run_id: str, namespace: str):
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
  EXECUTOR_SPEC = executor_specs.TemplatedExecutorContainerSpec(
      # TODO(b/143965964): move the image to private repo if the test is flaky
      # due to docker hub.
      image='gcr.io/google.com/cloudsdktool/cloud-sdk:latest',
      command=['sh', '-c'],
      args=[
          'echo "hello ' +
          ph.exec_property('word') +
          '" | gsutil cp - ' +
          ph.output('greeting')[0].uri
      ])

  def __init__(self, word, greeting=None):
    if not greeting:
      artifact = standard_artifacts.String()
      greeting = channel_utils.as_channel([artifact])
    super().__init__(_HelloWorldSpec(word=word, greeting=greeting))


class ByeWorldComponent(BaseComponent):
  """Consumer component."""

  SPEC_CLASS = _ByeWorldSpec
  EXECUTOR_SPEC = executor_specs.TemplatedExecutorContainerSpec(
      image='bash:latest',
      command=['echo'],
      args=['received ' + ph.input('hearing')[0].value])

  def __init__(self, hearing):
    super().__init__(_ByeWorldSpec(hearing=hearing))


def create_primitive_type_components(pipeline_name: str) -> List[BaseComponent]:
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
    pipeline_root: str,
    csv_input_location: str,
    trainer_module: str,
) -> List[BaseComponent]:
  """Creates components for a simple Chicago Taxi TFX pipeline for testing.

  Args:
    pipeline_root: The root of the pipeline output.
    csv_input_location: The location of the input data directory.
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
      module_file=trainer_module)
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
          # TODO(b/244254788): Roll back to the 'latest' tag.
          tensorflow_serving=infra_validator_pb2.TensorFlowServing(
              tags=['2.8.2']),
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
