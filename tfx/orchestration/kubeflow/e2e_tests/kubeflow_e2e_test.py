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

import os
import subprocess
import time
from typing import List

from absl import logging
from grpc import insecure_channel
import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.orchestration import test_utils
from tfx.orchestration.experimental.core.testing import test_dynamic_exec_properties_pipeline
from tfx.orchestration.kubeflow import test_utils as kubeflow_test_utils
from tfx.orchestration.test_pipelines import download_grep_print_pipeline
from tfx.types import standard_artifacts

from ml_metadata.proto import metadata_store_pb2
from ml_metadata.proto import metadata_store_service_pb2
from ml_metadata.proto import metadata_store_service_pb2_grpc

# The range of port-forwarding addresses used by Kubeflow E2E test.
# If the current specified address is occupied, the test will scan forward until
# a unused port is met, or stop at _KFP_E2E_TEST_FORWARDING_PORT_END.
_KFP_E2E_TEST_FORWARDING_PORT_BEGIN = 8081
_KFP_E2E_TEST_FORWARDING_PORT_END = 8888

# Number of attempts to bind one port.
_MAX_ATTEMPTS = 5

# Context name of pipeline contexts.
_CONTEXT_TYPE_PIPELINE = 'pipeline'


class KubeflowEndToEndTest(kubeflow_test_utils.BaseKubeflowTest):

  @classmethod
  def setUpClass(cls):
    # Initializes the port-forward process to talk MLMD.
    super().setUpClass()
    cls._port_forwarding_process = cls._setup_mlmd_port_forward()

  @classmethod
  def tearDownClass(cls):
    super(KubeflowEndToEndTest, cls).tearDownClass()

    # Delete container image used in tests.
    logging.info('Killing the GRPC port-forwarding process.')
    cls._port_forwarding_process.kill()

  @classmethod
  def _get_grpc_port(cls) -> str:
    """Get the port number used by MLMD gRPC server."""
    get_grpc_port_command = [
        'kubectl', '-n', 'kubeflow', 'get', 'configmap',
        'metadata-grpc-configmap', '-o',
        'jsonpath={.data.METADATA_GRPC_SERVICE_PORT}'
    ]

    grpc_port = subprocess.check_output(get_grpc_port_command)
    return grpc_port.decode('utf-8')

  @classmethod
  def _setup_mlmd_port_forward(cls) -> subprocess.Popen:
    """Uses port forward to talk to MLMD gRPC server."""
    grpc_port = cls._get_grpc_port()

    is_bind = False
    forwarded_port = None

    for port in range(_KFP_E2E_TEST_FORWARDING_PORT_BEGIN,
                      _KFP_E2E_TEST_FORWARDING_PORT_END):
      grpc_forward_command = [
          'kubectl', 'port-forward', 'deployment/metadata-grpc-deployment',
          '-n', 'kubeflow', ('%s:%s' % (port, grpc_port))
      ]
      # Begin port forwarding.
      proc = subprocess.Popen(grpc_forward_command)
      try:
        # Wait while port forward to pod is being established
        poll_grpc_port_command = ['lsof', '-i', ':%s' % port]
        result = subprocess.run(  # pylint: disable=subprocess-run-check
            poll_grpc_port_command,
            stdout=subprocess.PIPE)
        for _ in range(_MAX_ATTEMPTS):
          if (result.returncode == 0 and
              'kubectl' in result.stdout.decode('utf-8')):
            is_bind = True
            break
          logging.info(
              'Waiting while gRPC port-forward is being established...')
          time.sleep(5)
          result = subprocess.run(  # pylint: disable=subprocess-run-check
              poll_grpc_port_command,
              stdout=subprocess.PIPE)

      except:  # pylint: disable=bare-except
        # Kill the process in case unexpected error occurred.
        proc.kill()

      if is_bind:
        forwarded_port = port
        break

    if not is_bind:
      raise RuntimeError('Failed to establish gRPC port-forward to cluster in '
                         'the specified range: port %s to %s' %
                         (_KFP_E2E_TEST_FORWARDING_PORT_BEGIN,
                          _KFP_E2E_TEST_FORWARDING_PORT_END))

    # Establish MLMD gRPC channel.
    forwarding_channel = insecure_channel('localhost:%s' % forwarded_port)
    cls._stub = metadata_store_service_pb2_grpc.MetadataStoreServiceStub(
        forwarding_channel)

    return proc

  def _get_artifacts_with_type_and_pipeline(
      self, type_name: str,
      pipeline_name: str) -> List[metadata_store_pb2.Artifact]:
    """Helper function returns artifacts of specified pipeline and type."""
    # 1. Find the pipeline context according to its name.
    request = metadata_store_service_pb2.GetContextByTypeAndNameRequest(
        type_name=_CONTEXT_TYPE_PIPELINE, context_name=pipeline_name)
    pipeline_context = self._stub.GetContextByTypeAndName(request)
    # 2. Find the artifacts associated with the pipeline context.
    request = metadata_store_service_pb2.GetArtifactsByContextRequest(
        context_id=pipeline_context.context.id)
    artifacts_response = self._stub.GetArtifactsByContext(request)
    # 3. Find the specified artifact type id.
    artifact_type_request = metadata_store_service_pb2.GetArtifactTypeRequest(
        type_name=type_name)
    artifact_type = self._stub.GetArtifactType(
        artifact_type_request).artifact_type
    # 4. Filter the returned artifacts according to their types and return.
    return [
        artifact for artifact in artifacts_response.artifacts
        if artifact.type_id == artifact_type.id
    ]

  def _get_value_of_string_artifact(
      self, string_artifact: metadata_store_pb2.Artifact) -> str:
    """Helper function returns the actual value of a ValueArtifact."""

    string_artifact_obj = standard_artifacts.String()
    string_artifact_obj.uri = string_artifact.uri
    string_artifact_obj.read()
    return string_artifact_obj.value

  def _get_executions_by_pipeline_name(
      self, pipeline_name: str) -> List[metadata_store_pb2.Execution]:
    """Helper function returns executions under a given pipeline name."""
    # step 1: get context id by context name
    request = metadata_store_service_pb2.GetContextByTypeAndNameRequest(
        type_name='pipeline', context_name=pipeline_name)
    context_id = self._stub.GetContextByTypeAndName(request).context.id
    # step 2: get executions by context id
    request = metadata_store_service_pb2.GetExecutionsByContextRequest(
        context_id=context_id)
    return self._stub.GetExecutionsByContext(request).executions

  def _get_executions_by_pipeline_name_and_state(
      self, pipeline_name: str, state: metadata_store_pb2.Execution.State
  ) -> List[metadata_store_pb2.Execution]:
    """Helper function returns executions for a given state."""
    executions = self._get_executions_by_pipeline_name(pipeline_name)
    result = []
    for e in executions:
      if e.last_known_state == state:
        result.append(e)

    return result

  def _assert_infra_validator_passed(self, pipeline_name: str):
    artifacts = self._get_artifacts_with_type_and_pipeline(
        type_name='InfraBlessing', pipeline_name=pipeline_name)
    self.assertGreaterEqual(len(artifacts), 1)
    for artifact in artifacts:
      blessed = os.path.join(artifact.uri, 'INFRA_BLESSED')
      self.assertTrue(
          fileio.exists(blessed),
          'Expected InfraBlessing results cannot be found under path %s for '
          'artifact %s' % (blessed, artifact))

  def testSimpleEnd2EndPipeline(self):
    """End-to-End test for simple pipeline."""
    pipeline_name = 'kubeflow-e2e-test-{}'.format(test_utils.random_id())
    # Test data is copied from the repository(tfx/components/testdata/) to an
    # ephemeral location in GCS bucket(BaseKubeflowTest._BUCKET_NAME).
    # See kubeflow_test_utils.BaseKubeflowTest.setUp() for the detail.
    components = kubeflow_test_utils.create_e2e_components(
        self._pipeline_root(pipeline_name),
        self._data_root,
        self._transform_module,
        self._trainer_module,
    )
    pipeline = self._create_pipeline(pipeline_name, components)

    self._compile_and_run_pipeline(pipeline)
    self._assert_infra_validator_passed(pipeline_name)

  def testPrimitiveEnd2EndPipeline(self):
    """End-to-End test for primitive artifacts passing."""
    pipeline_name = 'kubeflow-primitive-e2e-test-{}'.format(
        test_utils.random_id())
    components = kubeflow_test_utils.create_primitive_type_components(
        pipeline_name)
    # Test that the pipeline can be executed successfully.
    pipeline = self._create_pipeline(pipeline_name, components)
    self._compile_and_run_pipeline(
        pipeline=pipeline, workflow_name=pipeline_name + '-run-1')
    # Test if the correct value has been passed.
    str_artifacts = self._get_artifacts_with_type_and_pipeline(
        type_name='String', pipeline_name=pipeline_name)
    # There should be exactly one string artifact.
    self.assertEqual(1, len(str_artifacts))
    self.assertEqual(
        self._get_value_of_string_artifact(str_artifacts[0]),
        'hello %s\n' % pipeline_name)
    # Test caching.
    self._compile_and_run_pipeline(
        pipeline=pipeline, workflow_name=pipeline_name + '-run-2')
    cached_execution = self._get_executions_by_pipeline_name_and_state(
        pipeline_name=pipeline_name,
        state=metadata_store_pb2.Execution.State.CACHED)
    self.assertEqual(2, len(cached_execution))

  def testCreateContainerComponentEnd2EndPipeline(self):
    """End-to-End test for container components."""
    pipeline_name = 'kubeflow-container-e2e-test-{}'.format(
        test_utils.random_id())
    text_url = (
        'https://storage.googleapis.com/ml-pipeline-playground/hamlet.txt')
    pattern = 'art thou'
    component_instances = download_grep_print_pipeline.create_pipeline_component_instances(
        text_url=text_url,
        pattern=pattern,
    )
    # Test that the pipeline can be executed successfully.
    pipeline = self._create_pipeline(pipeline_name, component_instances)
    self._compile_and_run_pipeline(
        pipeline=pipeline, workflow_name=pipeline_name)
    # Test if the correct value has been passed.
    artifacts = self._get_artifacts_with_type_and_pipeline(
        type_name='ExternalArtifact', pipeline_name=pipeline_name)
    # There should be exactly two artifacts.
    self.assertEqual(len(artifacts), 2)
    for artifact in artifacts:
      # TODO(b/150515270) Remove the '/data' suffix when b/150515270 is fixed.
      artifact_value = fileio.open(artifact.uri + '/data', 'r').read()
      self.assertGreater(len(artifact_value), 100)

  def testDynamicPropertiesEnd2EndPipeline(self):
    pipeline_name = 'kubeflow-dynamic-exec-e2e-test-{}'.format(
        test_utils.random_id())
    components = test_dynamic_exec_properties_pipeline.create_components()
    pipeline = self._create_pipeline(pipeline_name, components)
    self._compile_and_run_pipeline(
        pipeline=pipeline, workflow_name=pipeline_name)
    artifacts = self._get_artifacts_with_type_and_pipeline(
        type_name='Integer', pipeline_name=pipeline_name)
    self.assertEqual(len(artifacts), 1)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.test.main()
