# Lint as: python3
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

import os
import subprocess
import time
from typing import List, Text

from absl import logging
from grpc import insecure_channel
import tensorflow as tf

from ml_metadata.proto import metadata_store_pb2
from ml_metadata.proto import metadata_store_service_pb2
from ml_metadata.proto import metadata_store_service_pb2_grpc
from tfx.orchestration import metadata
from tfx.orchestration.kubeflow import test_utils
from tfx.types import standard_artifacts

# TODO(jxzheng): Change this hard-coded port forwarding address to a dynamic
# one, perhaps with incrementing-retry logic, in order to guarantee different
# tests not to step on each other.

# The port-forwarding address used by Kubeflow E2E test.
_KFP_E2E_TEST_FORWARDING_PORT = '8081'


class KubeflowEndToEndTest(test_utils.BaseKubeflowTest):

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
  def _get_grpc_port(cls) -> Text:
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
    grpc_forward_command = [
        'kubectl', 'port-forward', 'deployment/metadata-grpc-deployment', '-n',
        'kubeflow', ('%s:%s' % (_KFP_E2E_TEST_FORWARDING_PORT, grpc_port))
    ]
    # Begin port forwarding.
    proc = subprocess.Popen(grpc_forward_command)
    try:
      # Wait while port forward to pod is being established
      poll_grpc_port_command = [
          'lsof', '-i', ':%s' % _KFP_E2E_TEST_FORWARDING_PORT
      ]
      result = subprocess.run(poll_grpc_port_command)  # pylint: disable=subprocess-run-check
      max_attempts = 30
      for _ in range(max_attempts):
        if result.returncode == 0:
          break
        logging.info('Waiting while gRPC port-forward is being established...')
        time.sleep(5)
        result = subprocess.run(poll_grpc_port_command)  # pylint: disable=subprocess-run-check
    except:
      # Kill the process in case unexpected error occurred.
      proc.kill()
      raise

    if result.returncode != 0:
      raise RuntimeError(
          'Failed to establish gRPC port-forward to cluster after %s retries. '
          'See error message: %s' % (max_attempts, result.stderr))

    # Establish MLMD gRPC channel.
    forwarding_channel = insecure_channel('localhost:%s' % (int(grpc_port) + 1))
    cls._stub = metadata_store_service_pb2_grpc.MetadataStoreServiceStub(
        forwarding_channel)

    return proc

  def _get_artifacts_with_type(
      self, type_name: Text) -> List[metadata_store_pb2.Artifact]:
    """Helper function returns artifacts with given type."""
    request = metadata_store_service_pb2.GetArtifactsByTypeRequest(
        type_name=type_name)
    return self._stub.GetArtifactsByType(request).artifacts

  def _get_artifacts_with_type_and_pipeline(
      self, type_name: Text,
      pipeline_name: Text) -> List[metadata_store_pb2.Artifact]:
    """Helper function returns artifacts of specified pipeline and type."""
    request = metadata_store_service_pb2.GetArtifactsByTypeRequest(
        type_name=type_name)
    all_artifacts = self._stub.GetArtifactsByType(request).artifacts
    return [
        artifact for artifact in all_artifacts
        if artifact.custom_properties['pipeline_name'].string_value ==
        pipeline_name
    ]

  def _get_value_of_string_artifact(
      self, string_artifact: metadata_store_pb2.Artifact) -> Text:
    """Helper function returns the actual value of a ValueArtifact."""
    file_path = os.path.join(string_artifact.uri,
                             standard_artifacts.String.VALUE_FILE)
    # Assert there is a file exists.
    if (not tf.io.gfile.exists(file_path)) or tf.io.gfile.isdir(file_path):
      raise RuntimeError(
          'Given path does not exist or is not a valid file: %s' % file_path)
    serialized_value = tf.io.gfile.GFile(file_path, 'rb').read()
    return standard_artifacts.String().decode(serialized_value)

  def _get_executions_by_pipeline_name(
      self, pipeline_name: Text) -> List[metadata_store_pb2.Execution]:
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
      self, pipeline_name: Text,
      state: Text) -> List[metadata_store_pb2.Execution]:
    """Helper function returns executions for a given state."""
    executions = self._get_executions_by_pipeline_name(pipeline_name)
    result = []
    for e in executions:
      if e.properties['state'].string_value == state:
        result.append(e)

    return result

  def _assert_infra_validator_passed(self, pipeline_name: Text):
    artifacts = self._get_artifacts_with_type_and_pipeline(
        type_name='InfraBlessing', pipeline_name=pipeline_name)
    self.assertGreaterEqual(len(artifacts), 1)
    for artifact in artifacts:
      blessed = os.path.join(artifact.uri, 'INFRA_BLESSED')
      self.assertTrue(
          tf.io.gfile.exists(blessed),
          'Expected InfraBlessing results cannot be found under path %s for '
          'artifact %s' % (blessed, artifact))

  def testSimpleEnd2EndPipeline(self):
    """End-to-End test for simple pipeline."""
    pipeline_name = 'kubeflow-e2e-test-{}'.format(self._random_id())
    components = test_utils.create_e2e_components(
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
    pipeline_name = 'kubeflow-primitive-e2e-test-{}'.format(self._random_id())
    components = test_utils.create_primitive_type_components(pipeline_name)
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
        pipeline_name=pipeline_name, state=metadata.EXECUTION_STATE_CACHED)
    self.assertEqual(2, len(cached_execution))


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.test.main()
