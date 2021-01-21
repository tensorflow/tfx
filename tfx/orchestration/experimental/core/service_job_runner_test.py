# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Tests for tfx.orchestration.experimental.core.service_job_runner."""

import time

from absl.testing.absltest import mock
import tensorflow as tf
from tfx.orchestration.experimental.core import service_job_runner as runner
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import test_case_utils as tu


class _FakeServiceJobRunner(runner.ServiceJobRunner):

  def run(self):
    while True:
      time.sleep(1)

  def cancel(self):
    pass


class ServiceJobRunnerRegistryTest(tu.TfxTest):

  def test_registration_and_creation(self):
    # Create a pipeline IR containing deployment config for testing.
    deployment_config = pipeline_pb2.IntermediateDeploymentConfig()
    executor_spec = pipeline_pb2.ExecutorSpec.PythonClassExecutorSpec(
        class_path='trainer.TrainerExecutor')
    deployment_config.executor_specs['Trainer'].Pack(executor_spec)
    pipeline = pipeline_pb2.Pipeline()
    pipeline.deployment_config.Pack(deployment_config)

    # Register a fake service job runner.
    spec_type_url = deployment_config.executor_specs['Trainer'].type_url
    runner.ServiceJobRunnerRegistry.register(spec_type_url,
                                             _FakeServiceJobRunner)

    job_runner = runner.ServiceJobRunnerRegistry.create_service_job_runner(
        mock.Mock(), pipeline, 'Trainer', mock.Mock())
    self.assertIsInstance(job_runner, _FakeServiceJobRunner)


if __name__ == '__main__':
  tf.test.main()
