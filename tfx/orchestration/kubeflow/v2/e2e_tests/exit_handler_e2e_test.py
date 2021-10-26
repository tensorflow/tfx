# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Tests for tfx.orchestration.kubeflow.v2.e2e_tests.exit_handler_e2e."""

import tensorflow as tf
from tfx import v1 as tfx
from tfx.orchestration.kubeflow.v2 import test_utils
from tfx.orchestration.kubeflow.v2.e2e_tests import base_test_case


@tfx.orchestration.experimental.exit_handler
def test_exit_handler(param1: tfx.dsl.components.Parameter[str]):
  del param1


class ExitHandlerE2ETest(base_test_case.BaseKubeflowV2Test):

  def testSimpleExitHandlerPipeline(self):
    """End-to-End test for a simple pipeline with exit handler."""
    pipeline = test_utils.two_step_pipeline()

    exit_handler = test_exit_handler(
        param1=tfx.orchestration.experimental.FinalStatusStr())

    self._run_pipeline(pipeline, exit_handler)
    # TODO(bug/194803330): verify the execution result in GCS bucket


if __name__ == '__main__':
  tf.test.main()
