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
"""Tests for tfx.orchestration.kubeflow.v2.e2e_tests.artifact_value_placeholder_integration."""

import tensorflow as tf
from tfx import v1 as tfx
from tfx.dsl.component.experimental import placeholders
from tfx.orchestration import test_utils
from tfx.orchestration.kubeflow.v2.e2e_tests import base_test_case
from tfx.types.experimental import simple_artifacts


def _tasks_for_pipeline_with_artifact_value_passing():
  """A simple pipeline with artifact consumed as value."""
  producer_component = tfx.dsl.experimental.create_container_component(
      name='Produce',
      outputs={
          'data': simple_artifacts.File,
      },
      parameters={
          'message': str,
      },
      image='gcr.io/ml-pipeline/mirrors/cloud-sdk',
      command=[
          'sh',
          '-exc',
          """
            message="$0"
            output_data_uri="$1"
            output_data_path=$(mktemp)

            # Running the main code
            echo "Hello $message" >"$output_data_path"

            # Getting data out of the container
            gsutil cp -r "$output_data_path" "$output_data_uri"
          """,
          placeholders.InputValuePlaceholder('message'),
          placeholders.OutputUriPlaceholder('data'),
      ],
  )

  print_value_component = tfx.dsl.experimental.create_container_component(
      name='Print',
      inputs={
          'text': simple_artifacts.File,
      },
      image='gcr.io/ml-pipeline/mirrors/cloud-sdk',
      command=[
          'echo',
          placeholders.InputValuePlaceholder('text'),
      ],
  )

  producer_task = producer_component(message='World!')
  print_task = print_value_component(text=producer_task.outputs['data'],)
  return [producer_task, print_task]


class ArtifactValuePlaceholderIntegrationTest(base_test_case.BaseKubeflowV2Test
                                             ):

  def testArtifactValuePlaceholders(self):
    component_instances = (_tasks_for_pipeline_with_artifact_value_passing())

    pipeline_name = 'kubeflow-v2-test-artifact-value-{}'.format(
        test_utils.random_id())

    pipeline = self._create_pipeline(
        pipeline_name,
        pipeline_components=component_instances,
    )

    self._run_pipeline(pipeline)


if __name__ == '__main__':
  tf.test.main()
