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
"""E2E test using kubeflow orchestrator for penguin template."""

from absl import logging
import tensorflow as tf
from tfx.experimental.templates import container_based_test_case


class PenguinTemplateKubeflowE2ETest(
    container_based_test_case.BaseKubeflowEndToEndTest):

  def _generate_pipeline_name(self, random_id: str):
    return f'penguin-template-kubeflow-e2e-test-{random_id}'

  def testPipeline(self):
    self._copyTemplate('penguin')

    # Prepare data
    self._prepare_data()
    self._replaceFileContent('kubeflow_runner.py', [
        ('DATA_PATH = \'gs://{}/tfx-template/data/penguin/\'.format(configs.GCS_BUCKET_NAME)',
         'DATA_PATH = \'gs://{{}}/{}/{}\'.format(configs.GCS_BUCKET_NAME)'
         .format(self._DATA_DIRECTORY_NAME, self._pipeline_name)),
    ])

    self._compile_pipeline()
    self._check_telemetry_label()

    # Create a pipeline with only one component.
    self._create_pipeline()
    self._run_pipeline()

    # Update the pipeline to include all components.
    updated_pipeline_file = self._addAllComponents()
    logging.info('Updated %s to add all components to the pipeline.',
                 updated_pipeline_file)
    self._update_pipeline()
    self._run_pipeline()


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.test.main()
