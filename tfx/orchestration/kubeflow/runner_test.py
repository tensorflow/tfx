# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for tfx.orchestration.kubeflow.runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tarfile
import tempfile
import tensorflow as tf
import yaml

from tfx.components.example_gen.big_query_example_gen import component as big_query_example_gen_component
from tfx.components.statistics_gen import component as statistics_gen_component
from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration.kubeflow import runner


# 2-step pipeline under test.
@tfx_pipeline.PipelineDecorator(
    pipeline_name='two_step_pipeline',
    log_root='/var/tmp/tfx/logs',
    pipeline_root='/pipeline/root',
)
def _two_step_pipeline():
  example_gen = big_query_example_gen_component.BigQueryExampleGen(
      query='SELECT * FROM TABLE')
  statistics_gen = statistics_gen_component.StatisticsGen(
      input_data=example_gen.outputs.examples)
  return [example_gen, statistics_gen]


class RunnerTest(tf.test.TestCase):

  def setUp(self):
    self.test_dir = tempfile.mkdtemp()
    os.chdir(self.test_dir)

  def tearDown(self):
    shutil.rmtree(self.test_dir)

  def test_two_step_pipeline(self):
    """Sanity-checks the construction and dependencies for a 2-step pipeline.
    """
    runner.KubeflowRunner().run(_two_step_pipeline())
    file_path = os.path.join(self.test_dir,
                             'two_step_pipeline.tar.gz')
    self.assertTrue(tf.gfile.Exists(file_path))

    with tarfile.TarFile.open(file_path).extractfile(
        'pipeline.yaml') as pipeline_file:
      self.assertIsNotNone(pipeline_file)
      pipeline = yaml.load(pipeline_file)

      containers = [
          c for c in pipeline['spec']['templates'] if 'container' in c
      ]
      self.assertEqual(2, len(containers))

      big_query_container = [
          c
          for c in containers
          if c['name'] == 'bigqueryexamplegen'
      ]
      self.assertEqual(1, len(big_query_container))
      self.assertEqual([
          'python',
          '/tfx-src/tfx/orchestration/kubeflow/container_entrypoint.py'
      ], big_query_container[0]['container']['command'])

      statistics_gen_container = [
          c
          for c in containers
          if c['name'] == 'statisticsgen'
      ]
      self.assertEqual(1, len(statistics_gen_container))

      # Ensure dependencies between components are captured.
      dag = [c for c in pipeline['spec']['templates'] if 'dag' in c]
      self.assertEqual(1, len(dag))

      parameter_value = ('{{tasks.bigqueryexamplegen.outputs.parameters'
                         '.bigqueryexamplegen-examples}}')

      self.assertEqual({
          'tasks': [{
              'name': 'bigqueryexamplegen',
              'template': 'bigqueryexamplegen'
          },
                    {
                        'name': 'statisticsgen',
                        'template': 'statisticsgen',
                        'dependencies': ['bigqueryexamplegen'],
                        'arguments': {
                            'parameters': [{
                                'name': 'bigqueryexamplegen-examples',
                                'value': parameter_value
                            }],
                        },
                    }]
      }, dag[0]['dag'])


if __name__ == '__main__':
  tf.test.main()
