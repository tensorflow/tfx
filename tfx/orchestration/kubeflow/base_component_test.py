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
"""Tests for tfx.orchestration.kubeflow.base_component."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kfp import dsl
import tensorflow as tf

from ml_metadata.proto import metadata_store_pb2
from tfx.components.example_gen.csv_example_gen import component as csv_example_gen_component
from tfx.components.statistics_gen import component as statistics_gen_component
from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration.kubeflow import base_component
from tfx.orchestration.kubeflow.proto import kubeflow_pb2
from tfx.types import channel_utils
from tfx.types import standard_artifacts


class BaseComponentTest(tf.test.TestCase):
  maxDiff = None  # pylint: disable=invalid-name

  def setUp(self):
    super(BaseComponentTest, self).setUp()
    examples = standard_artifacts.ExternalArtifact()
    example_gen = csv_example_gen_component.CsvExampleGen(
        input_base=channel_utils.as_channel([examples]))
    statistics_gen = statistics_gen_component.StatisticsGen(
        input_data=example_gen.outputs.examples)

    pipeline = tfx_pipeline.Pipeline(
        pipeline_name='test_pipeline',
        pipeline_root='test_pipeline_root',
        metadata_connection_config=metadata_store_pb2.ConnectionConfig(),
        components=[example_gen, statistics_gen],
    )

    self._metadata_config = kubeflow_pb2.KubeflowMetadataConfig()
    self._metadata_config.mysql_db_service_host.environment_variable = 'MYSQL_SERVICE_HOST'
    with dsl.Pipeline('test_pipeline'):

      self.component = base_component.BaseComponent(
          component=statistics_gen,
          depends_on=set([example_gen]),
          pipeline=pipeline,
          tfx_image='container_image',
          kubeflow_metadata_config=self._metadata_config,
      )

  def testContainerOpArguments(self):
    self.assertEqual(self.component.container_op.arguments[:20], [
        '--pipeline_name',
        'test_pipeline',
        '--pipeline_root',
        'test_pipeline_root',
        '--kubeflow_metadata_config',
        '{\n'
        '  "mysqlDbServiceHost": {\n'
        '    "environmentVariable": "MYSQL_SERVICE_HOST"\n'
        '  }\n'
        '}',
        '--additional_pipeline_args',
        '{}',
        '--component_id',
        'StatisticsGen',
        '--component_name',
        'StatisticsGen',
        '--component_type',
        'tfx.components.statistics_gen.component.StatisticsGen',
        '--name',
        'None',
        '--driver_class_path',
        'tfx.components.base.base_driver.BaseDriver',
        '--executor_spec',
        '{'
        '"__class__": "ExecutorClassSpec", '
        '"__module__": "tfx.components.base.executor_spec", '
        '"__tfx_object_type__": "jsonable", '
        '"executor_class": {'
        '"__class__": "Executor", '
        '"__module__": "tfx.components.statistics_gen.executor", '
        '"__tfx_object_type__": "class"}'
        '}',
    ])


if __name__ == '__main__':
  tf.test.main()
