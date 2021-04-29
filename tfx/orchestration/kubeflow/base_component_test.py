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

import json
import os

from absl import logging
from kfp import dsl
import tensorflow as tf
from tfx.components.example_gen.csv_example_gen import component as csv_example_gen_component
from tfx.components.statistics_gen import component as statistics_gen_component
from tfx.orchestration import data_types
from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration.kubeflow import base_component
from tfx.orchestration.kubeflow.proto import kubeflow_pb2
from tfx.orchestration.launcher import in_process_component_launcher
from tfx.proto.orchestration import pipeline_pb2

from ml_metadata.proto import metadata_store_pb2


class BaseComponentTest(tf.test.TestCase):
  maxDiff = None  # pylint: disable=invalid-name
  _test_pipeline_name = 'test_pipeline'

  def setUp(self):
    super(BaseComponentTest, self).setUp()
    example_gen = csv_example_gen_component.CsvExampleGen(
        input_base='data_input')
    statistics_gen = statistics_gen_component.StatisticsGen(
        examples=example_gen.outputs['examples']).with_id('foo')

    pipeline = tfx_pipeline.Pipeline(
        pipeline_name=self._test_pipeline_name,
        pipeline_root='test_pipeline_root',
        metadata_connection_config=metadata_store_pb2.ConnectionConfig(),
        components=[example_gen, statistics_gen],
    )

    test_pipeline_root = dsl.PipelineParam(name='pipeline-root-param')

    self._metadata_config = kubeflow_pb2.KubeflowMetadataConfig()
    self._metadata_config.mysql_db_service_host.environment_variable = 'MYSQL_SERVICE_HOST'
    self._tfx_ir = pipeline_pb2.Pipeline()
    with dsl.Pipeline('test_pipeline'):
      self.component = base_component.BaseComponent(
          component=statistics_gen,
          component_launcher_class=in_process_component_launcher
          .InProcessComponentLauncher,
          depends_on=set(),
          pipeline=pipeline,
          pipeline_name=self._test_pipeline_name,
          pipeline_root=test_pipeline_root,
          tfx_image='container_image',
          kubeflow_metadata_config=self._metadata_config,
          component_config=None,
          tfx_ir=self._tfx_ir,
      )
    self.tfx_component = statistics_gen

  def testContainerOpArguments(self):
    # TODO(hongyes): make the whole args list in one golden file to keep
    # source of truth in same file.
    source_data_dir = os.path.join(os.path.dirname(__file__), 'testdata')
    with open(os.path.join(source_data_dir,
                           'component.json')) as component_json_file:
      formatted_component_json = json.dumps(
          json.load(component_json_file), sort_keys=True)

    expected_args = [
        '--pipeline_name',
        'test_pipeline',
        '--pipeline_root',
        '{{pipelineparam:op=;name=pipeline-root-param}}',
        '--kubeflow_metadata_config',
        '{\n'
        '  "mysql_db_service_host": {\n'
        '    "environment_variable": "MYSQL_SERVICE_HOST"\n'
        '  }\n'
        '}',
        '--beam_pipeline_args',
        '[]',
        '--additional_pipeline_args',
        '{}',
        '--component_launcher_class_path',
        'tfx.orchestration.launcher.in_process_component_launcher.InProcessComponentLauncher',
        '--serialized_component',
        formatted_component_json,
        '--component_config',
        'null',
    ]
    try:
      self.assertEqual(
          self.component.container_op.arguments[:len(expected_args)],
          expected_args)

    except AssertionError:
      # Print out full arguments for debugging.
      logging.error('==== BEGIN CONTAINER OP ARGUMENT DUMP ====')
      logging.error(json.dumps(self.component.container_op.arguments, indent=2))
      logging.error('==== END CONTAINER OP ARGUMENT DUMP ====')
      raise

  def testContainerOpName(self):
    self.assertEqual('foo', self.tfx_component.id)
    self.assertEqual('foo', self.component.container_op.name)


class BaseComponentWithPipelineParamTest(tf.test.TestCase):
  """Test the usage of RuntimeParameter."""
  maxDiff = None  # pylint: disable=invalid-name
  _test_pipeline_name = 'test_pipeline'

  def setUp(self):
    super(BaseComponentWithPipelineParamTest, self).setUp()

    test_pipeline_root = dsl.PipelineParam(name='pipeline-root-param')
    example_gen_buckets = data_types.RuntimeParameter(
        name='example-gen-buckets', ptype=int, default=10)

    example_gen = csv_example_gen_component.CsvExampleGen(
        input_base='data_root',
        output_config={
            'split_config': {
                'splits': [{
                    'name': 'examples',
                    'hash_buckets': example_gen_buckets
                }]
            }
        })
    statistics_gen = statistics_gen_component.StatisticsGen(
        examples=example_gen.outputs['examples']).with_id('foo')

    pipeline = tfx_pipeline.Pipeline(
        pipeline_name=self._test_pipeline_name,
        pipeline_root='test_pipeline_root',
        metadata_connection_config=metadata_store_pb2.ConnectionConfig(),
        components=[example_gen, statistics_gen],
    )

    self._metadata_config = kubeflow_pb2.KubeflowMetadataConfig()
    self._metadata_config.mysql_db_service_host.environment_variable = 'MYSQL_SERVICE_HOST'
    self._tfx_ir = pipeline_pb2.Pipeline()
    with dsl.Pipeline('test_pipeline'):
      self.example_gen = base_component.BaseComponent(
          component=example_gen,
          component_launcher_class=in_process_component_launcher
          .InProcessComponentLauncher,
          depends_on=set(),
          pipeline=pipeline,
          pipeline_name=self._test_pipeline_name,
          pipeline_root=test_pipeline_root,
          tfx_image='container_image',
          kubeflow_metadata_config=self._metadata_config,
          component_config=None,
          tfx_ir=self._tfx_ir)
      self.statistics_gen = base_component.BaseComponent(
          component=statistics_gen,
          component_launcher_class=in_process_component_launcher
          .InProcessComponentLauncher,
          depends_on=set(),
          pipeline=pipeline,
          pipeline_name=self._test_pipeline_name,
          pipeline_root=test_pipeline_root,
          tfx_image='container_image',
          kubeflow_metadata_config=self._metadata_config,
          component_config=None
      )

    self.tfx_example_gen = example_gen
    self.tfx_statistics_gen = statistics_gen

  def testContainerOpArguments(self):
    # TODO(hongyes): make the whole args list in one golden file to keep
    # source of truth in same file.
    source_data_dir = os.path.join(os.path.dirname(__file__), 'testdata')
    with open(os.path.join(source_data_dir,
                           'statistics_gen.json')) as component_json_file:
      formatted_statistics_gen = json.dumps(
          json.load(component_json_file), sort_keys=True)
    with open(os.path.join(source_data_dir,
                           'example_gen.json')) as component_json_file:
      formatted_example_gen = json.dumps(
          json.load(component_json_file), sort_keys=True)

    statistics_gen_expected_args = [
        '--pipeline_name',
        'test_pipeline',
        '--pipeline_root',
        '{{pipelineparam:op=;name=pipeline-root-param}}',
        '--kubeflow_metadata_config',
        '{\n'
        '  "mysql_db_service_host": {\n'
        '    "environment_variable": "MYSQL_SERVICE_HOST"\n'
        '  }\n'
        '}',
        '--beam_pipeline_args',
        '[]',
        '--additional_pipeline_args',
        '{}',
        '--component_launcher_class_path',
        'tfx.orchestration.launcher.in_process_component_launcher.InProcessComponentLauncher',
        '--serialized_component',
        formatted_statistics_gen,
        '--component_config',
        'null',
        '--node_id',
        'foo',
    ]
    example_gen_expected_args = [
        '--pipeline_name',
        'test_pipeline',
        '--pipeline_root',
        '{{pipelineparam:op=;name=pipeline-root-param}}',
        '--kubeflow_metadata_config',
        '{\n'
        '  "mysql_db_service_host": {\n'
        '    "environment_variable": "MYSQL_SERVICE_HOST"\n'
        '  }\n'
        '}',
        '--beam_pipeline_args',
        '[]',
        '--additional_pipeline_args',
        '{}',
        '--component_launcher_class_path',
        'tfx.orchestration.launcher.in_process_component_launcher.InProcessComponentLauncher',
        '--serialized_component',
        formatted_example_gen,
        '--component_config',
        'null',
        '--node_id',
        'CsvExampleGen',
        '--tfx_ir',
        '{}',
    ]
    try:
      self.assertEqual(
          self.statistics_gen.container_op
          .arguments[:len(statistics_gen_expected_args)],
          statistics_gen_expected_args)
      self.assertEqual(
          self.example_gen.container_op.arguments[:len(example_gen_expected_args
                                                      )],
          example_gen_expected_args)
    except AssertionError:
      # Print out full arguments for debugging.
      logging.error('==== BEGIN STATISTICSGEN CONTAINER OP ARGUMENT DUMP ====')
      logging.error(
          json.dumps(self.statistics_gen.container_op.arguments, indent=2))
      logging.error('==== END STATISTICSGEN CONTAINER OP ARGUMENT DUMP ====')
      logging.error('==== BEGIN EXAMPLEGEN CONTAINER OP ARGUMENT DUMP ====')
      logging.error(
          json.dumps(self.example_gen.container_op.arguments, indent=2))
      logging.error('==== END EXAMPLEGEN CONTAINER OP ARGUMENT DUMP ====')
      raise

  def testContainerOpName(self):
    self.assertEqual('foo', self.tfx_statistics_gen.id)
    self.assertEqual('foo', self.statistics_gen.container_op.name)


if __name__ == '__main__':
  tf.test.main()
