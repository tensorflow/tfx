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
"""Tests for tfx.orchestration.kubeflow.v2.e2e_tests.bigquery_integration."""

import datetime
from unittest import mock

import tensorflow as tf
from tfx import v1 as tfx
from tfx.dsl.components.base import base_component
from tfx.orchestration.kubeflow.v2.e2e_tests import base_test_case
from tfx.utils import proto_utils


@tfx.dsl.components.component
def range_config_generator(input_date: tfx.dsl.components.Parameter[str],
                           range_config: tfx.dsl.components.OutputArtifact[
                               tfx.types.standard_artifacts.String]):
  """Implements the custom compo`Rannent to convert date into span number.

  Args:
    input_date: input date to generate range_config.
    range_config: range_config to ExampleGen.
  """
  start_time = datetime.datetime(2022, 1,
                                 1)  # start time calculate span number from.
  datem = datetime.datetime.strptime(input_date, '%Y%m%d')
  span_number = (datetime.datetime(datem.year, datem.month, datem.day) -
                 start_time).days
  range_config_str = proto_utils.proto_to_json(
      tfx.proto.RangeConfig(
          static_range=tfx.proto.StaticRange(
              start_span_number=span_number, end_span_number=span_number)))
  range_config.value = range_config_str


def two_step_pipeline_with_dynamic_exec_properties():
  """Returns a simple 2-step pipeline under test with the second component's execution property depending dynamically on the first one's output."""

  input_config_generator = range_config_generator(  # pylint: disable=no-value-for-parameter
      input_date='07-13-22')
  example_gen = tfx.extensions.google_cloud_big_query.BigQueryExampleGen(
      query='SELECT * FROM TABLE',
      range_config=input_config_generator.outputs['range_config'].future()
      [0].value).with_beam_pipeline_args([
          '--runner=DataflowRunner',
      ])
  return tfx.dsl.Pipeline(
      pipeline_name='test-dynamic-exec-properties',
      pipeline_root='gs://tfx-testing-bucket/dep-test',
      components=[input_config_generator, example_gen],
      beam_pipeline_args=[
          '--project=tfx-testing',
      ])


class DynamicExecutionPropertiesE2ETest(base_test_case.BaseKubeflowV2Test):

  @mock.patch.object(base_component.BaseComponent, '_resolve_pip_dependencies')
  def testSimpleEnd2EndPipeline(self, moke_resolve_dependencies):
    """End-to-End test for a simple pipeline."""
    pipeline = two_step_pipeline_with_dynamic_exec_properties()
    self._run_pipeline(pipeline)
    moke_resolve_dependencies.assert_called()


if __name__ == '__main__':
  tf.test.main()
