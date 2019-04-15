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
"""Tests for tfx.orchestration.airflow.airflow_pipeline."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os

from airflow.operators import dummy_operator
import tensorflow as tf

from tfx.orchestration.airflow import airflow_pipeline
from tfx.utils.types import TfxType


class AirflowPipelineTest(tf.test.TestCase):

  def setUp(self):
    self._temp_dir = os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',
                                    self.get_temp_dir())
    self.pipeline = airflow_pipeline.AirflowPipeline(
        pipeline_name='pipeline_name',
        start_date=datetime.datetime(2018, 1, 1),
        schedule_interval=None,
        pipeline_root='pipeline_root',
        metadata_db_root=self._temp_dir,
        metadata_connection_config=None,
        additional_pipeline_args=None,
        docker_operator_cfg=None,
        enable_cache=True)

  def test_initialize_pipeline(self):
    self.assertEqual(self.pipeline.project_path,
                     os.path.join('pipeline_root', 'pipeline_name'))
    self.assertEqual(
        self.pipeline.metadata_connection_config.sqlite.filename_uri,
        os.path.join(self._temp_dir, 'pipeline_name', 'metadata.db'))

  def test_build_graph(self):
    r"""Tests building airflow DAG graph using add_node_to_graph().

    The dependency graph beside is as below:
                     component_one
                     /           \
                    /             \
          component_two         component_three
                    \             /
                     \           /
                     component_four
    """

    component_one = dummy_operator.DummyOperator(
        task_id='one', dag=self.pipeline)
    component_two = dummy_operator.DummyOperator(
        task_id='two', dag=self.pipeline)
    component_three = dummy_operator.DummyOperator(
        task_id='three', dag=self.pipeline)
    component_four = dummy_operator.DummyOperator(
        task_id='four', dag=self.pipeline)

    component_one_input_a = TfxType('i1a')
    component_one_input_b = TfxType('i1b')
    component_one_output_a = TfxType('o1a')
    component_one_output_b = TfxType('o1b')
    component_two_output = TfxType('o2')
    component_three_output = TfxType('o3')
    component_four_output = TfxType('o4')

    component_one_input_dict = {
        'i1a': [component_one_input_a],
        'i1b': [component_one_input_b]
    }
    component_one_output_dict = {
        'o1a': [component_one_output_a],
        'o1b': [component_one_output_b]
    }
    component_two_input_dict = {
        'i2a': [component_one_output_a],
        'i2b': [component_one_output_b]
    }
    component_two_output_dict = {'o2': [component_two_output]}
    component_three_input_dict = {
        'i3a': [component_one_output_a],
        'i3b': [component_one_output_b]
    }
    component_three_output_dict = {'o3': [component_two_output]}
    component_four_input_dict = {
        'i4a': [component_two_output],
        'i4b': [component_three_output]
    }
    component_four_output_dict = {'o4': [component_four_output]}

    self.pipeline.add_node_to_graph(
        component_one,
        consumes=component_one_input_dict.values(),
        produces=component_one_output_dict.values())
    self.pipeline.add_node_to_graph(
        component_two,
        consumes=component_two_input_dict.values(),
        produces=component_two_output_dict.values())
    self.pipeline.add_node_to_graph(
        component_three,
        consumes=component_three_input_dict.values(),
        produces=component_three_output_dict.values())
    self.pipeline.add_node_to_graph(
        component_four,
        consumes=component_four_input_dict.values(),
        produces=component_four_output_dict.values())

    self.assertItemsEqual(component_one.upstream_list, [])
    self.assertItemsEqual(component_two.upstream_list, [component_one])
    self.assertItemsEqual(component_three.upstream_list, [component_one])
    self.assertItemsEqual(component_four.upstream_list,
                          [component_two, component_three])


if __name__ == '__main__':
  tf.test.main()
