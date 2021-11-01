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
"""Tests for tfx.orchestration.kubeflow.kubeflow_metadata_adapter."""

import os

import tensorflow as tf
from tfx.orchestration import data_types
from tfx.orchestration.kubeflow import kubeflow_metadata_adapter

from ml_metadata.proto import metadata_store_pb2


class KubeflowMetadataAdapterTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._connection_config = metadata_store_pb2.ConnectionConfig()
    self._connection_config.sqlite.SetInParent()
    self._pipeline_info = data_types.PipelineInfo(
        pipeline_name='fake_pipeline_name',
        pipeline_root='/fake_pipeline_root',
        run_id='fake_run_id')
    self._pipeline_info2 = data_types.PipelineInfo(
        pipeline_name='fake_pipeline_name',
        pipeline_root='/fake_pipeline_root',
        run_id='fake_run_id2')
    self._component_info = data_types.ComponentInfo(
        component_type='fake.component.type',
        component_id='fake_component_id',
        pipeline_info=self._pipeline_info)
    self._component_info2 = data_types.ComponentInfo(
        component_type='fake.component.type',
        component_id='fake_component_id',
        pipeline_info=self._pipeline_info2)

  def testPrepareExecution(self):
    with kubeflow_metadata_adapter.KubeflowMetadataAdapter(
        connection_config=self._connection_config) as m:
      contexts = m.register_pipeline_contexts_if_not_exists(self._pipeline_info)
      exec_properties = {'arg_one': 1}
      os.environ['KFP_POD_NAME'] = 'fake_pod_name'
      m.register_execution(
          exec_properties=exec_properties,
          pipeline_info=self._pipeline_info,
          component_info=self._component_info,
          contexts=contexts)
      [execution] = m.store.get_executions_by_context(contexts[0].id)
      # Skip verifying time sensitive fields.
      execution.ClearField('type_id')
      execution.ClearField('create_time_since_epoch')
      execution.ClearField('last_update_time_since_epoch')
      self.assertProtoEquals(
          """
        id: 1
        last_known_state: RUNNING
        properties {
          key: "state"
          value {
            string_value: "new"
          }
        }
        properties {
          key: "pipeline_name"
          value {
            string_value: "fake_pipeline_name"
          }
        }
        properties {
          key: "pipeline_root"
          value {
            string_value: "/fake_pipeline_root"
          }
        }
        properties {
          key: "run_id"
          value {
            string_value: "fake_run_id"
          }
        }
        properties {
          key: "component_id"
          value {
            string_value: "fake_component_id"
          }
        }
        properties {
          key: "arg_one"
          value {
            string_value: "1"
          }
        }
        properties {
          key: "kfp_pod_name"
          value {
            string_value: "fake_pod_name"
          }
        }""", execution)

  def testIsEligiblePreviousExecution(self):
    with kubeflow_metadata_adapter.KubeflowMetadataAdapter(
        connection_config=self._connection_config) as m:
      contexts_one = m.register_pipeline_contexts_if_not_exists(
          self._pipeline_info)
      contexts_two = m.register_pipeline_contexts_if_not_exists(
          self._pipeline_info2)
      exec_properties = {'arg_one': 1}
      os.environ['KFP_POD_NAME'] = 'fake_pod_name1'
      m.register_execution(
          exec_properties=exec_properties,
          pipeline_info=self._pipeline_info,
          component_info=self._component_info,
          contexts=contexts_one)
      os.environ['KFP_POD_NAME'] = 'fake_pod_name2'
      m.register_execution(
          exec_properties=exec_properties,
          pipeline_info=self._pipeline_info2,
          component_info=self._component_info2,
          contexts=contexts_two)
      [execution1,
       execution2] = m.store.get_executions_by_context(contexts_one[0].id)
      self.assertTrue(m._is_eligible_previous_execution(execution1, execution2))


if __name__ == '__main__':
  tf.test.main()
