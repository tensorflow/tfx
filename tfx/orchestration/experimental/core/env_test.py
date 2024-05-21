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
"""Tests for tfx.orchestration.experimental.core.env."""

from typing import Optional, Sequence

import tensorflow as tf
from tfx.orchestration.experimental.core import env
from tfx.orchestration.experimental.core import test_utils
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import status as status_lib

from ml_metadata.proto import metadata_store_pb2


class _TestEnv(env.Env):

  def get_orchestration_options(self, pipeline):
    raise NotImplementedError()

  def get_base_dir(self):
    raise NotImplementedError()

  def label_and_tag_pipeline_run(
      self, mlmd_handle, pipeline_id, pipeline_run_id, labels, tags
  ):
    raise NotImplementedError()

  def max_mlmd_str_value_length(self):
    raise NotImplementedError()

  def concurrent_pipeline_runs_enabled(self):
    raise NotImplementedError()

  def is_pure_service_node(self, pipeline_state, node_id) -> bool:
    raise NotImplementedError()

  def health_status(self) -> status_lib.Status:
    raise NotImplementedError()

  def set_health_status(self, status: status_lib.Status) -> None:
    raise NotImplementedError()

  def check_if_can_orchestrate(self, pipeline) -> None:
    raise NotImplementedError()

  def prepare_orchestrator_for_pipeline_run(
      self, pipeline: pipeline_pb2.Pipeline
  ):
    raise NotImplementedError()

  def create_sync_or_upsert_async_pipeline_run(
      self,
      owner: str,
      pipeline_name: str,
      execution: metadata_store_pb2.Execution,
      pipeline: pipeline_pb2.Pipeline,
      pipeline_run_metadata: Optional[str] = None,
      base_pipeline_run_id: Optional[str] = None,
  ) -> None:
    raise NotImplementedError()

  def update_pipeline_run_status(
      self,
      owner: str,
      pipeline_name: str,
      pipeline: pipeline_pb2.Pipeline,
      original_execution: metadata_store_pb2.Execution,
      modified_execution: metadata_store_pb2.Execution,
      sub_pipeline_ids: Optional[Sequence[str]] = None,
  ) -> None:
    raise NotImplementedError()

  def should_orchestrate(self, pipeline: pipeline_pb2.Pipeline) -> bool:
    raise NotImplementedError()


class EnvTest(test_utils.TfxTest):

  def test_env_context(self):
    default_env = env.get_env()
    self.assertIsInstance(default_env, env._DefaultEnv)
    test_env = _TestEnv()
    with test_env:
      self.assertIs(env.get_env(), test_env)
    self.assertIs(env.get_env(), default_env)


if __name__ == '__main__':
  tf.test.main()
