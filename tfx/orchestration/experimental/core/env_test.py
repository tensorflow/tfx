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

import tensorflow as tf
from tfx.orchestration.experimental.core import env
from tfx.orchestration.experimental.core import test_utils


class _TestEnv(env.Env):

  def get_orchestration_options(self, pipeline):
    raise NotImplementedError()

  def get_base_dir(self):
    raise NotImplementedError()

  def max_mlmd_str_value_length(self):
    raise NotImplementedError()

  def concurrent_pipeline_runs_enabled(self):
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
