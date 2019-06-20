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
"""Tests for tfx.tools.cli.handler.base_handler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from tfx.tools.cli import labels
from tfx.tools.cli.handler import base_handler


class FakeHandler(base_handler.BaseHandler):

  def create_pipeline(self) -> None:
    pass

  def update_pipeline(self) -> None:
    pass

  def list_pipelines(self) -> None:
    pass

  def delete_pipeline(self) -> None:
    pass

  def run_pipeline(self) -> None:
    pass


class BaseHandlerTest(tf.test.TestCase):

  def setUp(self):
    self.engine = 'airflow'
    self.chicago_taxi_pipeline_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    self.pipeline_path = os.path.join(self.chicago_taxi_pipeline_dir,
                                      'taxi_pipeline_simple.py')

  def test_check_pipeline_dsl_path_invalid(self):
    flags_dict = {labels.ENGINE_FLAG: self.engine,
                  labels.PIPELINE_DSL_PATH: 'taxi_pipeline.py'}
    handler = FakeHandler(flags_dict)
    with self.assertRaises(SystemExit) as err:
      handler._check_pipeline_dsl_path()
    self.assertEqual(str(err.exception), 'Invalid pipeline path: {}'
                     .format(flags_dict[labels.PIPELINE_DSL_PATH]))

if __name__ == '__main__':
  tf.test.main()
