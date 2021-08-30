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
"""Tests for presto_component.component."""

import tensorflow as tf

from tfx.examples.custom_components.presto_example_gen.presto_component import component
from tfx.examples.custom_components.presto_example_gen.proto import presto_config_pb2
from tfx.proto import example_gen_pb2
from tfx.types import standard_artifacts
from tfx.utils import proto_utils


class ComponentTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.conn_config = presto_config_pb2.PrestoConnConfig(
        host='localhost', port=8080)

  def _extract_conn_config(self, custom_config):
    unpacked_custom_config = example_gen_pb2.CustomConfig()
    proto_utils.json_to_proto(custom_config, unpacked_custom_config)

    conn_config = presto_config_pb2.PrestoConnConfig()
    unpacked_custom_config.custom_config.Unpack(conn_config)
    return conn_config

  def testConstruct(self):
    presto_example_gen = component.PrestoExampleGen(
        self.conn_config, query='query')
    self.assertEqual(
        self.conn_config,
        self._extract_conn_config(
            presto_example_gen.exec_properties['custom_config']))
    self.assertEqual(standard_artifacts.Examples.TYPE_NAME,
                     presto_example_gen.outputs['examples'].type_name)

  def testBadConstruction(self):
    empty_config = presto_config_pb2.PrestoConnConfig()
    self.assertRaises(
        RuntimeError,
        component.PrestoExampleGen,
        conn_config=empty_config,
        query='')

    port_only_config = presto_config_pb2.PrestoConnConfig(port=8080)
    self.assertRaises(
        RuntimeError,
        component.PrestoExampleGen,
        conn_config=port_only_config,
        query='')


if __name__ == '__main__':
  tf.test.main()
