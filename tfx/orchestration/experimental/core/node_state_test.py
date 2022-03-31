# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Tests for tfx.orchestration.experimental.core.node_state."""

from tfx.orchestration.experimental.core import node_state as nstate
from tfx.orchestration.experimental.core import test_utils
from tfx.utils import status as status_lib


class NodeStateTest(test_utils.TfxTest):

  def test_node_state_update(self):
    node_state = nstate.NodeState()
    self.assertEqual(nstate.NodeState.STARTED, node_state.state)
    self.assertIsNone(node_state.status)

    status = status_lib.Status(code=status_lib.Code.CANCELLED, message='foobar')
    node_state.update(nstate.NodeState.STOPPING, status)
    self.assertEqual(nstate.NodeState.STOPPING, node_state.state)
    self.assertEqual(status, node_state.status)

    node_state.update(nstate.NodeState.STARTING)
    self.assertEqual(nstate.NodeState.STARTING, node_state.state)
    self.assertIsNone(node_state.status)
