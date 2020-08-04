# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Definition of execution_spec."""

from dataclasses import dataclass
from typing import List, Text

from tensorflow.core.framework import graph_pb2


@dataclass
class ExecutionSpec:
  """A spec that represents a partitioned subgraph.

  Attributes:
    subgraph: A subgraph's graph_def.
    input_names: A list of subgraph's input node names.
    output_names: A list of subgraph's output node names.
    is_remote_op: A boolean indicating the type of the subgraph
                  (two types: regular subgraph or remote op subgraph).
  """
  subgraph: graph_pb2.GraphDef
  input_names: List[Text]
  output_names: List[Text]
  is_remote_op: bool
