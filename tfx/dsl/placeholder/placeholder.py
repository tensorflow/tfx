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
"""Placeholders represent not-yet-available values at component authoring time."""

# This is much like an __init__ file in that it only re-exports symbols. But
# for historical reasons, it's not actually in the __init__ file.
# pylint: disable=g-multiple-import,g-importing-member,unused-import,g-bad-import-order,redefined-builtin
from tfx.dsl.placeholder.placeholder_base import Placeholder, Predicate, ListPlaceholder
from tfx.dsl.placeholder.placeholder_base import dirname
from tfx.dsl.placeholder.placeholder_base import logical_not, logical_and, logical_or
from tfx.dsl.placeholder.placeholder_base import join, join_path, make_list
from tfx.dsl.placeholder.placeholder_base import ListSerializationFormat, ProtoSerializationFormat
from tfx.dsl.placeholder.artifact_placeholder import ArtifactPlaceholder, input, output
from tfx.dsl.placeholder.runtime_placeholders import environment_variable, EnvironmentVariablePlaceholder
from tfx.dsl.placeholder.runtime_placeholders import execution_invocation, ExecInvocationPlaceholder
from tfx.dsl.placeholder.runtime_placeholders import exec_property, ExecPropertyPlaceholder
from tfx.dsl.placeholder.runtime_placeholders import runtime_info, RuntimeInfoPlaceholder, RuntimeInfoKeys
from tfx.dsl.placeholder.proto_placeholder import make_proto, MakeProtoPlaceholder
from tfx.types.channel import ChannelWrappedPlaceholder
