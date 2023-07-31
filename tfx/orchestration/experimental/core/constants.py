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
"""Constants shared across modules."""

EXECUTION_ERROR_CODE_KEY = '__execution_error_code__'
EXECUTION_ERROR_MSG_KEY = '__execution_error_msg__'
EXECUTION_START_TIME_CUSTOM_PROPERTY_KEY = '__execution_start_time__'
# LINT.IfChange(backfill_token)
BACKFILL_TOKEN_CUSTOM_PROPERTY_KEY = '__backfill_token__'
# LINT.ThenChange()

IMPORTER_NODE_TYPE = 'tfx.dsl.components.common.importer.Importer'
RESOLVER_NODE_TYPE = 'tfx.dsl.components.common.resolver.Resolver'
MANUAL_NODE_TYPE = 'tfx.dsl.components.common.manual_node.ManualNode'
SUBPIPELINE_NODE_TYPE = 'tfx.orchestration.pipeline.Pipeline'
SUBPIPELINE_BEGIN_NODE_TYPE = 'tfx.orchestration.pipeline.Pipeline_begin'
SUBPIPELINE_END_NODE_TYPE = 'tfx.orchestration.pipeline.Pipeline_end'
