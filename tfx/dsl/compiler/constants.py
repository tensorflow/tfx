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
"""Constant values for DSL Compiler."""

# The parameter name for pipeline root in PipelineRuntimeSpec.
PIPELINE_ROOT_PARAMETER_NAME = "pipeline_root"

# The parameter name for pipeline run id in PipelineRuntimeSpec.
PIPELINE_RUN_ID_PARAMETER_NAME = "pipeline_run_id"

# The type name for the contexts representing different pipelines.
PIPELINE_CONTEXT_TYPE_NAME = "pipeline"

# The type name for the contexts representing different pipeline runs.
PIPELINE_RUN_CONTEXT_TYPE_NAME = "pipeline_run"

# The type name for the contexts representing different nodes. Note that a
# context of this type only covers node runs within a pipeline domain but not
# across pipelines.
NODE_CONTEXT_TYPE_NAME = "node"
