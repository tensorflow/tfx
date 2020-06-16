# Lint as: python2, python3
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

# TODO(b/149535307): Remove __future__ imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# The parameter name for pipeline root in PipelineRuntimeSpec.
PIPELINE_ROOT_PARAMETER_NAME = "pipeline_root"

# The parameter name for pipeline run id in PipelineRuntimeSpec.
PIPELINE_RUN_ID_PARAMETER_NAME = "pipeline_run_id"

# The name for pipeline node's pipeline context.
PIPELINE_CONTEXT_TYPE_NAME = "pipeline"

# The name for pipeline node's pipeline run context.
PIPELINE_RUN_CONTEXT_TYPE_NAME = "pipeline_run"
