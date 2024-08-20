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
"""Settings for pytest."""

import sys

collect_ignore = []
if sys.version_info.major == 2:
  collect_ignore.append(
      'tfx/examples/chicago_taxi_pipeline/taxi_pipeline_kubeflow_test.py')
  collect_ignore.append('tfx/orchestration/kubeflow')
