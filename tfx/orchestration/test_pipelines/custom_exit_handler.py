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
"""Custom component for exit handler."""

from tfx.orchestration.kubeflow import decorators
from tfx.utils import io_utils
import tfx.v1 as tfx


@decorators.exit_handler
def test_exit_handler(final_status: tfx.dsl.components.Parameter[str],
                      file_dir: tfx.dsl.components.Parameter[str]):

  io_utils.write_string_file(file_dir, final_status)
