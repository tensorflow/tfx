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

"""TFX external dependencies that can be loaded in WORKSPACE files."""

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")

# Sanitize a dependency so that it works correctly from code that includes
# TFX as a submodule.
def clean_dep(dep):
    return str(Label(dep))

def tfx_workspace():
    """All TFX external dependencies."""
    tf_workspace(
        path_prefix = "",
        tf_repo_name = "org_tensorflow",
    )
