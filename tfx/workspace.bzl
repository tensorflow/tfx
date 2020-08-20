# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TFX external dependencies that can be loaded in WORKSPACE files.
"""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")

def tf_extended_workspace():
    """All TFX external dependencies."""
    tf_workspace(
        path_prefix = "",
        tf_repo_name = "org_tensorflow",
    )

    # This condition switch is ON if bazel is invoked with
    # --use_master_branch flag.
    config_setting(
        name = "use_master_branch",
        values = {
            "use_master_branch": "true",
        },
    )

    # LINT.IfChange
    # Fetch MLMD repo from GitHub. Do not modify the BEGIN and END comment
    # below (used for copybara replace).
    # BEGIN_MLMD_REPO
    git_repository(
        name = "com_github_google_ml_metadata",
        tag = select({
            ":use_master_branch": None,
            "//conditions:default": "v0.23.0",
        }),
        branch = select({
            ":use_master_branch": "master",
            "//conditions:default": None,
        }),
        remote = "https://github.com/google/ml-metadata.git",
    )
    # END_MLMD_REPO
    # LINT.ThenChange(../../dependencies.py)
