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
"""TFX external dependencies that can be loaded in WORKSPACE files."""

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")

def _github_archive_url(org, repo, ref):
    return "https://github.com/{0}/{1}/archive/{2}.zip".format(org, repo, ref)

def _tfx_github_archive(ctx):
    if "/" not in ctx.attr.repo:
        fail("repo must be <org_or_username>/<repo_name> format")
    if int(bool(ctx.attr.branch)) + int(bool(ctx.attr.tag)) + int(bool(ctx.attr.commit)) != 1:
        fail("Exactly one of 'branch', 'tag', or 'commit' should be specified.")

    org, repo = ctx.attr.repo.split("/")
    if ctx.os.environ.get("TFX_DEPENDENCY_SELECTOR") == "GIT_MASTER":
        url = _github_archive_url(org, repo, "master")
        skip_prefix = "{0}-master".format(repo)
    else:
        ref = ctx.attr.branch or ctx.attr.tag or ctx.attr.commit
        url = _github_archive_url(org, repo, ref)
        if ctx.attr.tag and ctx.attr.tag.startswith("v"):
            # Github archive omit "v" prefix in the tag.
            skip_prefix = "{0}-{1}".format(repo, ref[1:])
        else:
            skip_prefix = "{0}-{1}".format(repo, ref)
    ctx.download_and_extract(
        url,
        output = "",
        stripPrefix = skip_prefix,
    )

# Repository rule that is similar to git_repository, but uses master branch
# regardless of given parameter if TFX_DEPENDENCY_SELECTOR environment
# variable is set to "GIT_MASTER". Normally this environment variable is set when
# running setup.py (e.g. TFX_DEPENDENCY_SELECTOR=blahblah pip wheel .) where the
# variable is also used for baking proper dependency constraints.
#
# Usage:
#   tfx_github_archive(
#       name = "org_tensorflow",
#       repo = "tensorflow/tensorflow",  # github tensorflow repository
#       tag = "v2.3.0",
#   )
tfx_github_archive = repository_rule(
    attrs = {
        "repo": attr.string(mandatory = True),
        "branch": attr.string(),
        "commit": attr.string(),
        "tag": attr.string(),
    },
    environ = [
        "TFX_DEPENDENCY_SELECTOR",
    ],
    implementation = _tfx_github_archive,
)

def tfx_workspace():
    """All TFX external dependencies."""
    tf_workspace(
        path_prefix = "",
        tf_repo_name = "org_tensorflow",
    )

    # Fetch MLMD repo from GitHub.
    tfx_github_archive(
        name = "com_github_google_ml_metadata",
        repo = "google/ml-metadata",
        # LINT.IfChange
        tag = "v1.3.0",
        # LINT.ThenChange(//tfx/dependencies.py)
    )
