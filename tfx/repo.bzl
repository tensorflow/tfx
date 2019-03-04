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

""" TensorFlow Http Archive

Modified http_archive that allows us to override the TensorFlow commit that is
downloaded by setting an environment variable. This override is to be used for
testing purposes.

Add the following to your Bazel build command in order to override the
TensorFlow revision.

build: --action_env TF_REVISION="<git commit hash>"

  * `TF_REVISION`: tensorflow revision override (git commit hash)
"""

_TF_REVISION = "TF_REVISION"

def _tensorflow_http_archive(ctx):
    git_commit = ctx.attr.git_commit
    sha256 = ctx.attr.sha256

    override_git_commit = ctx.os.environ.get(_TF_REVISION)
    if override_git_commit:
        sha256 = ""
        git_commit = override_git_commit

    strip_prefix = "tensorflow-%s" % git_commit
    urls = [
        "https://mirror.bazel.build/github.com/tensorflow/tensorflow/archive/%s.tar.gz" % git_commit,
        "https://github.com/tensorflow/tensorflow/archive/%s.tar.gz" % git_commit,
    ]
    ctx.download_and_extract(
        urls,
        "",
        sha256,
        "",
        strip_prefix,
    )

tensorflow_http_archive = repository_rule(
    implementation = _tensorflow_http_archive,
    attrs = {
        "git_commit": attr.string(mandatory = True),
        "sha256": attr.string(mandatory = True),
    },
)
