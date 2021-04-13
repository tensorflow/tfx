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
"""Doc control utils."""

try:
  from tensorflow.tools.docs import doc_controls  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
except ModuleNotFoundError:
  doc_controls = None

if doc_controls:

  do_not_doc_in_subclasses = doc_controls.do_not_doc_in_subclasses
  do_not_generate_docs = doc_controls.do_not_generate_docs

else:

  def do_not_doc_in_subclasses(obj):
    return obj

  def do_not_generate_docs(obj):
    return obj
