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
  do_not_doc_inheritable = doc_controls.do_not_doc_inheritable
  do_not_generate_docs = doc_controls.do_not_generate_docs

else:

  def do_not_doc_in_subclasses(obj):
    return obj

  def do_not_doc_inheritable(obj):
    return obj

  def do_not_generate_docs(obj):
    return obj


EXTRA_DOCS = dict()


def documented(obj, doc):
  """Adds a docstring to typealias by overriding the `__doc__` attribute.

  Note: Overriding `__doc__` is only possible after python 3.7.

  Args:
    obj: Typealias object that needs to be documented.
    doc: Docstring of the typealias. It should follow the standard pystyle
      docstring rules.

  Returns:
    Documented variables.
  """
  if isinstance(obj, int) or obj in [(), None, ""]:
    raise ValueError(f"Can't add docs to singletons: `{obj}`.")
  try:
    obj.__doc__ = doc
  except AttributeError:
    EXTRA_DOCS[id(obj)] = doc
  return obj
