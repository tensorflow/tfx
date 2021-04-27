# Lint as: python3
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
r"""Script to generate api_docs for tfx.

# How to run

Install tensorflow_docs (if necessary):

```
pip install git+https://github.com/tensorflow/docs
```

Run the script:

```shell
python build_docs.py
```

Note:
  If duplicate or spurious docs are generated, consider
  denylisting them via the `private_map` argument below. Or
  `api_generator.doc_controls`
"""
from absl import app
from absl import flags
import tensorflow_docs.api_generator as api_generator
from tensorflow_docs.api_generator import generate_lib
from tfx import v1 as tfx
from tfx import version
from tfx.utils import doc_controls

from google.protobuf.reflection import GeneratedProtocolMessageType

GITHUB_URL_PREFIX = ("https://github.com/tensorflow/tfx/blob/{}/tfx".format(
    version.__version__))

flags.DEFINE_string("output_dir", "/tmp/tfx_api", "Where to output the docs")
flags.DEFINE_string(
    "code_url_prefix",
    GITHUB_URL_PREFIX,
    "The url prefix for links to code.")
flags.DEFINE_bool("search_hints", True,
                  "Include metadata search hints in the generated files")
flags.DEFINE_string("site_path", "tfx/api_docs/python",
                    "Path prefix in the _toc.yaml")
flags.DEFINE_bool("gen_report", False,
                  ("Generate an API report containing the health of the"
                   "docstrings of the public API."))

FLAGS = flags.FLAGS


def ignore_test_objects(path, parent, children):
  """Removes "test" and "example" modules. These are not part of the public api.

  Args:
    path: A tuple of name parts forming the attribute-lookup path to this
      object. For `tf.keras.layers.Dense` path is:
        ("tf","keras","layers","Dense")
    parent: The parent object.
    children: A list of (name, value) pairs. The attributes of the patent.

  Returns:
    A filtered list of children `(name, value)` pairs. With all test modules
    removed.
  """
  del path
  del parent
  new_children = []
  for (name, obj) in children:
    if name.endswith("_test"):
      continue
    if name.startswith("test_"):
      continue

    new_children.append((name, obj))
  return new_children


def ignore_proto_method(path, parent, children):
  """Remove all the proto inherited methods.

  Args:
    path: A tuple of name parts forming the attribute-lookup path to this
      object. For `tf.keras.layers.Dense` path is:
        ("tf","keras","layers","Dense")
    parent: The parent object.
    children: A list of (name, value) pairs. The attributes of the patent.

  Returns:
    A filtered list of children `(name, value)` pairs. With all proto methods
    removed.
  """
  del path
  new_children = []
  if not isinstance(parent, GeneratedProtocolMessageType):
    return children
  new_children = []
  for (name, obj) in children:
    if "function" in str(obj.__class__):
      continue
    new_children.append((name, obj))
  return new_children


def main(_):

  doc_generator = generate_lib.DocGenerator(
      root_title="TFX",
      # TODO(b/181877171): change to 'tfx.v1' and update related _book.yaml.
      py_modules=[("tfx", tfx)],
      code_url_prefix=FLAGS.code_url_prefix,
      search_hints=FLAGS.search_hints,
      site_path=FLAGS.site_path,
      gen_report=FLAGS.gen_report,
      private_map={},
      # local_definitions_filter ensures that shared modules are only
      # documented in the location that defines them, instead of every location
      # that imports them.
      callbacks=[
          api_generator.public_api.explicit_package_contents_filter,
          ignore_test_objects, ignore_proto_method
      ],
      extra_docs=doc_controls.EXTRA_DOCS)
  doc_generator.build(output_dir=FLAGS.output_dir)


if __name__ == "__main__":
  app.run(main)
