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
"""Tests for tfx.utils.proto_utils."""

import tensorflow as tf
from tfx.utils import proto_utils
from tfx.utils.testdata import foo_pb2


class ProtoUtilsTest(tf.test.TestCase):

  def test_gather_file_descriptors(self):
    fd_names = set()
    for fd in proto_utils.gather_file_descriptors(foo_pb2.Foo.DESCRIPTOR):
      fd_names.add(fd.name)
    self.assertEqual(
        fd_names, {
            'tfx/utils/testdata/bar.proto',
            'tfx/utils/testdata/foo.proto'
        })


if __name__ == '__main__':
  tf.test.main()
