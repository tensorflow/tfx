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
"""Tests for tfx.utils.import_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# Standard Imports

import tensorflow as tf
from tfx.utils import import_utils


class ImportUtilsTest(tf.test.TestCase):

  def test_import_class_by_path(self):
    """Test import_class_by_path."""
    class_path = '.'.join(
        [ImportUtilsTest.__module__, ImportUtilsTest.__name__])
    imported_class = import_utils.import_class_by_path(class_path)
    self.assertEqual(ImportUtilsTest, imported_class)


if __name__ == '__main__':
  tf.test.main()
