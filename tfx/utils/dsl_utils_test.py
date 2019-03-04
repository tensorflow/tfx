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
"""Tests for tfx.utils.dsl_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Standard Imports

import tensorflow as tf
from tfx.utils import dsl_utils


class DslUtilsTest(tf.test.TestCase):

  def csv_input(self):
    [csv] = dsl_utils.csv_input(uri='path')
    self.assertEqual('ExternalPath', csv.type_name)
    self.assertEqual('path', csv.uri)


if __name__ == '__main__':
  tf.test.main()
