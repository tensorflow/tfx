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
"""Tests for tfx.utils.doc_controls."""

import tensorflow as tf

from tfx.utils import doc_controls as tfx_doc_controls
from tensorflow.tools.docs import doc_controls  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top


class DocControlsTest(tf.test.TestCase):

  def testDocControls(self):
    self.assertEqual(tfx_doc_controls.do_not_generate_docs,
                     doc_controls.do_not_generate_docs)
    self.assertEqual(tfx_doc_controls.do_not_doc_in_subclasses,
                     doc_controls.do_not_doc_in_subclasses)

  def testDocumentSuccess(self):
    documented_test_key = tfx_doc_controls.documented('test key', 'test value')
    self.assertEqual(1, len(tfx_doc_controls.EXTRA_DOCS))
    self.assertEqual('test value',
                     tfx_doc_controls.EXTRA_DOCS.get(id(documented_test_key)))


if __name__ == '__main__':
  tf.test.main()
