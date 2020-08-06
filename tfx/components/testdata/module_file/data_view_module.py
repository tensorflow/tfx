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
"""A module file used by tests that exercise DataView related components."""
from typing import Any, Dict, Text

import tensorflow as tf
from tfx_bsl.coders import tf_graph_record_decoder


class SimpleDecoder(tf_graph_record_decoder.TFGraphRecordDecoder):
  """Simply converts the input records (1-D dense tensor) to a sparse tensor."""

  def __init__(self):
    super(SimpleDecoder, self).__init__(name="SimpleDecoder")

  def _decode_record_internal(self, record: tf.Tensor) -> Dict[Text, Any]:
    indices = tf.transpose(
        tf.stack([
            tf.range(tf.size(record), dtype=tf.int64),
            tf.zeros(tf.size(record), dtype=tf.int64)
        ]))

    return {
        "sparse_tensor":
            tf.SparseTensor(
                values=record,
                indices=indices,
                dense_shape=[tf.size(record), 1])
    }


def create_simple_decoder() -> tf_graph_record_decoder.TFGraphRecordDecoder:
  return SimpleDecoder()
