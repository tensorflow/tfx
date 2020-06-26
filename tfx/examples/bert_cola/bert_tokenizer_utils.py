# Lint as: python2, python3
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
"""Prepressing using tensorflow_text BertTokenizer"""

import tensorflow as tf
import tensorflow_text as text

class SpecialBertTokenizer():

  def __init__(self, vocab_dir):
    self.vocab_dir = vocab_dir
    self._find_special_token()

  def _find_special_token(self):
    """Find the special token ID's for [CLS] [PAD] [SEP]"""
    f = open(self.vocab_dir, 'r')
    self._SEP_ID = None
    self._CLS_ID = None
    self._PAD_ID = None
    lines = f.read().split('\n')
    for i, line in lines:
      if line == '[PAD]':
        self._PAD_ID = tf.constant(i, dtype=tf.int64)
      elif line == '[CLS]':
        self._CLS_ID = tf.constant(i, dtype=tf.int64)
      elif line == '[SEP]':
        self._SEP_ID = tf.constant(i, dtype=tf.int64)
      if self._PAD_ID is not None \
        and self._CLS_ID is not None \
        and self._SEP_ID is not None:
        break

  def tokenize_single_sentence(
    self,
    sequence,
    max_len=128,
    add_cls=True,
    add_sep=True):
    """Tokenize a single sentence according to the vocab.txt provided.
    Add special tokens according to config.
    """

    tokenizer = text.BertTokenizer(self.vocab_dir, token_out_type=tf.int64)
    word_id = tokenizer.tokenize(sequence)
    word_id = word_id.merge_dims(1, 2)[:, :max_len]
    word_id = word_id.to_tensor(default_value=self._PAD_ID)
    if add_cls:
      clsToken = tf.fill([tf.shape(sequence)[0], 1], self._CLS_ID)
      word_id = word_id[:, :max_len-1]
      word_id = tf.concat([clsToken, word_id], 1)

    if add_sep:
      sepToken = tf.fill([tf.shape(sequence)[0], 1], self._SEP_ID)
      word_id = word_id[:, :max_len-1]
      word_id = tf.concat([word_id, sepToken], 1)

    word_id = tf.pad(
      word_id,
      [[0, 0], [0, max_len]],
      constant_values=self._PAD_ID)

    word_id = tf.slice(word_id, [0, 0], [-1, max_len])

    # Mask to distinguish padded values.
    input_mask = tf.cast(word_id > 0, tf.int64)
    # Mask to distinguish two sentences. In this case, just one sentence.
    segment_id = tf.fill(
      tf.shape(input_mask),
      tf.constant(0, dtype=tf.int64))

    return word_id, input_mask, segment_id
