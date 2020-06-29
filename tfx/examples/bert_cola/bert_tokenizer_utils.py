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
  """ Bert Tokenizer built ontop of tensorflow_text.BertTokenizer"""

  def __init__(self, vocab_dir):
    self.vocab_dir = vocab_dir
    self._find_special_token()

  def _find_special_token(self):
    """Find the special token ID's for [CLS] [PAD] [SEP]"""
    f = open(self.vocab_dir, 'r')
    self._sep_id = None
    self._cls_id = None
    self._pad_id = None
    lines = f.read().split('\n')
    for i, line in enumerate(lines):
      if line == '[PAD]':
        self._pad_id = tf.constant(i, dtype=tf.int64)
      elif line == '[CLS]':
        self._cls_id = tf.constant(i, dtype=tf.int64)
      elif line == '[SEP]':
        self._sep_id = tf.constant(i, dtype=tf.int64)
      if self._pad_id is not None \
        and self._cls_id is not None \
        and self._sep_id is not None:
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
    word_id = word_id.to_tensor(default_value=self._pad_id)
    if add_cls:
      cls_token = tf.fill([tf.shape(sequence)[0], 1], self._cls_id)
      word_id = word_id[:, :max_len-1]
      word_id = tf.concat([cls_token, word_id], 1)

    if add_sep:
      sep_token = tf.fill([tf.shape(sequence)[0], 1], self._sep_id)
      word_id = word_id[:, :max_len-1]
      word_id = tf.concat([word_id, sep_token], 1)

    word_id = tf.pad(
        word_id,
        [[0, 0], [0, max_len]],
        constant_values=self._pad_id)

    word_id = tf.slice(word_id, [0, 0], [-1, max_len])

    # Mask to distinguish padded values.
    input_mask = tf.cast(word_id > 0, tf.int64)
    # Mask to distinguish two sentences. In this case, just one sentence.
    segment_id = tf.fill(
        tf.shape(input_mask),
        tf.constant(0, dtype=tf.int64))

    return word_id, input_mask, segment_id
