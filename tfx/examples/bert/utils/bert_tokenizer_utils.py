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
from tensorflow.python.eager.context import eager_mode
import tensorflow_hub as hub

_CLS = '[CLS]'
_PAD = '[PAD]'
_SEP = '[SEP]'

class SpecialBertTokenizer():
  """ Bert Tokenizer built ontop of tensorflow_text.BertTokenizer"""

  def __init__(self, model_link):
    self._model_link = model_link
    self._find_special_tokens()

  def _find_special_tokens(self):
    """Find the special token ID's for [CLS] [PAD] [SEP]

    Since each Bert model is trained on different vocaburary, it's important
    to find the special token index pertaining to that model.
    Since in Transform, tensorflow_hub.KerasLayer loads a symbolic tensor, turn
    on eager mode to get the actual vocab_file location.
    """

    with eager_mode():
      model = hub.KerasLayer(self._model_link)
      vocab = model.resolved_object.vocab_file.asset_path.numpy()
      f = open(vocab, 'r')
      self._sep_id = None
      self._cls_id = None
      self._pad_id = None
      lines = f.read().split('\n')
      for i, line in enumerate(lines):
        if line == _PAD:
          self._pad_id = i
        elif line == _CLS:
          self._cls_id = i
        elif line == _SEP:
          self._sep_id = i
        if (self._pad_id is not None and
            self._cls_id is not None and
            self._sep_id is not None):
          break

  def tokenize_single_sentence(
      self,
      sequence,
      max_len=128,
      add_cls=True,
      add_sep=True,
      add_pad=True):
    """Tokenize a single sentence according to the vocab used by the Bert model.

    Add special tokens according to config.

    Args:
      sequence: Tensor of shape [batch_size, 1].
      max_len: The number of tokens after padding and truncating.
      add_cls: Whether to add CLS token at the front of each sequence.
      add_sep: Whether to add SEP token at the end of each sequence.
      add_pad: If not add_pad, would return ragged tensor instead of tensors
        input_mask and segment_id would be none.

    Returns:
      word_id: Tokenized sequences [batch_size, max_len].
      input_mask: Mask padded tokens [batch_size, max_len].
      segment_id: Distinguish multiple sequences [batch_size, max_len].
    """
    model = hub.KerasLayer(self._model_link)
    vocab_file_path = model.resolved_object.vocab_file.asset_path
    tokenizer = text.BertTokenizer(vocab_file_path, token_out_type=tf.int64)
    word_id = tokenizer.tokenize(sequence)
    # Tokenizer default puts tokens into array of size 1. merge_dims flattens it
    word_id = word_id.merge_dims(1, 2)
    if add_cls:
      cls_token = tf.fill(
          [tf.shape(sequence)[0], 1],
          tf.constant(self._cls_id, dtype=tf.int64))

      word_id = tf.concat([cls_token, word_id], 1)

    if add_sep:
      sep_token = tf.fill(
          [tf.shape(sequence)[0], 1],
          tf.constant(self._sep_id, dtype=tf.int64))

      word_id = word_id[:, :max_len-1]
      word_id = tf.concat([word_id, sep_token], 1)

    if not add_pad:
      return word_id, None, None

    word_id = word_id.to_tensor(
        default_value=tf.constant(self._pad_id, dtype=tf.int64))

    word_id = tf.pad(
        word_id,
        [[0, 0], [0, max_len]],
        constant_values=tf.constant(self._pad_id, dtype=tf.int64))

    word_id = tf.slice(word_id, [0, 0], [-1, max_len])

    input_mask = tf.cast(tf.not_equal(word_id, self._pad_id), tf.int64)
    segment_id = tf.fill(
        tf.shape(input_mask),
        tf.constant(0, dtype=tf.int64))

    return word_id, input_mask, segment_id

  def tokenize_sentence_pair(
      self,
      sequence_a,
      sequence_b,
      max_len):
    """Tokenize a sequence pair.

    Tokenize each sequence with self.tokenize_single_sentence. Then add CLS
    token in front of the first sequence, add SEP tokens between the two
    sequences and at the end of the second sequence.

    Args:
      sequence_a: [batch_size, 1]
      sequence_b: [batch_size, 1]
      max_len: The length of the concatenated tokenized sentences.

    Returns:
      word_id: Tokenized sequences [batch_size, max_len].
      input_mask: Mask padded tokens [batch_size, max_len].
      segment_id: Distinguish multiple sequences [batch_size, max_len].
    """
    sentence_len = max_len // 2
    word_id_a, _, _ = self.tokenize_single_sentence(
        sequence_a,
        sentence_len,
        True,
        True,
        False
    )

    word_id_b, _, _ = self.tokenize_single_sentence(
        sequence_b,
        sentence_len,
        False,
        True,
        False
    )

    word_id = tf.concat([word_id_a, word_id_b], 1)
    word_id = word_id.to_tensor(
        default_value=tf.constant(self._pad_id, dtype=tf.int64))

    word_id = tf.pad(
        word_id,
        [[0, 0], [0, max_len]],
        constant_values=tf.constant(self._pad_id, dtype=tf.int64))

    word_id = tf.slice(word_id, [0, 0], [-1, max_len])
    input_mask = tf.cast(tf.not_equal(word_id, self._pad_id), tf.int64)
    segment_id = tf.cast(word_id_a < 0, tf.int64)
    segment_id = segment_id.to_tensor(
        default_value=tf.constant(1, dtype=tf.int64))

    segment_id = tf.pad(
        segment_id,
        [[0, 0], [0, max_len]],
        constant_values=tf.constant(1, dtype=tf.int64))

    segment_id = tf.slice(segment_id, [0, 0], [-1, max_len])

    return word_id, input_mask, segment_id

