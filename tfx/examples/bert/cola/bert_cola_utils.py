# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Python source file include cola pipeline functions and necessary utils."""

from typing import List

import tensorflow as tf
import tensorflow_data_validation as tfdv
import tensorflow_hub as hub
import tensorflow_transform as tft

from tfx import v1 as tfx
from tfx.components.transform import stats_options_util
from tfx.examples.bert.utils.bert_models import build_and_compile_bert_classifier
from tfx.examples.bert.utils.bert_tokenizer_utils import BertPreprocessor


from tfx_bsl.public import tfxio

from google.protobuf import text_format

_BERT_VOCAB = 'bert_vocab'
_INPUT_WORD_IDS = 'input_word_ids'
_INPUT_MASK = 'input_mask'
_SEGMENT_IDS = 'segment_ids'
_TRAIN_BATCH_SIZE = 16
_EVAL_BATCH_SIZE = 16
_FEATURE_KEY = 'sentence'
_LABEL_KEY = 'label'
_BERT_LINK = 'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/2'
_MAX_LEN = 256
_EPOCHS = 1


def _tokenize(feature):
  """Tokenize the two sentences and insert appropriate tokens."""
  processor = BertPreprocessor(_BERT_LINK)
  vocab = processor.get_vocab_name()
  # Annotate asset provides the mapping between the name (_BERT_VOCAB) and the
  # path within the StatsOptions object passed to TFDV (
  # https://github.com/tensorflow/data-validation/blob/master/tensorflow_data_validation/statistics/stats_options.py).
  # This vocab can then be used to compute NLP statistics (see the description
  # of the stats_options_updater_fn below_.
  tft.annotate_asset(_BERT_VOCAB, vocab.decode())
  return processor.tokenize_single_sentence_pad(
      tf.reshape(feature, [-1]), max_len=_MAX_LEN)


def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed feature Tensors.
  """
  input_word_ids, input_mask, segment_ids = _tokenize(inputs[_FEATURE_KEY])
  return {
      _LABEL_KEY: inputs[_LABEL_KEY],
      _INPUT_WORD_IDS: input_word_ids,
      _INPUT_MASK: input_mask,
      _SEGMENT_IDS: segment_ids
  }


def stats_options_updater_fn(
    stats_type: stats_options_util.StatsType,
    stats_options: tfdv.StatsOptions) -> tfdv.StatsOptions:
  """Update transform stats.

  This function is called by the Transform component before it computes
  pre-transform or post-transform statistics. It takes as input a stats_type,
  which indicates whether this call is intended for pre-transform or
  post-transform statistics. It also takes as argument the StatsOptions that
  are to be (optionally) modified before being passed onto TDFV.

  Args:
    stats_type: The type of statistics that are to be computed (pre-transform or
      post-transform).
    stats_options: The configuration to pass to TFDV for computing the desired
      statistics.

  Returns:
    An updated StatsOptions object.
  """
  if stats_type == stats_options_util.StatsType.POST_TRANSFORM:
    for f in stats_options.schema.feature:
      if f.name == _INPUT_WORD_IDS:
        # Here we extend the schema for the input_word_ids feature to enable
        # NLP statistics to be computed. We pass the vocabulary (_BERT_VOCAB)
        # that was used in tokenizing this feature, key tokens of interest
        # (e.g. "[CLS]",  "[PAD]", "[SEP]", "[UNK]") and key thresholds to
        # validate. For more information on the field descriptions, see here:
        # https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto
        text_format.Parse(
            """
            vocabulary: "{vocab}"
            coverage: {{
              min_coverage: 1.0
              min_avg_token_length: 3.0
              excluded_string_tokens: ["[CLS]", "[PAD]", "[SEP]"]
              oov_string_tokens: ["[UNK]"]
             }}
             token_constraints {{
               string_value: "[CLS]"
               min_per_sequence: 1
               max_per_sequence: 1
               min_fraction_of_sequences: 1
               max_fraction_of_sequences: 1
             }}
             token_constraints {{
               string_value: "[PAD]"
               min_per_sequence: 0
               max_per_sequence: {max_pad_per_seq}
               min_fraction_of_sequences: 0
               max_fraction_of_sequences: 1
             }}
             token_constraints {{
               string_value: "[SEP]"
               min_per_sequence: 1
               max_per_sequence: 1
               min_fraction_of_sequences: 1
               max_fraction_of_sequences: 1
             }}
             token_constraints {{
               string_value: "[UNK]"
               min_per_sequence: 0
               max_per_sequence: {max_unk_per_seq}
               min_fraction_of_sequences: 0
               max_fraction_of_sequences: 1
             }}
             sequence_length_constraints {{
               excluded_string_value: ["[PAD]"]
               min_sequence_length: 3
               max_sequence_length: {max_seq_len}
             }}
            """.format(
                vocab=_BERT_VOCAB,
                max_pad_per_seq=_MAX_LEN - 3,  # [CLS], [SEP], Token
                max_unk_per_seq=_MAX_LEN - 2,  # [CLS], [SEP]
                max_seq_len=_MAX_LEN),
            f.natural_language_domain)
  return stats_options


def _input_fn(file_pattern: List[str],
              data_accessor: tfx.components.DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
  """Generates features and label for tuning/training.

  Args:
    file_pattern: List of paths or patterns of materialized transformed input
      tfrecord files.
    data_accessor: DataAccessor for converting input to RecordBatch.
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  dataset = data_accessor.tf_dataset_factory(
      file_pattern,
      tfxio.TensorFlowDatasetOptions(
          batch_size=batch_size, label_key=_LABEL_KEY),
      tf_transform_output.transformed_metadata.schema)
  dataset = dataset.repeat()
  return dataset.prefetch(tf.data.AUTOTUNE)


def _get_serve_tf_examples_fn(model, tf_transform_output):
  """Returns a function that parses a serialized tf.Example."""

  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    feature_spec = tf_transform_output.raw_feature_spec()
    feature_spec.pop(_LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

    transformed_features = model.tft_layer(parsed_features)

    return model(transformed_features)

  return serve_tf_examples_fn


# TFX Trainer will call this function.
def run_fn(fn_args: tfx.components.FnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  train_dataset = _input_fn(
      fn_args.train_files,
      fn_args.data_accessor,
      tf_transform_output,
      batch_size=_TRAIN_BATCH_SIZE)

  eval_dataset = _input_fn(
      fn_args.eval_files,
      fn_args.data_accessor,
      tf_transform_output,
      batch_size=_EVAL_BATCH_SIZE)

  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
    bert_layer = hub.KerasLayer(_BERT_LINK, trainable=True)
    model = build_and_compile_bert_classifier(bert_layer, _MAX_LEN, 2)

  model.fit(
      train_dataset,
      epochs=_EPOCHS,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps)

  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model,
                                    tf_transform_output).get_concrete_function(
                                        tf.TensorSpec(
                                            shape=[None],
                                            dtype=tf.string,
                                            name='examples')),
  }
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
