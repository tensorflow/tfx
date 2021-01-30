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

"""A central place for feature name constants.

These names will be shared between the transform and the model.
"""

import tensorflow as tf
from tfx.examples.ranking import struct2tensor_parsing_utils

# Labels are expected to be dense. In case of a batch of ELWCs have different
# number of documents, the shape of the label is [N, D], where N is the batch
# size, D is the maximum number of documents in the batch. If an ELWC in the
# batch has D_0 < D documents, then the value of label at D0 <= d < D must be
# negative to indicate that the document is invalid.
LABEL_PADDING_VALUE = -1

# Names of features in the ELWC.
QUERY_TOKENS = 'query_tokens'
DOCUMENT_TOKENS = 'document_tokens'
LABEL = 'relevance'

# This "feature" does not exist in the data but will be created on the fly.
LIST_SIZE_FEATURE_NAME = 'example_list_size'


def get_features():
  """Defines the context features and example features spec for parsing."""

  context_features = [
      struct2tensor_parsing_utils.Feature(QUERY_TOKENS, tf.string)
  ]

  example_features = [
      struct2tensor_parsing_utils.Feature(DOCUMENT_TOKENS, tf.string)
  ]

  label = struct2tensor_parsing_utils.Feature(LABEL, tf.int64)

  return context_features, example_features, label
