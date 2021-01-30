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
"""Tests for tfx.examples.ranking.struct2tensor_parsing_utils."""

import itertools
import unittest

import tensorflow as tf
from tfx.examples.ranking import struct2tensor_parsing_utils

from google.protobuf import text_format
from tensorflow_serving.apis import input_pb2


_ELWCS = [
    text_format.Parse(
        """
context {
  features {
    feature {
      key: "ctx.int"  # dot in the feature name is intended.
      value {
        int64_list {
          value: [1, 2]
        }
      }
    }
    feature {
      key: "ctx.float"
      value {
        float_list {
          value: [1.0, 2.0]
        }
      }
    }
    feature {
      key: "ctx.bytes"
      value {
        bytes_list {
          value: []
        }
      }
    }
  }
}
examples {
  features {
    feature {
      key: "example_int"
      value {
        int64_list {
          value: [11]
        }
      }
    }
    feature {
      key: "example_float"
      value {
        float_list {
          value: [11.0, 12.0]
        }
      }
    }
    feature {
      key: "example_bytes"
      value {
        bytes_list {
          value: ["u", "v"]
        }
      }
    }
  }
}
examples {
  features {
    feature {
      key: "example_int"
      value {
        int64_list {
          value: [22]
        }
      }
    }
    # example_float is not present.
    feature {
      key: "example_bytes"
      value {
        bytes_list {
          value: ["w"]
        }
      }
    }
  }
}
""", input_pb2.ExampleListWithContext()).SerializeToString(),
    text_format.Parse("""
context {
  features {
    feature {
      key: "ctx.int"
      value {
        int64_list {
          value: [3]
        }
      }
    }
    feature {
      key: "ctx.float"
      value {
        float_list {
          value: [3.0]
        }
      }
    }
    feature {
      key: "ctx.bytes"
      value {
        bytes_list {
          value: ["c"]
        }
      }
    }
  }
}
examples {
  features {
    feature {
      key: "example_int"
      value {
        int64_list {
          value: [33]
        }
      }
    }
    feature {
      key: "example_float"
      value {
        float_list {
          value: [14.0, 15.0]
        }
      }
    }
    feature {
      key: "example_bytes"
      value {
        bytes_list {
          value: ["x", "y", "z"]
        }
      }
    }
  }
}
""", input_pb2.ExampleListWithContext()).SerializeToString()
]


@unittest.skipIf(tf.__version__ < '2', reason='TF 1.x not supported.')
class ELWCDecoderTest(tf.test.TestCase):

  def testAllDTypes(self):
    context_features = [
        struct2tensor_parsing_utils.Feature('ctx.int', tf.int64),
        struct2tensor_parsing_utils.Feature('ctx.float', tf.float32),
        struct2tensor_parsing_utils.Feature('ctx.bytes', tf.string),
    ]
    example_features = [
        struct2tensor_parsing_utils.Feature('example_int', tf.int64),
        struct2tensor_parsing_utils.Feature('example_float', tf.float32),
        struct2tensor_parsing_utils.Feature('example_bytes', tf.string),
    ]
    decoder = struct2tensor_parsing_utils.ELWCDecoder(
        'test_decoder', context_features, example_features,
        size_feature_name=None, label_feature=None)

    result = decoder.decode_record(tf.convert_to_tensor(_ELWCS))
    self.assertLen(result, len(context_features) + len(example_features))
    for f in itertools.chain(context_features, example_features):
      self.assertIn(f.name, result)
      self.assertIsInstance(result[f.name], tf.RaggedTensor)

    expected = {
        'ctx.int': [[1, 2], [3]],
        'ctx.float': [[1.0, 2.0], [3.0]],
        'ctx.bytes': [[], [b'c']],
        'example_int': [[[11], [22]], [[33]]],
        'example_float': [[[11.0, 12.0], []], [[14.0, 15.0]]],
        'example_bytes': [[[b'u', b'v'], [b'w']], [[b'x', b'y', b'z']]],
    }
    self.assertEqual({k: v.to_list() for k, v in result.items()}, expected)

  def testDefaultFilling(self):
    context_features = [
        struct2tensor_parsing_utils.Feature('ctx.bytes', tf.string,
                                            default_value=b'g', length=1),
    ]
    example_features = [
        struct2tensor_parsing_utils.Feature('example_float', tf.float32,
                                            default_value=-1.0, length=2),
    ]
    decoder = struct2tensor_parsing_utils.ELWCDecoder(
        'test_decoder', context_features, example_features,
        size_feature_name=None, label_feature=None)

    result = decoder.decode_record(tf.convert_to_tensor(_ELWCS))
    self.assertLen(result, len(context_features) + len(example_features))
    for f in itertools.chain(context_features, example_features):
      self.assertIn(f.name, result)
      self.assertIsInstance(result[f.name], tf.RaggedTensor)

    expected = {
        'ctx.bytes': [[b'g'], [b'c']],
        'example_float': [[[11.0, 12.0], [-1.0, -1.0]], [[14.0, 15.0]]],
    }
    self.assertEqual({k: v.to_list() for k, v in result.items()}, expected)

  def testLabelFeature(self):
    decoder = struct2tensor_parsing_utils.ELWCDecoder(
        'test_decoder', [], [],
        size_feature_name=None,
        label_feature=struct2tensor_parsing_utils.Feature(
            'example_int', tf.int64))
    result = decoder.decode_record(tf.convert_to_tensor(_ELWCS))

    self.assertLen(result, 1)
    self.assertEqual(result['example_int'].to_list(), [[11.0, 22.0], [33.0]])

  def testSizeFeature(self):
    decoder = struct2tensor_parsing_utils.ELWCDecoder(
        'test_decoder', [], [],
        size_feature_name='example_list_size')
    result = decoder.decode_record(tf.convert_to_tensor(_ELWCS))
    self.assertLen(result, 1)
    self.assertEqual(result['example_list_size'].to_list(), [[2], [1]])


if __name__ == '__main__':
  tf.test.main()
