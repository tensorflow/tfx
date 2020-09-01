# Lint as: python2, python3
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
"""Tests for tfx.dsl.compiler.placeholder_utils."""

# TODO(b/149535307): Remove __future__ imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tfx.dsl.compiler import placeholder_utils
from tfx.proto import infra_validator_pb2
from tfx.proto.orchestration import placeholder_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts

from google.protobuf import descriptor_pb2
from google.protobuf import text_format

# Concatenate the URI of `examples` input artifact's `train` split with /1
TEST_PLACEHOLDER_EXPRESSION_1 = """
operator {
  concat_op {
    expressions {
      operator {
        artifact_uri_op {
          expression {
            operator {
              index_op{
                expression {
                  placeholder {
                    type: INPUT_ARTIFACT
                    key: "examples"
                  }
                }
                index: 0
              }
            }
          }
          split: "train"
        }
      }
    }
    expressions{
      primitive_value {
        string_value: "/"
      }
    }
    expressions{
      primitive_value {
        int_value: 1
      }
    }
  }
}
"""

# Access a proto field
TEST_PLACEHOLDER_EXPRESSION_2 = """
operator {
  proto_op {
    expression {
      placeholder {
        type: EXEC_PROPERTY
        key: "proto_property"
      }
    }
    proto_schema {
      message_type: "tfx.components.infra_validator.ServingSpec"
    }
    proto_field_path: "tensorflow_serving.tags"
  }
}
"""

# Access a specific index of an exec property list
TEST_PLACEHOLDER_EXPRESSION_3 = """
operator {
  index_op {
    expression {
      placeholder {
        type: EXEC_PROPERTY
        key: "double_list_property"
      }
    }
    index: 1
  }
}
"""


class PlaceholderUtilsTest(tf.test.TestCase):

  def setUp(self):
    super(PlaceholderUtilsTest, self).setUp()
    examples = [standard_artifacts.Examples()]
    examples[0].uri = "/tmp"
    examples[0].split_names = artifact_utils.encode_split_names(
        ["train", "eval"])
    serving_spec = infra_validator_pb2.ServingSpec()
    serving_spec.tensorflow_serving.tags.extend(["latest", "1.15.0-gpu"])
    self._resolution_context = placeholder_utils.ResolutionContext(
        input_dict={
            "model": [standard_artifacts.Model()],
            "examples": examples,
        },
        output_dict={"blessing": [standard_artifacts.ModelBlessing()]},
        exec_properties={
            "proto_property": serving_spec.SerializeToString(),
            "double_list_property": [0.7, 0.8, 0.9],
        })

  def testConcatArtifactUri(self):
    pb = text_format.Parse(TEST_PLACEHOLDER_EXPRESSION_1,
                           placeholder_pb2.PlaceholderExpression())
    self.assertEqual(
        placeholder_utils.resolve_placeholder_expression(
            pb, self._resolution_context), "/tmp/train/1")

  def testProtoExecProperty(self):
    pb = text_format.Parse(TEST_PLACEHOLDER_EXPRESSION_2,
                           placeholder_pb2.PlaceholderExpression())

    # Prepare FileDescriptorSet
    fd = descriptor_pb2.FileDescriptorProto()
    infra_validator_pb2.ServingSpec().DESCRIPTOR.file.CopyToProto(fd)
    pb.operator.proto_op.proto_schema.file_descriptors.file.append(fd)

    self.assertEqual(
        placeholder_utils.resolve_placeholder_expression(
            pb, self._resolution_context), ["latest", "1.15.0-gpu"])

  def testExecPropertyIndex(self):
    pb = text_format.Parse(TEST_PLACEHOLDER_EXPRESSION_3,
                           placeholder_pb2.PlaceholderExpression())

    self.assertEqual(
        placeholder_utils.resolve_placeholder_expression(
            pb, self._resolution_context), 0.8)

if __name__ == "__main__":
  tf.test.main()
