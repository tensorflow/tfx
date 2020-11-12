# Lint as: python3
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

import base64

import tensorflow as tf
from tfx.dsl.compiler import placeholder_utils
from tfx.orchestration.portable import data_types
from tfx.proto import infra_validator_pb2
from tfx.proto.orchestration import executor_invocation_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.proto.orchestration import placeholder_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts

from google.protobuf import descriptor_pb2
from google.protobuf import json_format
from google.protobuf import text_format
from ml_metadata.proto import metadata_store_pb2

# Concatenate the URI of `examples` input artifact's `train` split with /1
CONCAT_SPLIT_URI_EXPRESSION = """
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
      value {
        string_value: "/"
      }
    }
    expressions{
      value {
        int_value: 1
      }
    }
  }
}
"""

_WANT_EXEC_INVOCATION = """
execution_properties {
  key: "proto_property"
  value {
    string_value: "{\\n\\"tensorflow_serving\\": {\\n\\"tags\\": [\\n\\"latest\\",\\n\\"1.15.0-gpu\\"\\n]\\n}\\n}"
  }
}
output_metadata_uri: "test_executor_output_uri"
input_dict {
  key: "examples"
  value {
    elements {
      artifact {
        artifact {
          uri: "/tmp"
          properties {
            key: "split_names"
            value {
              string_value: "[\\"train\\", \\"eval\\"]"
            }
          }
        }
        type {
          name: "Examples"
          properties {
            key: "span"
            value: INT
          }
          properties {
            key: "split_names"
            value: STRING
          }
          properties {
            key: "version"
            value: INT
          }
        }
      }
    }
  }
}
input_dict {
  key: "model"
  value {
    elements {
      artifact {
        artifact {
        }
        type {
          name: "Model"
        }
      }
    }
  }
}
output_dict {
  key: "blessing"
  value {
    elements {
      artifact {
        artifact {
        }
        type {
          name: "ModelBlessing"
        }
      }
    }
  }
}
stateful_working_dir: "test_stateful_working_dir"
pipeline_info {
   id: "test_pipeline_id"
}
pipeline_node {
  node_info {
    type {
      name: "infra_validator"
    }
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
        exec_info=data_types.ExecutionInfo(
            input_dict={
                "model": [standard_artifacts.Model()],
                "examples": examples,
            },
            output_dict={"blessing": [standard_artifacts.ModelBlessing()]},
            exec_properties={
                "proto_property":
                    json_format.MessageToJson(
                        message=serving_spec,
                        sort_keys=True,
                        preserving_proto_field_name=True,
                        indent=0)
            },
            execution_output_uri="test_executor_output_uri",
            stateful_working_dir="test_stateful_working_dir",
            pipeline_node=pipeline_pb2.PipelineNode(
                node_info=pipeline_pb2.NodeInfo(
                    type=metadata_store_pb2.ExecutionType(
                        name="infra_validator"))),
            pipeline_info=pipeline_pb2.PipelineInfo(id="test_pipeline_id")))

  def testConcatArtifactUri(self):
    pb = text_format.Parse(CONCAT_SPLIT_URI_EXPRESSION,
                           placeholder_pb2.PlaceholderExpression())
    self.assertEqual(
        placeholder_utils.resolve_placeholder_expression(
            pb, self._resolution_context), "/tmp/train/1")

  def testProtoExecPropertyPrimitiveField(self):
    # Access a non-message type proto field
    placeholder_expression = """
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
          proto_field_path: ".tensorflow_serving"
          proto_field_path: ".tags"
          proto_field_path: "[1]"
        }
      }
    """
    pb = text_format.Parse(placeholder_expression,
                           placeholder_pb2.PlaceholderExpression())

    # Prepare FileDescriptorSet
    fd = descriptor_pb2.FileDescriptorProto()
    infra_validator_pb2.ServingSpec().DESCRIPTOR.file.CopyToProto(fd)
    pb.operator.proto_op.proto_schema.file_descriptors.file.append(fd)

    self.assertEqual(
        placeholder_utils.resolve_placeholder_expression(
            pb, self._resolution_context), "1.15.0-gpu")

  def testProtoExecPropertyMessageFieldTextFormat(self):
    # Access a message type proto field
    placeholder_expression = """
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
          proto_field_path: ".tensorflow_serving"
          serialization_format: TEXT_FORMAT
        }
      }
    """
    pb = text_format.Parse(placeholder_expression,
                           placeholder_pb2.PlaceholderExpression())

    fd = descriptor_pb2.FileDescriptorProto()
    infra_validator_pb2.ServingSpec().DESCRIPTOR.file.CopyToProto(fd)
    pb.operator.proto_op.proto_schema.file_descriptors.file.append(fd)

    # If proto_field_path points to a message type field, the message will
    # be rendered using text_format.
    self.assertEqual(
        placeholder_utils.resolve_placeholder_expression(
            pb, self._resolution_context),
        "tags: \"latest\"\ntags: \"1.15.0-gpu\"\n")

  def testProtoExecPropertyRepeatedField(self):
    # Access a repeated field.
    placeholder_expression = """
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
          proto_field_path: ".tensorflow_serving"
          proto_field_path: ".tags"
        }
      }
    """
    pb = text_format.Parse(placeholder_expression,
                           placeholder_pb2.PlaceholderExpression())

    # Prepare FileDescriptorSet
    fd = descriptor_pb2.FileDescriptorProto()
    infra_validator_pb2.ServingSpec().DESCRIPTOR.file.CopyToProto(fd)
    pb.operator.proto_op.proto_schema.file_descriptors.file.append(fd)

    self.assertEqual(
        placeholder_utils.resolve_placeholder_expression(
            pb, self._resolution_context), ["latest", "1.15.0-gpu"])

  def testSerializeDoubleValue(self):
    # Read a primitive value
    placeholder_expression = """
      value {
        double_value: 1.000000009
      }
    """
    pb = text_format.Parse(placeholder_expression,
                           placeholder_pb2.PlaceholderExpression())
    self.assertEqual(
        placeholder_utils.resolve_placeholder_expression(
            pb, self._resolution_context), 1.000000009)

  def testRuntimeInfoPlaceholderSimple(self):
    placeholder_expression = """
      placeholder {
        type: RUNTIME_INFO
        key: "executor_output_uri"
      }
    """
    pb = text_format.Parse(placeholder_expression,
                           placeholder_pb2.PlaceholderExpression())
    self.assertEqual(
        placeholder_utils.resolve_placeholder_expression(
            pb, self._resolution_context), "test_executor_output_uri")

  def testProtoRuntimeInfoPlaceholderMessageField(self):
    placeholder_expression = """
      operator {
        proto_op {
          expression {
            placeholder {
              type: RUNTIME_INFO
              key: "node_info"
            }
          }
          proto_field_path: ".type"
          proto_field_path: ".name"
        }
      }
    """
    pb = text_format.Parse(placeholder_expression,
                           placeholder_pb2.PlaceholderExpression())
    self.assertEqual(
        placeholder_utils.resolve_placeholder_expression(
            pb, self._resolution_context), "infra_validator")

  def testProtoSerializationJSON(self):
    placeholder_expression = """
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
          serialization_format: JSON
        }
      }
    """
    pb = text_format.Parse(placeholder_expression,
                           placeholder_pb2.PlaceholderExpression())

    # Prepare FileDescriptorSet
    fd = descriptor_pb2.FileDescriptorProto()
    infra_validator_pb2.ServingSpec().DESCRIPTOR.file.CopyToProto(fd)
    pb.operator.proto_op.proto_schema.file_descriptors.file.append(fd)

    expected_json_serialization = """\
{
  "tensorflow_serving": {
    "tags": [
      "latest",
      "1.15.0-gpu"
    ]
  }
}"""

    self.assertEqual(
        placeholder_utils.resolve_placeholder_expression(
            pb, self._resolution_context), expected_json_serialization)

  def testProtoWithoutSerializationFormat(self):
    placeholder_expression = """
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
        }
      }
    """
    pb = text_format.Parse(placeholder_expression,
                           placeholder_pb2.PlaceholderExpression())

    # Prepare FileDescriptorSet
    fd = descriptor_pb2.FileDescriptorProto()
    infra_validator_pb2.ServingSpec().DESCRIPTOR.file.CopyToProto(fd)
    pb.operator.proto_op.proto_schema.file_descriptors.file.append(fd)

    with self.assertRaises(ValueError):
      placeholder_utils.resolve_placeholder_expression(pb,
                                                       self._resolution_context)

  def testExecutionInvocationPlaceholderSimple(self):
    # TODO(b/170469176): Update when proto encoding Operator is available.
    placeholder_expression = """
      placeholder {
        type: EXEC_INVOCATION
      }
    """
    pb = text_format.Parse(placeholder_expression,
                           placeholder_pb2.PlaceholderExpression())
    resolved = placeholder_utils.resolve_placeholder_expression(
        pb, self._resolution_context)
    got_exec_invocation = executor_invocation_pb2.ExecutorInvocation.FromString(
        base64.b64decode(resolved))

    want_exec_invocation = text_format.Parse(
        _WANT_EXEC_INVOCATION, executor_invocation_pb2.ExecutorInvocation())
    self.assertProtoEquals(want_exec_invocation, got_exec_invocation)


if __name__ == "__main__":
  tf.test.main()
