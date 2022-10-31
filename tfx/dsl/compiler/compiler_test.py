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
r"""Tests for tfx.dsl.compiler.compiler.

To update the golden IR proto, use --persist_test_protos flag.
"""

import os
import threading
import types
from typing import List, Dict, Any
import unittest

from absl import flags
from absl.testing import parameterized
import tensorflow as tf
from tfx.dsl.compiler import compiler
from tfx.dsl.compiler.testdata import additional_properties_test_pipeline_async
from tfx.dsl.compiler.testdata import channel_union_pipeline
from tfx.dsl.compiler.testdata import composable_pipeline
from tfx.dsl.compiler.testdata import conditional_pipeline
from tfx.dsl.compiler.testdata import consumer_pipeline
from tfx.dsl.compiler.testdata import dynamic_exec_properties_pipeline
from tfx.dsl.compiler.testdata import external_artifacts_pipeline
from tfx.dsl.compiler.testdata import foreach_pipeline
from tfx.dsl.compiler.testdata import iris_pipeline_async
from tfx.dsl.compiler.testdata import iris_pipeline_sync
from tfx.dsl.compiler.testdata import non_existent_component_pipeline
from tfx.dsl.compiler.testdata import optional_and_allow_empty_pipeline
from tfx.dsl.compiler.testdata import pipeline_root_placeholder
from tfx.dsl.compiler.testdata import pipeline_with_annotations
from tfx.dsl.compiler.testdata import resolver_function_pipeline
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import artifact
from tfx.types import channel
from tfx.utils import golden_utils

from google.protobuf import text_format

FLAGS = flags.FLAGS
persist_test_protos = flags.DEFINE_bool(
    "persist_test_protos", False, "Use for regenerating test data. Needs "
    "test_strategy=local.")

_GLODEN_IR_HEADER = """\
# proto-file: third_party/py/tfx/proto/orchestration/pipeline.proto
# proto-message: Pipeline
#
# This file contains the IR of an example pipeline
# tfx/dsl/compiler/testdata/{module_name}.py
"""


def _golden_path(filename: str) -> str:
  return os.path.join(os.path.dirname(__file__), "testdata", filename)


def _persist_pipeline_proto(
    module_name: str, golden_filename: str,
    pipeline_proto: pipeline_pb2.Pipeline) -> None:
  module_name = module_name.rpartition(".")[-1]
  source_path = golden_utils.get_source_path(_golden_path(golden_filename))
  print(f"Persisting to {source_path}")
  with open(source_path, mode="w+") as f:
    f.write(_GLODEN_IR_HEADER.format(module_name=module_name))
    f.write("\n")
    f.write(text_format.MessageToString(pipeline_proto))


def _get_test_cases_params(
    pipeline_modules: List[types.ModuleType],
) -> List[Dict[str, Any]]:
  result = []
  for module in pipeline_modules:
    testcase_name_segments = [module.__name__.rpartition(".")[-1]]
    # TODO(b/256081156) Clean up the "input_v2" suffix.
    testcase_name_segments.append("input_v2")
    testcase_name = "_".join(testcase_name_segments)
    golden_filename = f"{testcase_name}_ir.pbtxt"
    result.append(
        dict(
            testcase_name=testcase_name,
            pipeline_module=module,
            golden_filename=golden_filename,
        ))
  return result


class _MyType(artifact.Artifact):
  TYPE_NAME = "MyTypeName"
  PROPERTIES = {
      "string_value": artifact.Property(artifact.PropertyType.STRING),
  }


class CompilerTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # pylint: disable=g-bad-name
    self.maxDiff = 80 * 1000  # Let's hear what assertEqual has to say.

  def _get_pipeline_ir(self, filename: str) -> pipeline_pb2.Pipeline:
    """Reads expected pipeline pb from a text proto file."""
    with open(_golden_path(filename)) as f:
      return text_format.ParseLines(f, pipeline_pb2.Pipeline())

  @unittest.skipIf(tf.__version__ < "2",
                   "Large proto comparison has a bug not fixed with TF < 2.")
  @parameterized.named_parameters(
      *_get_test_cases_params([
          additional_properties_test_pipeline_async,
          iris_pipeline_sync,
          iris_pipeline_async,
          conditional_pipeline,
          foreach_pipeline,
          channel_union_pipeline,
          pipeline_root_placeholder,
          dynamic_exec_properties_pipeline,
          pipeline_with_annotations,
          composable_pipeline,
          resolver_function_pipeline,
          optional_and_allow_empty_pipeline,
          consumer_pipeline,
          external_artifacts_pipeline,
      ])
  )
  def testCompile(
      self,
      pipeline_module: types.ModuleType,
      golden_filename: str,
  ):
    """Tests compiling the whole pipeline."""
    dsl_compiler = compiler.Compiler()
    compiled_pb = dsl_compiler.compile(pipeline_module.create_test_pipeline())
    try:
      expected_pb = self._get_pipeline_ir(golden_filename)
    except FileNotFoundError:
      if persist_test_protos.value:
        _persist_pipeline_proto(
            pipeline_module.__name__, golden_filename, compiled_pb)
      raise
    if persist_test_protos.value:
      _persist_pipeline_proto(
          pipeline_module.__name__, golden_filename, compiled_pb)
    self.assertProtoEquals(expected_pb, compiled_pb)

  def testCompileAdditionalPropertyTypeError(self):
    dsl_compiler = compiler.Compiler()
    test_pipeline = (
        additional_properties_test_pipeline_async.create_test_pipeline())
    custom_producer = next(
        c for c in test_pipeline.components if isinstance(
            c, additional_properties_test_pipeline_async.CustomProducer))
    custom_producer.outputs["stats"].additional_properties[
        "span"] = "wrong_type"
    with self.assertRaisesRegex(TypeError, "Expected INT but given STRING"):
      dsl_compiler.compile(test_pipeline)

  def testCompileDynamicExecPropTypeError(self):
    dsl_compiler = compiler.Compiler()
    test_pipeline = dynamic_exec_properties_pipeline.create_test_pipeline()
    downstream_component = next(
        c for c in test_pipeline.components
        if isinstance(c, dynamic_exec_properties_pipeline.DownstreamComponent))
    test_wrong_type_channel = channel.Channel(_MyType).future()
    downstream_component.exec_properties["input_num"] = test_wrong_type_channel
    with self.assertRaisesRegex(
        ValueError,
        "Dynamic execution property only supports ValueArtifact typed channel."
    ):
      dsl_compiler.compile(test_pipeline)

  def testCompileNoneExistentNodeError(self):
    dsl_compiler = compiler.Compiler()
    test_pipeline = non_existent_component_pipeline.create_test_pipeline()
    with self.assertRaisesRegex(
        ValueError,
        "Node .* references downstream node .* which is not present in the "
        "pipeline"):
      dsl_compiler.compile(test_pipeline)

  def test_DefineAtSub_CompileAtMain(self):
    result_holder = []

    def define_pipeline():
      result_holder.append(conditional_pipeline.create_test_pipeline())

    t = threading.Thread(target=define_pipeline)
    t.start()
    t.join()
    self.assertLen(result_holder, 1)
    p = result_holder[0]
    compiled_pb = compiler.Compiler().compile(p)
    expected_pb = self._get_pipeline_ir(
        "conditional_pipeline_input_v2_ir.pbtxt")
    self.assertProtoEquals(expected_pb, compiled_pb)

  def test_DefineAtSub_CompileAtSub(self):
    result_holder = []

    def define_and_compile_pipeline():
      p = conditional_pipeline.create_test_pipeline()
      result_holder.append(compiler.Compiler().compile(p))

    t = threading.Thread(target=define_and_compile_pipeline)
    t.start()
    t.join()
    self.assertLen(result_holder, 1)
    expected_pb = self._get_pipeline_ir(
        "conditional_pipeline_input_v2_ir.pbtxt")
    self.assertProtoEquals(expected_pb, result_holder[0])


if __name__ == "__main__":
  tf.test.main()
