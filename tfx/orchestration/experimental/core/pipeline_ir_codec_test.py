# Copyright 2024 Google LLC. All Rights Reserved.
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
"""Tests for tfx.orchestration.experimental.core.pipeline_ir_codec."""
import json
import os
from typing import List, Optional
import tensorflow as tf
from tfx.orchestration.experimental.core import env
from tfx.orchestration.experimental.core import pipeline_ir_codec
from tfx.orchestration.experimental.core import test_utils
from tfx.proto.orchestration import pipeline_pb2


def _test_pipeline(
    pipeline_id,
    execution_mode: pipeline_pb2.Pipeline.ExecutionMode = (
        pipeline_pb2.Pipeline.ASYNC
    ),
    param=1,
    pipeline_nodes: Optional[List[str]] = None,
    pipeline_run_id: str = 'run0',
    pipeline_root: str = '',
):
  pipeline = pipeline_pb2.Pipeline()
  pipeline.pipeline_info.id = pipeline_id
  pipeline.execution_mode = execution_mode
  if pipeline_nodes:
    for node in pipeline_nodes:
      pipeline.nodes.add().pipeline_node.node_info.id = node
    pipeline.nodes[0].pipeline_node.parameters.parameters[
        'param'
    ].field_value.int_value = param
  if execution_mode == pipeline_pb2.Pipeline.SYNC:
    pipeline.runtime_spec.pipeline_run_id.field_value.string_value = (
        pipeline_run_id
    )
  pipeline.runtime_spec.pipeline_root.field_value.string_value = pipeline_root
  return pipeline


class TestEnv(env._DefaultEnv):

  def __init__(self, base_dir, max_str_len):
    self.base_dir = base_dir
    self.max_str_len = max_str_len

  def get_base_dir(self):
    return self.base_dir

  def max_mlmd_str_value_length(self):
    return self.max_str_len


class PipelineIRCodecTest(test_utils.TfxTest):

  def setUp(self):
    super().setUp()
    self._pipeline_root = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self.id(),
    )

  def test_encode_decode_no_base_dir(self):
    with TestEnv(None, None):
      pipeline = _test_pipeline('pipeline1', pipeline_nodes=['Trainer'])
      pipeline_encoded = pipeline_ir_codec.PipelineIRCodec.get().encode(
          pipeline
      )
    self.assertProtoEquals(
        pipeline,
        pipeline_ir_codec._base64_decode_pipeline(pipeline_encoded),
        'Expected pipeline IR to be base64 encoded.',
    )
    self.assertProtoEquals(
        pipeline,
        pipeline_ir_codec.PipelineIRCodec.get().decode(pipeline_encoded),
    )

  def test_encode_decode_with_base_dir(self):
    with TestEnv(self._pipeline_root, None):
      pipeline = _test_pipeline('pipeline1', pipeline_nodes=['Trainer'])
      pipeline_encoded = pipeline_ir_codec.PipelineIRCodec.get().encode(
          pipeline
      )
    self.assertProtoEquals(
        pipeline,
        pipeline_ir_codec._base64_decode_pipeline(pipeline_encoded),
        'Expected pipeline IR to be base64 encoded.',
    )
    self.assertProtoEquals(
        pipeline,
        pipeline_ir_codec.PipelineIRCodec.get().decode(pipeline_encoded),
    )

  def test_encode_decode_exceeds_max_len(self):
    with TestEnv(self._pipeline_root, 0):
      pipeline = _test_pipeline(
          'pipeline1',
          pipeline_nodes=['Trainer'],
          pipeline_root=self.create_tempdir().full_path,
      )
      pipeline_encoded = pipeline_ir_codec.PipelineIRCodec.get().encode(
          pipeline
      )
    self.assertProtoEquals(
        pipeline,
        pipeline_ir_codec.PipelineIRCodec.get().decode(pipeline_encoded),
    )
    self.assertEqual(
        pipeline_ir_codec.PipelineIRCodec._PIPELINE_IR_URL_KEY,
        next(iter(json.loads(pipeline_encoded).keys())),
        'Expected pipeline IR URL to be stored as json.',
    )


