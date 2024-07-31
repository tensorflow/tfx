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
"""A class for encoding / decoding pipeline IR."""

import base64
import json
import os
import threading
import uuid

from tfx.dsl.io import fileio
from tfx.orchestration.experimental.core import env
from tfx.orchestration.experimental.core import task as task_lib
from tfx.proto.orchestration import pipeline_pb2

from google.protobuf import message


class PipelineIRCodec:
  """A class for encoding / decoding pipeline IR."""

  _ORCHESTRATOR_METADATA_DIR = '.orchestrator'
  _PIPELINE_IRS_DIR = 'pipeline_irs'
  _PIPELINE_IR_URL_KEY = 'pipeline_ir_url'
  _obj = None
  _lock = threading.Lock()

  @classmethod
  def get(cls) -> 'PipelineIRCodec':
    with cls._lock:
      if not cls._obj:
        cls._obj = cls()
      return cls._obj

  @classmethod
  def testonly_reset(cls) -> None:
    """Reset global state, for tests only."""
    with cls._lock:
      cls._obj = None

  def encode(self, pipeline: pipeline_pb2.Pipeline) -> str:
    """Encodes pipeline IR."""
    # Attempt to store as a base64 encoded string. If base_dir is provided
    # and the length is too large, store the IR on disk and retain the URL.
    # TODO(b/248786921): Always store pipeline IR to base_dir once the
    # accessibility issue is resolved.

    # Note that this setup means that every *subpipeline* will have its own
    # "irs" dir. This is fine, though ideally we would put all pipeline IRs
    # under the root pipeline dir, which would require us to *also* store the
    # root pipeline dir in the IR.

    base_dir = pipeline.runtime_spec.pipeline_root.field_value.string_value
    if base_dir:
      pipeline_ir_dir = os.path.join(
          base_dir, self._ORCHESTRATOR_METADATA_DIR, self._PIPELINE_IRS_DIR
      )
      fileio.makedirs(pipeline_ir_dir)
    else:
      pipeline_ir_dir = None
    pipeline_encoded = _base64_encode(pipeline)
    max_mlmd_str_value_len = env.get_env().max_mlmd_str_value_length()
    if (
        base_dir
        and pipeline_ir_dir
        and max_mlmd_str_value_len is not None
        and len(pipeline_encoded) > max_mlmd_str_value_len
    ):
      pipeline_id = task_lib.PipelineUid.from_pipeline(pipeline).pipeline_id
      pipeline_url = os.path.join(
          pipeline_ir_dir, f'{pipeline_id}_{uuid.uuid4()}.pb'
      )
      with fileio.open(pipeline_url, 'wb') as file:
        file.write(pipeline.SerializeToString())
      pipeline_encoded = json.dumps({self._PIPELINE_IR_URL_KEY: pipeline_url})
    return pipeline_encoded

  def decode(self, value: str) -> pipeline_pb2.Pipeline:
    """Decodes pipeline IR."""
    # Attempt to load as JSON. If it fails, fallback to decoding it as a base64
    # encoded string for backward compatibility.
    try:
      pipeline_encoded = json.loads(value)
      with fileio.open(
          pipeline_encoded[self._PIPELINE_IR_URL_KEY], 'rb'
      ) as file:
        return pipeline_pb2.Pipeline.FromString(file.read())
    except json.JSONDecodeError:
      return _base64_decode_pipeline(value)


def _base64_encode(msg: message.Message) -> str:
  return base64.b64encode(msg.SerializeToString()).decode('utf-8')


def _base64_decode_pipeline(pipeline_encoded: str) -> pipeline_pb2.Pipeline:
  result = pipeline_pb2.Pipeline()
  result.ParseFromString(base64.b64decode(pipeline_encoded))
  return result
