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
"""Utilities for customizing tf.test.TestCase class."""

import contextlib
import copy
import os

from typing import Dict, Iterable, Optional, Union

import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.orchestration import metadata
from tfx.orchestration.portable.mlmd import event_lib
from tfx.utils import io_utils

from google.protobuf import message
from google.protobuf import text_format
from ml_metadata.proto import metadata_store_pb2


@contextlib.contextmanager
def override_env_var(name: str, value: str):
  """Overrides an environment variable and returns a context manager.

  Example:
    with test_case_utils.override_env_var('HOME', new_home_dir):

    or

    self.enter_context(test_case_utils.override_env_var('HOME', new_home_dir))

  Args:
    name: Name of the environment variable.
    value: Overriding value.

  Yields:
    None.
  """
  old_value = os.getenv(name)
  os.environ[name] = value

  yield

  if old_value is None:
    del os.environ[name]
  else:
    os.environ[name] = old_value


class TfxTest(tf.test.TestCase):
  """Convenient wrapper for tfx test cases."""

  def setUp(self):
    super().setUp()
    self.tmp_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    fileio.makedirs(self.tmp_dir)
    # TODO(b/176196624): Delete following block when we drop support for TF<2.4.
    # Manually set up exit_stack because absltest.TestCase.setUp() is not called
    # in TF<2.4.
    if self._exit_stack is None:
      self._exit_stack = contextlib.ExitStack()
      self.addCleanup(self._exit_stack.close)

  def load_proto_from_text(self, path: str,
                           proto_message: message.Message) -> message.Message:
    """Loads proto message from serialized text."""
    return io_utils.parse_pbtxt_file(path, proto_message)

  def assertProtoPartiallyEquals(
      self,
      expected: Union[str, message.Message],
      actual: message.Message,
      ignored_fields: Optional[Iterable[str]] = None,
  ):
    """Asserts proto messages are equal except the ignored fields."""
    if isinstance(expected, str):
      expected = text_format.Parse(expected, actual.__class__())
      actual = copy.deepcopy(actual)
    else:
      expected = copy.deepcopy(expected)
      actual = copy.deepcopy(actual)

    # Currently only supports one-level for ignored fields.
    for ignored_field in ignored_fields or []:
      expected.ClearField(ignored_field)
      actual.ClearField(ignored_field)

    return self.assertProtoEquals(expected, actual)


@contextlib.contextmanager
def change_working_dir(working_dir: str):
  """Changes working directory to a given temporary directory.

  Example:
    with test_case_utils.change_working_dir(tmp_dir):

    or

    self.enter_context(test_case_utils.change_working_dir(self.tmp_dir))

  Args:
    working_dir: The new working directory. This directoy should already exist.

  Yields:
    Old working directory.
  """

  old_dir = os.getcwd()
  os.chdir(working_dir)

  yield old_dir

  os.chdir(old_dir)


class MlmdMixins:
  """Populates a mock MLMD database with Contexts, Artifacts and Excutions."""
  mlmd_handler: metadata.Metadata
  _context_type_ids: Dict[str, int]
  _artifact_type_ids: Dict[str, int]
  _execution_type_ids: Dict[str, int]

  def init_mlmd(self):
    config = metadata_store_pb2.ConnectionConfig()
    config.fake_database.SetInParent()
    self.mlmd_handler = metadata.Metadata(config)
    self._context_type_ids = {}
    self._artifact_type_ids = {}
    self._execution_type_ids = {}

  @property
  def store(self):
    return self.mlmd_handler.store

  def _get_context_type_id(self, type_name: str):
    if type_name not in self._context_type_ids:
      result = self.store.put_context_type(
          metadata_store_pb2.ContextType(name=type_name))
      self._context_type_ids[type_name] = result
    return self._context_type_ids[type_name]

  def put_context(self, context_type: str, context_name: str):
    """Put a Context in the MLMD database."""
    result = metadata_store_pb2.Context(
        type_id=self._get_context_type_id(context_type), name=context_name)
    result.id = self.store.put_contexts([result])[0]
    return result

  def _get_artifact_type_id(self, type_name: str):
    if type_name not in self._artifact_type_ids:
      result = self.store.put_artifact_type(
          metadata_store_pb2.ArtifactType(name=type_name))
      self._artifact_type_ids[type_name] = result
    return self._artifact_type_ids[type_name]

  def put_artifact(self, artifact_type: str, uri: str = '/fake'):
    """Put an Artifact in the MLMD database."""
    result = metadata_store_pb2.Artifact(
        type_id=self._get_artifact_type_id(artifact_type),
        uri=uri,
        state=metadata_store_pb2.Artifact.LIVE)
    result.id = self.store.put_artifacts([result])[0]
    return result

  def _get_execution_type_id(self, type_name: str):
    if type_name not in self._execution_type_ids:
      result = self.store.put_execution_type(
          metadata_store_pb2.ExecutionType(name=type_name))
      self._execution_type_ids[type_name] = result
    return self._execution_type_ids[type_name]

  def put_execution(self, execution_type: str, inputs, outputs, contexts):
    """Put an Execution in the MLMD database."""
    result = metadata_store_pb2.Execution(
        type_id=self._get_execution_type_id(execution_type),
        last_known_state=metadata_store_pb2.Execution.COMPLETE)
    artifact_and_events = []
    for input_key, artifacts in inputs.items():
      for i, artifact in enumerate(artifacts):
        event = event_lib.generate_event(metadata_store_pb2.Event.INPUT,
                                         input_key, i)
        artifact_and_events.append((artifact, event))
    for output_key, artifacts in outputs.items():
      for i, artifact in enumerate(artifacts):
        event = event_lib.generate_event(metadata_store_pb2.Event.OUTPUT,
                                         output_key, i)
        artifact_and_events.append((artifact, event))
    result.id = self.store.put_execution(result, artifact_and_events,
                                         contexts)[0]
    return result
