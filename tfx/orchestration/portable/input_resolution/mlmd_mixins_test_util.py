# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Test utility for populating a mock MLMD database."""

from tfx.orchestration import metadata
from tfx.orchestration.portable.mlmd import event_lib

from ml_metadata.proto import metadata_store_pb2


class MlmdMixins:
  """Populates a mock MLMD database with Contexts, Artifacts and Excutions."""

  def init_mlmd(self):
    config = metadata_store_pb2.ConnectionConfig()
    config.fake_database.SetInParent()
    self.mlmd_handler = metadata.Metadata(config)
    self.__context_type_ids = {}
    self.__artifact_type_ids = {}
    self.__execution_type_ids = {}

  @property
  def store(self):
    return self.mlmd_handler.store  # pytype: disable=attribute-error

  def _get_context_type_id(self, type_name: str):
    if type_name not in self.__context_type_ids:  # pytype: disable=attribute-error
      result = self.store.put_context_type(
          metadata_store_pb2.ContextType(name=type_name))
      self.__context_type_ids[type_name] = result  # pytype: disable=attribute-error
    return self.__context_type_ids[type_name]  # pytype: disable=attribute-error

  def put_context(self, context_type: str, context_name: str):
    """Put a Context in the MLMD database."""
    result = metadata_store_pb2.Context(
        type_id=self._get_context_type_id(context_type), name=context_name)
    result.id = self.store.put_contexts([result])[0]
    return result

  def _get_artifact_type_id(self, type_name: str):
    if type_name not in self.__artifact_type_ids:  # pytype: disable=attribute-error
      result = self.store.put_artifact_type(
          metadata_store_pb2.ArtifactType(name=type_name))
      self.__artifact_type_ids[type_name] = result  # pytype: disable=attribute-error
    return self.__artifact_type_ids[type_name]  # pytype: disable=attribute-error

  def put_artifact(self, artifact_type: str, uri: str = '/fake'):
    """Put an Artifact in the MLMD database."""
    result = metadata_store_pb2.Artifact(
        type_id=self._get_artifact_type_id(artifact_type),
        uri=uri,
        state=metadata_store_pb2.Artifact.LIVE)
    result.id = self.store.put_artifacts([result])[0]
    return result

  def _get_execution_type_id(self, type_name: str):
    if type_name not in self.__execution_type_ids:  # pytype: disable=attribute-error
      result = self.store.put_execution_type(
          metadata_store_pb2.ExecutionType(name=type_name))
      self.__execution_type_ids[type_name] = result  # pytype: disable=attribute-error
    return self.__execution_type_ids[type_name]  # pytype: disable=attribute-error

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
