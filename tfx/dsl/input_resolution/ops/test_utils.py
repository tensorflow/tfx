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
"""Testing utility for builtin resolver ops."""
from typing import Type, Any, Dict, List, Optional, Sequence, Tuple, Union, Mapping
from unittest import mock

from absl.testing import parameterized

from tfx import types
from tfx.dsl.compiler import compiler_context
from tfx.dsl.compiler import node_inputs_compiler
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import base_driver
from tfx.dsl.components.base import base_executor
from tfx.dsl.components.base import executor_spec
from tfx.dsl.input_resolution import resolver_op
from tfx.dsl.input_resolution.ops import ops_utils
from tfx.orchestration import pipeline
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import artifact as tfx_artifact
from tfx.types import artifact_utils
from tfx.types import channel as channel_types
from tfx.types import component_spec
from tfx.utils import test_case_utils
from tfx.utils import typing_utils

import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2


class DummyArtifact(types.Artifact):
  """A dummy Artifact used for testing."""

  TYPE_NAME = 'DummyArtifact'

  PROPERTIES = {
      'span': tfx_artifact.Property(type=tfx_artifact.PropertyType.INT),
      'version': tfx_artifact.Property(type=tfx_artifact.PropertyType.INT),
  }

  # pylint: disable=redefined-builtin
  def __init__(
      self,
      id: Optional[str] = None,
      uri: Optional[str] = None,
      create_time_since_epoch: Optional[int] = None,
  ):
    super().__init__()
    if id is not None:
      self.id = id
    if uri is not None:
      self.uri = uri
    if create_time_since_epoch is not None:
      self.mlmd_artifact.create_time_since_epoch = create_time_since_epoch


class Examples(DummyArtifact):
  TYPE_NAME = ops_utils.EXAMPLES_TYPE_NAME


class TransformGraph(DummyArtifact):
  TYPE_NAME = ops_utils.TRANSFORM_GRAPH_TYPE_NAME


class Model(DummyArtifact):
  TYPE_NAME = ops_utils.MODEL_TYPE_NAME


class ModelBlessing(DummyArtifact):
  TYPE_NAME = ops_utils.MODEL_BLESSING_TYPE_NAME


class ModelInfraBlessing(DummyArtifact):
  TYPE_NAME = ops_utils.MODEL_INFRA_BLESSSING_TYPE_NAME


class ModelPush(DummyArtifact):
  TYPE_NAME = ops_utils.MODEL_PUSH_TYPE_NAME


class FakeSpec(component_spec.ComponentSpec):
  """FakeComponent component spec."""

  PARAMETERS = {}
  INPUTS = {
      'x': component_spec.ChannelParameter(
          type=tfx_artifact.Artifact, optional=True
      ),
      ops_utils.MODEL_KEY: component_spec.ChannelParameter(
          type=tfx_artifact.Artifact, optional=True
      ),
      ops_utils.MODEL_BLESSSING_KEY: component_spec.ChannelParameter(
          type=tfx_artifact.Artifact, optional=True
      ),
      ops_utils.MODEL_INFRA_BLESSING_KEY: component_spec.ChannelParameter(
          type=tfx_artifact.Artifact, optional=True
      ),
      ops_utils.MODEL_PUSH_KEY: component_spec.ChannelParameter(
          type=tfx_artifact.Artifact, optional=True
      ),
      ops_utils.EXAMPLES_KEY: component_spec.ChannelParameter(
          type=tfx_artifact.Artifact, optional=True
      ),
  }
  OUTPUTS = {}


class FakeComponent(base_component.BaseComponent):
  """FakeComponent that lets user overwrite input/output/exec_properties."""

  SPEC_CLASS = FakeSpec

  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(base_executor.EmptyExecutor)

  DRIVER_CLASS = base_driver.BaseDriver

  def __init__(self, id: str, inputs=None, exec_properties=None):  # pylint: disable=redefined-builtin
    super().__init__(spec=FakeSpec())
    self.with_id(id)

    # We override the inputs, exec_properties, and outputs.
    self._inputs = inputs or {}
    self._exec_properties = exec_properties or {}
    self._outputs = {}

  def output(self, key: str, artifact_type=DummyArtifact):
    if key not in self._outputs:
      self._outputs[key] = channel_types.OutputChannel(artifact_type, self, key)
    return self._outputs[key]

  @property
  def inputs(self) -> ...:
    return self._inputs

  @property
  def exec_properties(self) -> ...:
    return self._exec_properties

  @property
  def outputs(self) -> ...:
    return self._outputs


def compile_inputs(
    inputs: Dict[str, channel_types.BaseChannel]
) -> pipeline_pb2.PipelineNode:
  """Returns a compiled PipelineNode from the FakeComponent inputs dict."""
  node = FakeComponent('FakeComponent', inputs=inputs)
  p = pipeline.Pipeline(pipeline_name='pipeline', components=[node])
  ctx = compiler_context.PipelineContext(p)
  node_inputs = pipeline_pb2.NodeInputs()

  # Compile the NodeInputs and wrap in a PipelineNode.
  node_inputs_compiler.compile_node_inputs(ctx, node, node_inputs)
  return pipeline_pb2.PipelineNode(inputs=node_inputs)


class ResolverTestCase(
    test_case_utils.MlmdMixins,
    test_case_utils.TfxTest,
    parameterized.TestCase,
):
  """MLMD mixins for testing ResolverOps and resolver functions."""

  def prepare_tfx_artifact(
      self,
      artifact: Any,  # If set to types.Artifact, pytype throws spurious errors.
      properties: Optional[Dict[str, Union[int, str]]] = None,
      custom_properties: Optional[Dict[str, Union[int, str]]] = None,
  ) -> types.Artifact:
    """Adds a single artifact to MLMD and returns the TFleX Artifact object."""
    mlmd_artifact = self.put_artifact(
        artifact.TYPE_NAME,
        properties=properties,
        custom_properties=custom_properties,
    )
    artifact_type = self.store.get_artifact_type(artifact.TYPE_NAME)
    return artifact_utils.deserialize_artifact(artifact_type, mlmd_artifact)

  def unwrap_tfx_artifacts(
      self, artifacts: List[types.Artifact]
  ) -> List[types.Artifact]:
    """Return the underlying MLMD Artifacta of a list of TFleX Artifacts."""
    return [a.mlmd_artifact for a in artifacts]

  def create_examples(
      self,
      spans_and_versions: Sequence[Tuple[int, int]],
      contexts: Sequence[metadata_store_pb2.Context] = (),
  ) -> List[types.Artifact]:
    """Build Examples artifacts and add an ExampleGen execution to MLMD."""
    examples = []
    for span, version in spans_and_versions:
      examples.append(
          self.prepare_tfx_artifact(
              Examples, properties={'span': span, 'version': version}
          )
      )
    self.put_execution(
        'ExampleGen',
        inputs={},
        outputs={'examples': self.unwrap_tfx_artifacts(examples)},
        contexts=contexts,
    )
    return examples

  def train_on_examples(
      self,
      model: types.Artifact,
      examples: List[types.Artifact],
      transform_graph: Optional[types.Artifact] = None,
      contexts: Sequence[metadata_store_pb2.Context] = (),
  ):
    """Add an Execution to MLMD where a Trainer trains on the examples."""
    inputs = {'examples': self.unwrap_tfx_artifacts(examples)}
    if transform_graph is not None:
      inputs['transform_graph'] = self.unwrap_tfx_artifacts([transform_graph])
    self.put_execution(
        'TFTrainer',
        inputs=inputs,
        outputs={'model': self.unwrap_tfx_artifacts([model])},
        contexts=contexts,
    )

  def evaluator_bless_model(
      self,
      model: types.Artifact,
      blessed: bool = True,
      baseline_model: Optional[types.Artifact] = None,
      contexts: Sequence[metadata_store_pb2.Context] = (),
  ) -> types.Artifact:
    """Add an Execution to MLMD where the Evaluator blesses the model."""
    model_blessing = self.prepare_tfx_artifact(
        ModelBlessing, custom_properties={'blessed': int(blessed)}
    )

    inputs = {'model': self.unwrap_tfx_artifacts([model])}
    if baseline_model is not None:
      inputs['baseline_model'] = self.unwrap_tfx_artifacts([baseline_model])

    self.put_execution(
        'Evaluator',
        inputs=inputs,
        outputs={'blessing': self.unwrap_tfx_artifacts([model_blessing])},
        contexts=contexts,
    )

    return model_blessing

  def infra_validator_bless_model(
      self,
      model: types.Artifact,
      blessed: bool = True,
      contexts: Sequence[metadata_store_pb2.Context] = (),
  ) -> types.Artifact:
    """Add an Execution to MLMD where the InfraValidator blesses the model."""
    if blessed:
      custom_properties = {'blessing_status': 'INFRA_BLESSED'}
    else:
      custom_properties = {'blessing_status': 'INFRA_NOT_BLESSED'}
    model_infra_blessing = self.prepare_tfx_artifact(
        ModelInfraBlessing, custom_properties=custom_properties
    )

    self.put_execution(
        'InfraValidator',
        inputs={'model': self.unwrap_tfx_artifacts([model])},
        outputs={'result': self.unwrap_tfx_artifacts([model_infra_blessing])},
        contexts=contexts,
    )

    return model_infra_blessing

  def push_model(
      self,
      model: types.Artifact,
      contexts: Sequence[metadata_store_pb2.Context] = (),
  ):
    """Add an Execution to MLMD where the Pusher pushes the model."""
    model_push = self.prepare_tfx_artifact(ModelPush)
    self.put_execution(
        'ServomaticPusher',
        inputs={'model_export': self.unwrap_tfx_artifacts([model])},
        outputs={'model_push': self.unwrap_tfx_artifacts([model_push])},
        contexts=contexts,
    )
    return model_push


def run_resolver_op(
    op_type: Type[resolver_op.ResolverOp],
    *arg: Any,
    context: Optional[resolver_op.Context] = None,
    **kwargs: Any,
):
  op = op_type.create(**kwargs)
  if context:
    op.set_context(context)
  return op.apply(*arg)


def strict_run_resolver_op(
    op_type: Type[resolver_op.ResolverOp],
    *,
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
    store: Optional[mlmd.MetadataStore] = None,
):
  """Runs ResolverOp with strict type checking."""
  if len(args) != len(op_type.arg_data_types):
    raise TypeError(
        f'len({op_type}.arg_data_types) = {len(op_type.arg_data_types)} but'
        f' got len(args) = {len(args)}'
    )
  if op_type.arg_data_types:
    for i, arg, expected_data_type in zip(
        range(len(args)), args, op_type.arg_data_types
    ):
      if expected_data_type == resolver_op.DataType.ARTIFACT_LIST:
        if not typing_utils.is_artifact_list(arg):
          raise TypeError(f'Expected ARTIFACT_LIST but arg[{i}] = {arg}')
      elif expected_data_type == resolver_op.DataType.ARTIFACT_MULTIMAP:
        if not typing_utils.is_artifact_multimap(arg):
          raise TypeError(f'Expected ARTIFACT_MULTIMAP but arg[{i}] = {arg}')
      elif expected_data_type == resolver_op.DataType.ARTIFACT_MULTIMAP_LIST:
        if not typing_utils.is_list_of_artifact_multimap(arg):
          raise TypeError(
              f'Expected ARTIFACT_MULTIMAP_LIST but arg[{i}] = {arg}'
          )
  op = op_type.create(**kwargs)
  context = resolver_op.Context(
      store=store
      if store is not None
      else mock.MagicMock(spec=mlmd.MetadataStore)
  )
  op.set_context(context)
  result = op.apply(*args)
  if op_type.return_data_type == resolver_op.DataType.ARTIFACT_LIST:
    if not typing_utils.is_homogeneous_artifact_list(result):
      raise TypeError(f'Expected ARTIFACT_LIST result but got {result}')
  elif op_type.return_data_type == resolver_op.DataType.ARTIFACT_MULTIMAP:
    if not typing_utils.is_artifact_multimap(result):
      raise TypeError(f'Expected ARTIFACT_MULTIMAP result but got {result}')
  elif op_type.return_data_type == resolver_op.DataType.ARTIFACT_MULTIMAP_LIST:
    if not typing_utils.is_list_of_artifact_multimap(result):
      raise TypeError(
          f'Expected ARTIFACT_MULTIMAP_LIST result but got {result}'
      )
  return result
