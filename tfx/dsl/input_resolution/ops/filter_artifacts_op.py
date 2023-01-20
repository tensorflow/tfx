# Copyright 2023 Google LLC. All Rights Reserved.
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
"""FilterArtifacts resolver operator."""

from typing import Callable, Any, List, Sequence

from tfx import types as tfx_types
from tfx.dsl.compiler import placeholder_utils
from tfx.dsl.input_resolution import resolver_op
from tfx.dsl.placeholder import placeholder
from tfx.orchestration.portable import data_types
from tfx.orchestration.portable.input_resolution import exceptions
from tfx.proto.orchestration import placeholder_pb2


_DUMMY_INPUT_KEY = 'value'


def _encode_predicate(
    predicate_fn: Callable[
        [placeholder.ArtifactPlaceholder], placeholder.Predicate
    ]
) -> placeholder_pb2.PlaceholderExpression:
  """Encodes artifact predicate lambda into PlaceholderExpression."""
  result = predicate_fn(tfx_types.Channel(tfx_types.Artifact).future())
  if not isinstance(result, placeholder.Predicate):
    raise TypeError('predicate_fn does not return a placeholder.Predicate: '
                    f'type={type(result)}')
  return result.encode_with_keys(lambda _: _DUMMY_INPUT_KEY)


def _decode_predicate(
    encoded: placeholder_pb2.PlaceholderExpression,
) -> Callable[[tfx_types.Artifact], bool]:
  """Decodes PlaceholderExpression into artifact predicate function."""
  def predicate(artifact: tfx_types.Artifact) -> bool:
    try:
      result = placeholder_utils.resolve_placeholder_expression(
          encoded,
          placeholder_utils.ResolutionContext(
              exec_info=data_types.ExecutionInfo(
                  input_dict={_DUMMY_INPUT_KEY: [artifact]}
              )
          ),
      )
    except ValueError as e:
      # Placeholder evaluation errors are converted to ValueError.
      raise exceptions.InputResolutionError(str(e)) from e
    if not isinstance(result, bool):
      raise exceptions.FailedPreconditionError(
          f'Placeholder evaluates to non-boolean value {result}.'
          f'PlaceholderExpression proto: {encoded}'
      )
    return result

  return predicate


# Technically the argument and the return type is resolver_op.Node but the
# current resolver function tracing does not work well with pytype, so just use
# Any.
def FilterArtifacts(  # pylint: disable=invalid-name
    artifacts: Any,
    predicate_fn: Callable[
        [placeholder.ArtifactPlaceholder], placeholder.Predicate
    ],
) -> Any:
  """FilterArtifacts resolver operator.

  Usage:
    ```python
    FilterArtifacts(
        artifacts,
        lambda a: a.property('foo') == 42
    )
    ```

  Args:
    artifacts: ARTIFACT_LIST type argument node.
    predicate_fn: Placeholder function that returns boolean expression. You can
      use it with `ph.logical_and()` or `ph.logical_or()`.
  Returns:
    ARTIFACT_LIST type node.
  """
  return FilterArtifactsInternal(
      artifacts, encoded_predicate=_encode_predicate(predicate_fn)
  )


class FilterArtifactsInternal(
    resolver_op.ResolverOp,
    canonical_name='tfx.FilterArtifacts',
    arg_data_types=(resolver_op.DataType.ARTIFACT_LIST,),
    return_data_type=resolver_op.DataType.ARTIFACT_LIST,
):
  """FilterArtifacts implementation."""

  encoded_predicate = resolver_op.Property(
      type=placeholder_pb2.PlaceholderExpression
  )

  def apply(
      self, artifacts: Sequence[tfx_types.Artifact]
  ) -> List[tfx_types.Artifact]:
    if not artifacts:
      return []

    predicate = _decode_predicate(self.encoded_predicate)
    return [a for a in artifacts if predicate(a)]
