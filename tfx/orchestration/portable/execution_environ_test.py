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
"""Tests for tfx.orchestration.portable.execution_environ."""

from typing import Any, Callable, List, Optional, Type, Union
from absl.testing import parameterized
import tensorflow as tf

from tfx.orchestration.experimental.core import test_utils
from tfx.orchestration.portable import data_types
from tfx.orchestration.portable import execution_environ
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import artifact
from tfx.types import standard_artifacts
from tfx.utils.di import errors


_Example = standard_artifacts.Examples
_Model = standard_artifacts.Model
_Artifact = artifact.Artifact
_Integer = standard_artifacts.Integer


def _create_artifact(
    uri: str, artifact_type: Type[_Artifact] = _Example
) -> _Artifact:
  a = artifact_type()
  a.uri = uri
  return a


class ExecutionEnvironTest(parameterized.TestCase, test_utils.TfxTest):

  def setUp(self):
    super().setUp()
    self._execution_id = 111
    self._stateful_working_dir = 'stateful/working/dir'
    self._tmp_dir = 'tmp/dir'
    self._node_id = 'node_id'
    self._pipeline_id = 'pipeline_id'
    self._pipeline_run_id = 'pipeline_run_id'
    self._top_level_pipeline_run_id = 'top_level_pipeline_run_id'
    self._frontend_url = 'frontend_url'

    self._single_artifact_input = [_create_artifact('uri1')]
    self._multiple_artifacts_input = [
        _create_artifact('uri2'),
        _create_artifact('uri3'),
    ]
    self._single_artifact_output = [_create_artifact('uri4')]

    self._execution_info = data_types.ExecutionInfo(
        input_dict={
            'single_artifact_input': self._single_artifact_input,
            'multiple_artifacts_input': self._multiple_artifacts_input,
            'empty_artifact_input': [],
        },
        output_dict={
            'single_artifact_output': self._single_artifact_output,
        },
        exec_properties={
            'string_key': 'string_value',
            'int_key': 123,
        },
        execution_id=self._execution_id,
        stateful_working_dir=self._stateful_working_dir,
        tmp_dir=self._tmp_dir,
        pipeline_node=pipeline_pb2.PipelineNode(
            node_info=pipeline_pb2.NodeInfo(id='node_id')
        ),
        pipeline_info=pipeline_pb2.PipelineInfo(id='pipeline_id'),
        pipeline_run_id=self._pipeline_run_id,
        top_level_pipeline_run_id=self._top_level_pipeline_run_id,
        frontend_url=self._frontend_url,
    )

    self._environ = execution_environ.Environ(
        execution_info=self._execution_info
    )

  def test_strict_get_single_artifact(self):
    self.assertArtifactEqual(
        self._environ.strict_get('single_artifact_input', _Example),
        self._single_artifact_input[0],
    )
    self.assertArtifactEqual(
        self._environ.strict_get('single_artifact_output', _Example),
        self._single_artifact_output[0],
    )

  @parameterized.named_parameters(
      ('builtin_list', lambda t: list[t]),
      ('typing_list', lambda t: List[t]),
  )
  def test_strict_get_list_of_artifacts(
      self, type_wrapper: Callable[..., Type[Any]]
  ):
    self.assertArtifactListEqual(
        self._environ.strict_get(
            'multiple_artifacts_input', type_wrapper(_Example)
        ),
        self._multiple_artifacts_input,
    )
    self.assertEmpty(
        self._environ.strict_get('empty_artifact_input', type_wrapper(_Example))
    )

  @parameterized.named_parameters(
      ('optional_wrapper', lambda t: Optional[t]),
      ('union_with_none_wrapper', lambda t: Union[t, None]),
  )
  def test_strict_get_optional_artifact(
      self, type_wrapper: Callable[..., Type[Any]]
  ):
    self.assertArtifactEqual(
        self._environ.strict_get(
            'single_artifact_input', type_wrapper(_Example)
        ),
        self._single_artifact_input[0],
    )
    self.assertIsNone(
        self._environ.strict_get(
            'empty_artifact_input', type_wrapper(_Example)
        ),
    )

  def test_strict_get_single_artifact_raises_error_when_non_singular_list(self):
    with self.assertRaisesRegex(
        errors.InvalidTypeHintError,
        r'type_hint = <class \'(.*?)Examples\'> but got 2 artifacts\. Please'
        r' use list\[Examples\] or Optional\[Examples\] annotation instead\.',
    ):
      self._environ.strict_get('multiple_artifacts_input', _Example)
    with self.assertRaisesRegex(
        errors.InvalidTypeHintError,
        r'type_hint = <class \'(.*?)Examples\'> but got 0 artifacts\. Please'
        r' use list\[Examples\] or Optional\[Examples\] annotation instead\.',
    ):
      self._environ.strict_get('empty_artifact_input', _Example)

  def test_strict_get_artifact_raises_error_when_invalid_type_hint(self):
    with self.assertRaisesWithLiteralMatch(
        errors.InvalidTypeHintError,
        'Unsupported annotation: <class \'str\'>'
    ):
      self._environ.strict_get('single_artifact_output', str)

  def test_strict_get_raises_error_when_type_not_strictly_matched(self):
    with self.assertRaisesWithLiteralMatch(
        errors.InvalidTypeHintError,
        'type_hint uses Model but the resolved artifacts have type_name ='
        ' Examples',
    ):
      self._environ.strict_get('multiple_artifacts_input', list[_Model])
    with self.assertRaisesWithLiteralMatch(
        errors.InvalidTypeHintError,
        'type_hint uses Model but the resolved artifacts have type_name ='
        ' Examples',
    ):
      self._environ.strict_get('single_artifact_input', _Model)

  def test_strict_get_exec_properties(self):
    self.assertEqual(
        self._environ.strict_get('string_key', str), 'string_value'
    )
    self.assertEqual(self._environ.strict_get('int_key', int), 123)

  def test_strict_get_exec_properties_raises_error_when_invalid_type_hint(self):
    with self.assertRaisesWithLiteralMatch(
        errors.InvalidTypeHintError,
        "Given type_hint = <class 'int'> but exec_property[string_key] ="
        ' string_value is not compatible.',
    ):
      self._environ.strict_get('string_key', int)
    with self.assertRaisesWithLiteralMatch(
        errors.InvalidTypeHintError,
        "Given type_hint = <class 'str'> but exec_property[int_key] = 123 is"
        ' not compatible.',
    ):
      self._environ.strict_get('int_key', str)

  def test_strict_get_raises_error_when_unknown_name(self):
    with self.assertRaisesRegex(
        errors.NotProvidedError,
        r'No matching providers found for name=unknown_name, type_hint=<class'
        r' \'str\'>\. Available providers: (.*?)',
    ):
      self._environ.strict_get('unknown_name', str)


if __name__ == '__main__':
  tf.test.main()
