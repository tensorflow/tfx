# Lint as: python2, python3
# Copyright 2019 Google LLC. All Rights Reserved.
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
"""TFX Transform component definition."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
from typing import Any, Dict, Optional, Text, Union

import absl

from tfx import types
from tfx.components.base import base_component
from tfx.components.base import executor_spec
from tfx.components.transform import executor
from tfx.orchestration import data_types
from tfx.types import standard_artifacts
from tfx.types.standard_component_specs import TransformSpec


class Transform(base_component.BaseComponent):
  """A TFX component to transform the input examples.

  The Transform component wraps TensorFlow Transform (tf.Transform) to
  preprocess data in a TFX pipeline. This component will load the
  preprocessing_fn from input module file, preprocess both 'train' and 'eval'
  splits of input examples, generate the `tf.Transform` output, and save both
  transform function and transformed examples to orchestrator desired locations.

  ## Providing a preprocessing function
  The TFX executor will use the estimator provided in the `module_file` file
  to train the model.  The Transform executor will look specifically for the
  `preprocessing_fn()` function within that file.

  An example of `preprocessing_fn()` can be found in the [user-supplied
  code]((https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_utils.py))
  of the TFX Chicago Taxi pipeline example.

  ## Example
  ```
  # Performs transformations and feature engineering in training and serving.
  transform = Transform(
      examples=example_gen.outputs['examples'],
      schema=infer_schema.outputs['schema'],
      module_file=module_file)
  ```

  Please see https://www.tensorflow.org/tfx/transform for more details.
  """

  SPEC_CLASS = TransformSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(
      self,
      examples: types.Channel = None,
      schema: types.Channel = None,
      module_file: Optional[Union[Text, data_types.RuntimeParameter]] = None,
      preprocessing_fn: Optional[Union[Text,
                                       data_types.RuntimeParameter]] = None,
      transform_graph: Optional[types.Channel] = None,
      transformed_examples: Optional[types.Channel] = None,
      input_data: Optional[types.Channel] = None,
      instance_name: Optional[Text] = None,
      materialize: bool = True,
      custom_config: Optional[Dict[Text, Any]] = None):
    """Construct a Transform component.

    Args:
      examples: A Channel of type `standard_artifacts.Examples` (required).
        This should contain the two splits 'train' and 'eval'.
      schema: A Channel of type `standard_artifacts.Schema`. This should
        contain a single schema artifact.
      module_file: The file path to a python module file, from which the
        'preprocessing_fn' function will be loaded.
        Exactly one of 'module_file' or 'preprocessing_fn' must be supplied.

        The function needs to have the following signature:
        ```
        def preprocessing_fn(inputs: Dict[Text, Any]) -> Dict[Text, Any]:
          ...
        ```
        where the values of input and returned Dict are either tf.Tensor or
        tf.SparseTensor.

        If additional inputs are needed for preprocessing_fn, they can be passed
        in custom_config:

        ```
        def preprocessing_fn(inputs: Dict[Text, Any], custom_config:
                             Dict[Text, Any]) -> Dict[Text, Any]:
          ...
        ```
      preprocessing_fn: The path to python function that implements a
        'preprocessing_fn'. See 'module_file' for expected signature of the
        function. Exactly one of 'module_file' or 'preprocessing_fn' must be
        supplied.
      transform_graph: Optional output 'TransformPath' channel for output of
        'tf.Transform', which includes an exported Tensorflow graph suitable for
        both training and serving;
      transformed_examples: Optional output 'ExamplesPath' channel for
        materialized transformed examples, which includes both 'train' and
        'eval' splits.
      input_data: Backwards compatibility alias for the 'examples' argument.
      instance_name: Optional unique instance name. Necessary iff multiple
        transform components are declared in the same pipeline.
      materialize: If True, write transformed examples as an output. If False,
        `transformed_examples` must not be provided.
      custom_config: A dict which contains additional parameters that will be
        passed to preprocessing_fn.

    Raises:
      ValueError: When both or neither of 'module_file' and 'preprocessing_fn'
        is supplied.
    """
    if input_data:
      absl.logging.warning(
          'The "input_data" argument to the Transform component has '
          'been renamed to "examples" and is deprecated. Please update your '
          'usage as support for this argument will be removed soon.')
      examples = input_data
    if bool(module_file) == bool(preprocessing_fn):
      raise ValueError(
          "Exactly one of 'module_file' or 'preprocessing_fn' must be supplied."
      )

    transform_graph = transform_graph or types.Channel(
        type=standard_artifacts.TransformGraph,
        artifacts=[standard_artifacts.TransformGraph()])
    if materialize and transformed_examples is None:
      transformed_examples = types.Channel(
          type=standard_artifacts.Examples,
          # TODO(b/161548528): remove the hardcode artifact.
          artifacts=[standard_artifacts.Examples()],
          matching_channel_name='examples')
    elif not materialize and transformed_examples is not None:
      raise ValueError(
          'must not specify transformed_examples when materialize==False')
    spec = TransformSpec(
        examples=examples,
        schema=schema,
        module_file=module_file,
        preprocessing_fn=preprocessing_fn,
        transform_graph=transform_graph,
        transformed_examples=transformed_examples,
        custom_config=json.dumps(custom_config))
    super(Transform, self).__init__(spec=spec, instance_name=instance_name)
