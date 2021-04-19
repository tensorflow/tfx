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

from typing import Any, Dict, Optional, Text, Union

from tfx import types
from tfx.components.transform import executor
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import executor_spec
from tfx.orchestration import data_types
from tfx.proto import transform_pb2
from tfx.types import standard_artifacts
from tfx.types.standard_component_specs import TransformSpec
from tfx.utils import json_utils


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
      splits_config: transform_pb2.SplitsConfig = None,
      transform_graph: Optional[types.Channel] = None,
      transformed_examples: Optional[types.Channel] = None,
      analyzer_cache: Optional[types.Channel] = None,
      instance_name: Optional[Text] = None,
      materialize: bool = True,
      disable_analyzer_cache: bool = False,
      force_tf_compat_v1: bool = True,
      custom_config: Optional[Dict[Text, Any]] = None):
    """Construct a Transform component.

    Args:
      examples: A Channel of type `standard_artifacts.Examples` (required).
        This should contain custom splits specified in splits_config. If
        custom split is not provided, this should contain two splits 'train'
        and 'eval'.
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
      splits_config: A transform_pb2.SplitsConfig instance, providing splits
        that should be analyzed and splits that should be transformed. Note
        analyze and transform splits can have overlap. Default behavior (when
        splits_config is not set) is analyze the 'train' split and transform
        all splits. If splits_config is set, analyze cannot be empty.
      transform_graph: Optional output 'TransformPath' channel for output of
        'tf.Transform', which includes an exported Tensorflow graph suitable for
        both training and serving;
      transformed_examples: Optional output 'ExamplesPath' channel for
        materialized transformed examples, which includes transform splits as
        specified in splits_config. If custom split is not provided, this should
        include both 'train' and 'eval' splits.
      analyzer_cache: Optional input 'TransformCache' channel containing
        cached information from previous Transform runs. When provided,
        Transform will try use the cached calculation if possible.
      instance_name: Optional unique instance name. Necessary iff multiple
        transform components are declared in the same pipeline.
      materialize: If True, write transformed examples as an output. If False,
        `transformed_examples` must not be provided.
      disable_analyzer_cache: If False, Transform will use input cache if
        provided and write cache output. If True, `analyzer_cache` must not be
        provided.
      force_tf_compat_v1: (Optional) If True, Transform will use Tensorflow in
        compat.v1 mode irrespective of installed version of Tensorflow. Defaults
        to `True`. Note: The default value will be switched to `False` in a
        future release.
      custom_config: A dict which contains additional parameters that will be
        passed to preprocessing_fn.

    Raises:
      ValueError: When both or neither of 'module_file' and 'preprocessing_fn'
        is supplied.
    """
    if bool(module_file) == bool(preprocessing_fn):
      raise ValueError(
          "Exactly one of 'module_file' or 'preprocessing_fn' must be supplied."
      )

    transform_graph = transform_graph or types.Channel(
        type=standard_artifacts.TransformGraph)

    if materialize and transformed_examples is None:
      transformed_examples = types.Channel(
          type=standard_artifacts.Examples,
          matching_channel_name='examples')
    elif not materialize and transformed_examples is not None:
      raise ValueError(
          'Must not specify transformed_examples when materialize is False.')

    if disable_analyzer_cache:
      updated_analyzer_cache = None
      if analyzer_cache:
        raise ValueError(
            '`analyzer_cache` is set when disable_analyzer_cache is True.')
    else:
      updated_analyzer_cache = types.Channel(
          type=standard_artifacts.TransformCache)

    spec = TransformSpec(
        examples=examples,
        schema=schema,
        module_file=module_file,
        preprocessing_fn=preprocessing_fn,
        force_tf_compat_v1=int(force_tf_compat_v1),
        splits_config=splits_config,
        transform_graph=transform_graph,
        transformed_examples=transformed_examples,
        analyzer_cache=analyzer_cache,
        updated_analyzer_cache=updated_analyzer_cache,
        custom_config=json_utils.dumps(custom_config))
    super(Transform, self).__init__(spec=spec, instance_name=instance_name)
