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

from typing import Optional, Text
from tfx import types
from tfx.components.base import base_component
from tfx.components.base import executor_spec
from tfx.components.transform import executor
from tfx.types import artifact
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

  def __init__(self,
               examples: types.Channel = None,
               schema: types.Channel = None,
               module_file: Optional[Text] = None,
               preprocessing_fn: Optional[Text] = None,
               transform_graph: Optional[types.Channel] = None,
               transformed_examples: Optional[types.Channel] = None,
               input_data: Optional[types.Channel] = None,
               instance_name: Optional[Text] = None):
    """Construct a Transform component.

    Args:
      examples: A Channel of 'ExamplesPath' type (required). This should
        contain the two splits 'train' and 'eval'.
      schema: A Channel of 'SchemaPath' type. This should contain a single
        schema artifact.
      module_file: The file path to a python module file, from which the
        'preprocessing_fn' function will be loaded. The function must have the
        following signature.

        def preprocessing_fn(inputs: Dict[Text, Any]) -> Dict[Text, Any]:
          ...

        where the values of input and returned Dict are either tf.Tensor or
        tf.SparseTensor.  Exactly one of 'module_file' or 'preprocessing_fn'
        must be supplied.
      preprocessing_fn: The path to python function that implements a
         'preprocessing_fn'. See 'module_file' for expected signature of the
         function. Exactly one of 'module_file' or 'preprocessing_fn' must
         be supplied.
      transform_graph: Optional output 'TransformPath' channel for output of
        'tf.Transform', which includes an exported Tensorflow graph suitable for
        both training and serving;
      transformed_examples: Optional output 'ExamplesPath' channel for
        materialized transformed examples, which includes both 'train' and
        'eval' splits.
      input_data: Backwards compatibility alias for the 'examples' argument.
      instance_name: Optional unique instance name. Necessary iff multiple
        transform components are declared in the same pipeline.

    Raises:
      ValueError: When both or neither of 'module_file' and 'preprocessing_fn'
        is supplied.
    """
    examples = examples or input_data
    if bool(module_file) == bool(preprocessing_fn):
      raise ValueError(
          "Exactly one of 'module_file' or 'preprocessing_fn' must be supplied."
      )

    transform_graph = transform_graph or types.Channel(
        type=standard_artifacts.TransformGraph,
        artifacts=[standard_artifacts.TransformGraph()])
    transformed_examples = transformed_examples or types.Channel(
        type=standard_artifacts.Examples,
        artifacts=[
            standard_artifacts.Examples(split=split)
            for split in artifact.DEFAULT_EXAMPLE_SPLITS
        ])
    spec = TransformSpec(
        input_data=examples,
        schema=schema,
        module_file=module_file,
        preprocessing_fn=preprocessing_fn,
        transform_output=transform_graph,
        transformed_examples=transformed_examples)
    super(Transform, self).__init__(spec=spec, instance_name=instance_name)
