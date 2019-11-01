# The BulkInferrer TFX Pipeline Component

The BulkInferrer TFX component performs offline batch processing
on a model with unlabelled inference requests.
The generated InferenceResult(
[tensorflow_serving.apis.prediction_log_pb2.PredictionLog](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/prediction_log.proto))
contains the original features and the prediction results.

BulkInferrer consumes:
*   A Trained model in [SavedModel](
https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/saved_model)
format.
*   Model validation result from
[ModelValidator](https://www.tensorflow.org/tfx/guide/modelval) component.
*   Unlabelled tf.Examples that contain features.

BulkInferrer emits:
[InferenceResult](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/inference.proto)

## Using the BulkInferrer Component

A BulkInferrer TFX component is used to perform batch inference on unlabelled
tf.Examples. It is typically deployed after a
[ModelValidator](https://www.tensorflow.org/tfx/guide/modelval)
component to perform inference with a validated model, or after a
[Trainer](https://www.tensorflow.org/tfx/guide/trainer)
component to directly perform inference on exported model.

Typical code looks like this:

```python
from tfx import components

...

bulk_inferrer = components.BulkInferrer(
      examples=examples_gen.outputs['examples'],
      model=trainer.outputs['model'],
      model_blessing=model_validator.outputs['blessing'],
      data_spec=bulk_inferrer_pb2.DataSpec(),
      model_spec=bulk_inferrer_pb2.ModelSpec()
      )
```
