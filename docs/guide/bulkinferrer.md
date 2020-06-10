# The BulkInferrer TFX Pipeline Component

The BulkInferrer TFX component performs batch inference on unlabeled data. The
generated
InferenceResult([tensorflow_serving.apis.prediction_log_pb2.PredictionLog](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/prediction_log.proto))
contains the original features and the prediction results.

BulkInferrer consumes:

*   A Trained model in [SavedModel](/guide/saved_model) format.
*   Validation result from [Evaluator](evaluator) component.
*   Unlabeled tf.Examples that contain features.

BulkInferrer emits:

*   [InferenceResult](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/inference.proto)

## Using the BulkInferrer Component

A BulkInferrer TFX component is used to perform batch inference on unlabeled
tf.Examples. It is typically deployed after an [Evaluator](evaluator) component
to perform inference with a validated model, or after a [Trainer](trainer)
component to directly perform inference on exported model.

Typical code looks like this:

```python
from tfx import components

...

bulk_inferrer = components.BulkInferrer(
    examples=examples_gen.outputs['examples'],
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    data_spec=bulk_inferrer_pb2.DataSpec(),
    model_spec=bulk_inferrer_pb2.ModelSpec()
)
```

More details are available in the
[BulkInferrer API reference](/tfx/api_docs/python/tfx/components/BulkInferrer)
