# The BulkInferrer TFX Pipeline Component

The BulkInferrer TFX component performs batch inference on unlabeled data. The
generated
InferenceResult([tensorflow_serving.apis.prediction_log_pb2.PredictionLog](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/prediction_log.proto))
contains the original features and the prediction results.

BulkInferrer consumes:

*   A trained model in
    [SavedModel](https://www.tensorflow.org/guide/saved_model.md) format.
*   Unlabelled tf.Examples that contain features.
*   (Optional) Validation result from
    [Evaluator](https://www.tensorflow.org/tfx/guide/evaluator.md) component.

BulkInferrer emits:

*   [InferenceResult](https://github.com/tensorflow/tfx/blob/master/tfx/types/standard_artifacts.py)

## Using the BulkInferrer Component

A BulkInferrer TFX component is used to perform batch inference on unlabeled
tf.Examples. It is typically deployed after an
[Evaluator](https://www.tensorflow.org/tfx/guide/evaluator.md) component to
perform inference with a validated model, or after a
[Trainer](https://www.tensorflow.org/tfx/guide/trainer.md) component to directly
perform inference on exported model.

It currently performs in-memory model inference and remote inference.
Remote inference requires the model to be hosted on Cloud AI Platform.

Typical code looks like this:

```python
bulk_inferrer = BulkInferrer(
    examples=examples_gen.outputs['examples'],
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    data_spec=bulk_inferrer_pb2.DataSpec(),
    model_spec=bulk_inferrer_pb2.ModelSpec()
)
```

More details are available in the
[BulkInferrer API reference](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/BulkInferrer).
