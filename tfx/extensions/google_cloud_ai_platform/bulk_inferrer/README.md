# Google Cloud AI Platform BulkInferrer Component

Google Cloud AI Platform BulkInferrer component performs batch inference on
unlabeled data using the model hosted on
[Cloud AI Platform](https://cloud.google.com/ai-platform/prediction/docs/online-predict).
It generates
InferenceResult([tensorflow_serving.apis.prediction_log_pb2.PredictionLog](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/prediction_log.proto))
containing the prediction results.

Google Cloud AI Platform BulkInferrer consumes:

*   A trained model in
    [SavedModel](https://www.tensorflow.org/guide/saved_model.md) format.
*   Unlabelled tf.Examples that contain features.
*   (Optional) Validation result from
    [Evaluator](https://www.tensorflow.org/tfx/guide/evaluator.md) component.

Google Cloud AI Platform BulkInferrer emits:

*   [InferenceResult](https://github.com/tensorflow/tfx/blob/master/tfx/types/standard_artifacts.py)

## Using the Google Cloud AI Platform BulkInferrer Component
Google Cloud AI Platform BulkInferrer component is used to perform batch
inference on unlabeled tf.Examples.

This is different from the standard TFX [BulkInferrer component](https://github.com/tensorflow/tfx/blob/master/docs/guide/bulkinferrer.md)
for 2 reasons:

*   This component will create a new model (if necessary) and a new model
    version on Google Cloud AI Platform before doing inference. After inference,
    it will clean up the newly created resources.
*   This BulkInferrer component only sends inference request to a model that is
    hosted on Google Cloud AI Platform.

TODO(b/155325467): Consolidate the documentation with guide of
Cloud AI components.
