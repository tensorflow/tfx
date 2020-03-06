# The Evaluator TFX Pipeline Component

The Evaluator TFX pipeline component performs deep analysis on the training
results for your models, to help you understand how your model performs on
subsets of your data. The Evaluator also helps you validate your exported
models, ensuring that they are "good enough" to be pushed to production.

When validation is enabled, the Evaluator compares new models against a baseline
(such as the currently serving model) to determine if they're "good enough"
relative to the baseline. It does so by evaluating both models on an eval
dataset and computing their performance on metrics (e.g. AUC, loss). If the new
model's metrics meet developer-specified criteria relative to the baseline model
(e.g. AUC is not lower), the model is "blessed" (marked as good), indicating to
the [Pusher](pusher.md) that it is ok to push the model to production.

*   Consumes:
    *   An eval split from ExampleGen
    *   A trained model from [Trainer](trainer.md)
    *   A previously blessed model (if validation to be performed)
*   Emits:
    *   Analysis results to [ML Metadata](mlmd.md)
    *   Validation results to [ML Metadata](mlmd.md) (if validation to be
        performed)

## Evaluator and TensorFlow Model Analysis

Evaluator leverages the [TensorFlow Model Analysis](tfma.md) library to perform
the analysis, which in turn use [Apache Beam](beam.md) for scalable processing.

## Using the Evaluator Component

An Evaluator pipeline component is typically very easy to deploy and requires
little customization, since most of the work is done by the Evaluator TFX
component.

To setup the evaluator the following information is needed:

*   Metrics to configure (only reqired if additional metrics are being added
    outside of those saved with the model). See
    [Tensorflow Model Analysis Metrics](https://github.com/tensorflow/model-analysis/blob/master/g3doc/metrics.md)
    for more information.
*   Slices to configure (if not slices are given then an "overall" slice will be
    added by default). See
    [Tensorflow Model Analysis Setup](https://github.com/tensorflow/model-analysis/blob/master/g3doc/setup.md)
    for more information.

If validation is to be included, the following additional information is needed:

*   Which model to compare against (latest blessed, etc).
*   Model validations (thresholds) to verify. See
    [Tensorflow Model Analysis Model Validations](https://github.com/tensorflow/model-analysis/blob/master/g3doc/model_validations.md)
    for more information.

When enabled, validation will be performed against all of the metrics and slices
that were defined.

Typical code looks like this:

```python
from tfx import components
import tensorflow_model_analysis as tfma

...

# For TFMA evaluation

eval_config = tfma.EvalConfig(
    model_specs=[
        # This assumes a serving model with signature 'serving_default'. If
        # using estimator based EvalSavedModel, add signature_name='eval' and
        # remove the label_key. Note, if using a TFLite model, then you must set
        # model_type='tf_lite'.
        tfma.ModelSpec(label_key='<label_key>')
    ],
    metrics_specs=[
        tfma.MetricsSpec(
            # The metrics added here are in addition to those saved with the
            # model (assuming either a keras model or EvalSavedModel is used).
            # Any metrics added into the saved model (for example using
            # model.compile(..., metrics=[...]), etc) will be computed
            # automatically.
            metrics=[
                tfma.MetricConfig(class_name='ExampleCount')
            ],
            # To add validation thresholds for metrics saved with the model,
            # add them keyed by metric name to the thresholds map.
            thresholds = {
                "binary_accuracy": tfma.MetricThreshold(
                    value_threshold=tfma.GenericValueThreshold(
                        lower_bound={'value': 0.5}),
                    change_threshold=tfma.GenericChangeThreshold(
                       direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                       absolute={'value': -1e-10}))
            }
        )
    ],
    slicing_specs=[
        # An empty slice spec means the overall slice, i.e. the whole dataset.
        tfma.SlicingSpec(),
        # Data can be sliced along a feature column. In this case, data is
        # sliced along feature column trip_start_hour.
        tfma.SlicingSpec(feature_keys=['trip_start_hour'])
    ])

# The following component is experimental and may change in the future. This is
# required to specify the latest blessed model will be used as the baseline.
model_resolver = ResolverNode(
      instance_name='latest_blessed_model_resolver',
      resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
      model=Channel(type=Model),
      model_blessing=Channel(type=ModelBlessing))

model_analyzer = components.Evaluator(
      examples=examples_gen.outputs['examples'],
      model=trainer.outputs['model'],
      baseline_model=model_resolver.outputs['model'],
      # Change threshold will be ignored if there is no baseline (first run).
      eval_config=eval_config)
```
