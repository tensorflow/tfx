# The ModelValidator TFX Pipeline Component (Deprecated)

The ModelValidator was used to check if a model was good enough to be used in
production. We still think that validation is useful, but since the model
[Evaluator](evaluator.md) has already computed all the metrics you want to
validate against we decided to fuse the two so you don't have to duplicate the
computations.

While we have deprecated the ModelValidator and don't recommend it's use, if you
need to maintain an existing ModelValidator component an example configuration
is as follows:

```python
import tfx
import tensorflow_model_analysis as tfma
from tfx.components.model_validator.component import ModelValidator

...

model_validator = ModelValidator(
      examples=example_gen.outputs['output_data'],
      model=trainer.outputs['model'])
```

For those that would like to migrate the configuration to the Evaluator, a
similar configuration for the Evaluator would look as follows:

```python
from tfx import components
import tensorflow_model_analysis as tfma

...

eval_config = tfma.EvalConfig(
    model_specs=[
        # This assumes a serving model with signature 'serving_default'. If
        # using estimator based EvalSavedModel, add signature_name: 'eval' and
        # remove the label_key.
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
                tfma.MetricConfig(class_name='ExampleCount'),
                tfma.MetricConfig(
                    class_name='BinaryAccuracy',
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={'value': 0.5}),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value': -1e-10})))
            ]
        )
    ],
    slicing_specs=[
        # An empty slice spec means the overall slice, i.e. the whole dataset.
        tfma.SlicingSpec(),
        # Data can be sliced along a feature column. In this case, data is
        # sliced along feature column trip_start_hour.
        tfma.SlicingSpec(feature_keys=['trip_start_hour'])
    ])

model_resolver = Resolver(
      strategy_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
      model=Channel(type=Model),
      model_blessing=Channel(type=ModelBlessing)
).with_id('latest_blessed_model_resolver')

model_analyzer = components.Evaluator(
      examples=examples_gen.outputs['examples'],
      model=trainer.outputs['model'],
      baseline_model=model_resolver.outputs['model'],
      # Change threshold will be ignored if there is no baseline (first run).
      eval_config=eval_config)
```
