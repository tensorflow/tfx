# Model Rewriting Library

The TFX model rewriting library is designed to be invoked from the trainer
component to perform post-training modifications to the model (i.e. rewrites).
This library is designed to simplify the development and use of custom rewriters
that satisfy specific user needs. In addition, we provide a set of commonly-used
rewriters that can be readily-used.

## Using rewriters
To instantiate a rewriter, use the rewriter factory.

```python
from tfx.components.trainer.rewriting import rewriter_factory

...

tfrw = rewriter_factory.create_rewriter(
    rewriter_factory.TFLITE_REWRITER, name='my_rewriter')
```

Then use the appropriate converter (`RewritingExporter` for Estimators or
`rewrite_saved_model` for TF2) to rewrite your model. We recommend you invoke
these converters in the `trainer_fn` definition in the utils file of your
pipeline. For example, for the chicago taxi pipeline, this would be the
taxi_utils.py [file](https://github.com/tensorflow/tfx/blob/r0.21/tfx/examples/chicago_taxi_pipeline/taxi_utils.py).

If using Estimators, this would look as follows:

```python
import tensorflow as tf
from tfx.components.trainer.rewriting import converters

...

base_exporter = tf.estimator.FinalExporter('chicago-taxi', serving_receiver_fn)
rewriting_exporter = converters.RewritingExporter(base_exporter, tfrw)
eval_spec = tf.estimator.EvalSpec(
    eval_input_fn,
    steps=trainer_fn_args.eval_steps,
    exporters=[rewriting_exporter],
    name='chicago-taxi-eval')
```
For the case of TF2, this would look as follows:

```python
import tensorflow as tf
from tfx.components.trainer.rewriting import converters

...

tf.saved_model.save(model, '/path/to/model')
converters.rewrite_saved_model('/path/to/model', '/path/to/rewritten/model',
                               tfrw)
```

## Creating new rewriters

To create new rewriters, simply take the following steps:

* Define a rewriter that inherits from `BaseRewriter` in rewriter.py.

* Import the rewriter and add a constant to rewriter_factory.py.
