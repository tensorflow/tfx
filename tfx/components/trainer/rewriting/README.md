# Model Rewriting Library

The TFX model rewriting library makes it simple to make post-training
modifications (i.e. rewrites) to models within TFX. These modifications can vary
from small-scale edits (e.g. signature changes) to wholesale model conversions
from one type to another (e.g. from SavedModel to
[TFLite](https://www.tensorflow.org/lite)).

The library is invoked from user code in the Trainer. We both make it simple to
create custom rewrites and provide a set of commonly-used ones. For example,
the
[TFLiteRewriter](https://github.com/tensorflow/tfx/blob/master/tfx/components/trainer/rewriting/tflite_rewriter.py)
converts SavedModels to TFLite.

## Using rewriters
To instantiate a rewriter, use the rewriter factory.

```python
from tfx.components.trainer.rewriting import rewriter_factory

...

tfrw = rewriter_factory.create_rewriter(
    rewriter_factory.TFLITE_REWRITER, name='my_rewriter')
```

Then use the appropriate converter (`RewritingExporter` for Estimators or
`rewrite_saved_model` for Keras) to rewrite your model.

When using Estimators, we recommend you invoke these converters in the
`trainer_fn` definition in the utils file of your pipeline. For example, in the
chicago taxi pipeline, this would be the taxi_utils.py
[file](https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_utils.py)
and the changes would be as follows:

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
For Keras, we recommend you invoke these converters in the `run_fn` definition
in the utils file of your pipeline. For example, for the MNIST pipeline, this
would be the mnist_utils_native_keras_lite.py
[file](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_utils_native_keras_lite.py)
and the changes would be as follows:

```python
import tensorflow as tf
from tfx.components.trainer.rewriting import converters

...

model.save('/path/to/model', save_format='tf', signatures=signatures)
converters.rewrite_saved_model('/path/to/model', '/path/to/rewritten/model',
                               tfrw)
```
A complete end-to-end pipeline that uses the TFLite rewriter can be found [here](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_pipeline_native_keras.py).


## Creating new rewriters

To create new rewriters, simply take the following steps:

* Define a rewriter that inherits from `BaseRewriter` in rewriter.py.

* Import the rewriter and add a constant to rewriter_factory.py.
