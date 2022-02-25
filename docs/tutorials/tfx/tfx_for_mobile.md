# TFX for Mobile

## Introduction

This guide demonstrates how Tensorflow Extended (TFX) can create and
evaluate machine learning models that will be deployed on-device. TFX now
provides native support for [TFLite](https://www.tensorflow.org/lite), which
makes it possible to perform highly efficient inference on mobile devices.

This guide walks you through the changes that can be made to any pipeline to
generate and evaluate TFLite models. We provide a complete example [here](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_pipeline_native_keras.py),
demonstrating how TFX can train and evaluate TFLite models that are
trained off of the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset. Further,
we show how the same pipeline can be used to simulataneously export both the
standard Keras-based [SavedModel](https://www.tensorflow.org/guide/saved_model)
as well as the TFLite one, allowing users to compare the quality of the two.

We assume you are familiar with TFX, our components, and our pipelines. If not,
then please see this [tutorial](https://www.tensorflow.org/tfx/tutorials/tfx/components).

## Steps
Only two steps are required to create and evaluate a TFLite model in TFX. The
first step is invoking the TFLite rewriter within the context of the
[TFX Trainer](https://www.tensorflow.org/tfx/guide/trainer) to convert the
trained TensorFlow model into a TFLite one. The second step is
configuring the Evaluator to evaluate TFLite models. We now discuss each in turn.

### Invoking the TFLite rewriter within the Trainer.
The TFX Trainer expects a user-defined `run_fn` to be specified in
a module file. This `run_fn` defines the model to be trained,
trains it for the specified number of iterations, and exports the trained model.

In the rest of this section, we provide code snippets which show the changes
required to invoke the TFLite rewriter and export a TFLite model. All of this
code is located in the `run_fn` of the [MNIST TFLite module](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_utils_native_keras_lite.py).

As shown in the code below,
we must first create a signature that takes a `Tensor` for every feature as
input. Note that this is a departure from most existing models in TFX, which take
serialized [tf.Example](https://www.tensorflow.org/api_docs/python/tf/train/Example)
protos as input.

```python
 signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(
              model, tf_transform_output).get_concrete_function(
                  tf.TensorSpec(
                      shape=[None, 784],
                      dtype=tf.float32,
                      name='image_floats'))
  }
```

Then the Keras model is saved as a SavedModel in the same way that it
normally is.

```python
  temp_saving_model_dir = os.path.join(fn_args.serving_model_dir, 'temp')
  model.save(temp_saving_model_dir, save_format='tf', signatures=signatures)
```

Finally, we create an instance of the TFLite rewriter (`tfrw`), and invoke it
on the SavedModel to obtain the TFLite model. We store this TFLite model
in the `serving_model_dir` provided by the caller of the `run_fn`.
This way, the TFLite model is stored in the location where all downstream TFX
components will be expecting to find the model.



```python
  tfrw = rewriter_factory.create_rewriter(
      rewriter_factory.TFLITE_REWRITER, name='tflite_rewriter')
  converters.rewrite_saved_model(temp_saving_model_dir,
                                 fn_args.serving_model_dir,
                                 tfrw,
                                 rewriter.ModelType.TFLITE_MODEL)
```


### Evaluating the TFLite model.
The [TFX Evaluator](https://www.tensorflow.org/tfx/guide/evaluator) provides the
ability to analyze trained models to understand their quality across a wide range
of metrics. In addition to analyzing SavedModels, the TFX Evaluator is now able
to analyze TFLite models as well.

The following code snippet (reproduced from the [MNIST pipeline](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_pipeline_native_keras.py)),
shows how to configure an Evaluator that analyzes a TFLite model.

```python
  # Informs the evaluator that the model is a TFLite model.
  eval_config_lite.model_specs[0].model_type = 'tf_lite'

  ...

  # Uses TFMA to compute the evaluation statistics over features of a TFLite
  # model.
  model_analyzer_lite = Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer_lite.outputs['model'],
      eval_config=eval_config_lite,
  ).with_id('mnist_lite')
```

As shown above, the only change that we need to make is to set the `model_type`
field to `tf_lite`. No other configuration changes are required to analyze the
TFLite model. Regardless of whether a TFLite model or the a SavedModel
is analyzed, the output of the `Evaluator` will have exactly the same structure.

However, please note that the Evaluator assumes that the TFLite model is saved
in a file named `tflite` within trainer_lite.outputs['model'].

