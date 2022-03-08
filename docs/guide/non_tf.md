# Using Other ML Frameworks in TFX

TFX as a platform is framework neutral, and can be used with other ML
frameworks, e.g., JAX, scikit-learn.

For model developers, this means they do not need to rewrite their model
code implemented in another ML framework, but can instead reuse the bulk of the
training code as-is in TFX, and benefit from other capabilities TFX and the
rest of the TensorFlow Ecosystem offers.

The TFX pipeline SDK and most modules in TFX, e.g., pipeline orchestrator,
don't have any direct dependency on TensorFlow, but there are some aspects
which are oriented towards TensorFlow, such as data formats. With some
consideration of the needs of a particular modeling framework, a TFX pipeline
can be used to train models in any other Python-based ML framework. This includes
Scikit-learn, XGBoost, and PyTorch, among others. Some of the considerations for
using the standard TFX components with other frameworks include:

*   **ExampleGen** outputs
    [tf.train.Example](https://www.tensorflow.org/tutorials/load_data/tfrecord)
    in TFRecord files. It's a generic representation for training data, and
    downstream components use
    [TFXIO](https://github.com/tensorflow/community/blob/master/rfcs/20191017-tfx-standardized-inputs.md)
    to read it as Arrow/RecordBatch in memory, which can be further converted to
    `tf.dataset`, `Tensors` or other formats. Payload/File formats other than
    tf.train.Example/TFRecord are being considered, but for TFXIO users it
    should be a blackbox.
*   **Transform** can be used to generate transformed training examples no
    matter what framework is used for training, but if the model format is not
    `saved_model`, users won't be able to embed the transform graph into the
    model. In that case, model prediction needs to take transformed features
    instead of raw features, and users can run transform as a preprocessing
    step before calling the model prediction when serving.
*   **Trainer** supports
    [GenericTraining](https://www.tensorflow.org/tfx/guide/trainer#generic_trainer)
    so users can train their models using any ML framework.
*   **Evaluator** by default only supports `saved_model`, but users can provide
    a UDF that generates predictions for model evaluation.

Training a model in a non-Python-based framework will require isolating a
custom training component in a Docker container, as part of a pipeline which is
running in a containerized environment such as Kubernetes.

## JAX

[JAX](https://github.com/google/jax) is Autograd and XLA, brought together for
high-performance machine learning research.
[Flax](https://github.com/google/flax)
is a neural network library and ecosystem for JAX, designed for flexibility.

With [jax2tf](https://github.com/google/jax/tree/main/jax/experimental/jax2tf),
we are able to convert trained JAX/Flax models into `saved_model` format, 
which can be used seamlessly in TFX with generic training and model evaluation.
For details, check this [example](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_utils_flax_experimental.py).

## scikit-learn

[Scikit-learn](https://scikit-learn.org/stable/) is a machine learning library
for the Python programming language. We have an e2e
[example](https://github.com/tensorflow/tfx-addons/tree/main/examples/sklearn_penguins)
with customized training and evaluation in TFX-Addons.
