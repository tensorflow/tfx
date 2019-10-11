# CIFAR 10 Example

The CIFAR 10 example demonstrates the end-to-end workflow and steps of how to
analyze, validate and transform data, train a model, analyze and serve it. It
uses the following [TFX](https://www.tensorflow.org/tfx) components:

*   [ExampleGen](https://github.com/tensorflow/tfx/blob/master/docs/guide/examplegen.md)
    ingests and splits the input dataset.
*   [StatisticsGen](https://github.com/tensorflow/tfx/blob/master/docs/guide/statsgen.md)
    calculates statistics for the dataset.
*   [SchemaGen](https://github.com/tensorflow/tfx/blob/master/docs/guide/schemagen.md)
    SchemaGen examines the statistics and creates a data schema.
*   [ExampleValidator](https://github.com/tensorflow/tfx/blob/master/docs/guide/exampleval.md)
    looks for anomalies and missing values in the dataset.
*   [Transform](https://github.com/tensorflow/tfx/blob/master/docs/guide/transform.md)
    performs feature engineering on the dataset.
*   [Trainer](https://github.com/tensorflow/tfx/blob/master/docs/guide/trainer.md)
    trains the model using TensorFlow
    [Estimators](https://www.tensorflow.org/guide/estimators)
*   [Evaluator](https://github.com/tensorflow/tfx/blob/master/docs/guide/evaluator.md)
    performs deep analysis of the training results.
*   [ModelValidator](https://github.com/tensorflow/tfx/blob/master/docs/guide/modelval.md)
    ensures that the model is "good enough" to be pushed to production.
*   [Pusher](https://github.com/tensorflow/tfx/blob/master/docs/guide/pusher.md)
    deploys the model to a serving infrastructure.

## The dataset

This example uses the
[CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) released by the
The Canadian Institute for Advanced Research (CIFAR).

Note: This site provides applications using data that has been modified for use
from its original source, The Canadian Institute for Advanced Research (CIFAR).
The Canadian Institute for Advanced Research (CIFAR) makes no claims as to the
content, accuracy, timeliness, or completeness of any of the data provided at
this site. The data provided at this site is subject to change at any time. It
is understood that the data provided at this site is being used at one’s own
risk.

You can read more about the dataset in
[CIFAR dataset homepage](https://www.cs.toronto.edu/~kriz/cifar.html).

// TODO(ruoyu): Add instruction for generating the dataset.


# Learn more

Please see the
[TFX User Guide](https://github.com/tensorflow/tfx/blob/master/docs/guide/index.md)
to learn more.
