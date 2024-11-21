# Designing TensorFlow Modeling Code For TFX

When designing your TensorFlow modeling code for TFX there are a few items to be
aware of, including the choice of a modeling API.

* Consumes: SavedModel from [Transform](transform.md), and data from
[ExampleGen](examplegen.md)
* Emits: Trained model in SavedModel format

!!! note

    TFX supports nearly all of TensorFlow 2.X, with minor exceptions. TFX also fully supports TensorFlow 1.15.

    - New TFX pipelines should use TensorFlow 2.x with Keras models via the [Generic Trainer](https://github.com/tensorflow/community/blob/master/rfcs/20200117-tfx-generic-trainer.md").
    - Full support for TensorFlow 2.X, including improved support for tf.distribute, will be added incrementally in upcoming releases.
    - Previous TFX pipelines can continue to use TensorFlow 1.15. To switch them to TensorFlow 2.X, see the [TensorFlow migration guide](https://www.tensorflow.org/guide/migrate).

    To keep up to date on TFX releases, see the [TFX OSS Roadmap](https://github.com/tensorflow/tfx/blob/master/ROADMAP.md), read [the TFX blog](https://blog.tensorflow.org/search?label=TFX&max-results=20) and subscribe to the [TensorFlow newsletter](https://services.google.com/fb/forms/tensorflow/).

Your model's input layer should consume from the SavedModel that was created by
a [Transform](transform.md) component, and the layers of the Transform model should
be included with your model so that when you export your SavedModel and
EvalSavedModel they will include the transformations that were created by the
[Transform](transform.md) component.
