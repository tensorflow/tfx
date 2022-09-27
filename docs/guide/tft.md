# Transform library for non-TFX users

Transform is available as a standalone library.

-   [Getting Started with TensorFlow Transform](https://www.tensorflow.org/tfx/transform/get_started)
-   [TensorFlow Transform API Reference](https://www.tensorflow.org/tfx/transform/api_docs/python/tft)

The `tft` module documentation is the only module that is relevant to TFX users.
The `tft_beam` module is relevant only when using Transform as a standalone library. Typically, a TFX user constructs a `preprocessing_fn`, and the rest of the
Transform library calls are made by the Transform component.

