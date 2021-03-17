# TensorFlow Ranking Example

This example pipeline has two extra dependencies:

- `struct2tensor`: for decoding the training data
  (which is not in tf.Example, but a different proto, ExampleListWithContext)
- `tensorflow_ranking` for ranking specific keras layers, losses and metrics.

To try this example, install tfx using

```
pip install -U tfx[examples]
```

This file will be updated with more information.

<!-- TODO(b/179804018): update the documentation.-->
