# Native Keras training example for MNIST

This example illustrates the use of Native Keras APIs in TF2. It also demonstrates how TFLite is supported in TFX.

Here is short summary of the example codes, status and known restrictions:

*   This example does not use the Estimator API for model training/evaluation.

*   The trainer must be configured with TFX [Generic Trainer]
    (https://github.com/tensorflow/community/blob/master/rfcs/20200117-tfx-generic-trainer.md).

*   The dataset included in this example consists of a selection of 1000 records
    from the MNIST dataset, converted to tfrecord format. Each record is
    a tf.Example with 2 columns of data: 'image_floats' representing
    the 28x28 image as 784 float values in the range [-0.5, 0.5], and
    'image_class' representing the label with values [0-9] corresponding to
    decimal digit in the image.

    For a detailed description of tf.Example and TFRecord see
    https://www.tensorflow.org/tutorials/load_data/tfrecord. Several
    independently written tools can be used to generate the tfrecords from
    images. See [Preparing MNIST
    data](https://docs.databricks.com/_static/notebooks/deep-learning/mnist-tensorflow-to-tfrecords.html)
    from one such publicly available set of instructions.

## Instruction

Create a Python 3 virtual environment for this example and activate the
`virtualenv`:

<pre class="devsite-terminal devsite-click-to-copy">
virtualenv -p python3.5 mnist
source ./mnist/bin/activate
</pre>

Next, install the dependencies required by the MNIST example (appropriate
version of TF2 will be installed automatically).

<pre class="devsite-terminal devsite-click-to-copy">
pip install -U tfx[examples]
</pre>

Then, clone the tfx repo and copy mnist/ folder to home directory:

<pre class="devsite-terminal devsite-click-to-copy">
git clone https://github.com/tensorflow/tfx ~/tfx-source && pushd ~/tfx-source
cp -r ~/tfx-source/tfx/examples/mnist ~/
</pre>

Execute the pipeline python file and output can be found at `~/tfx`:

<pre class="devsite-terminal devsite-click-to-copy">
python ~/mnist/mnist_pipeline_native_keras.py
</pre>
