# CIFAR-10 Transfer Learning and MLKit integration Example

This example illustrates how to use Transfer Learning for image classification
with TFX, and use trained model to do object detection with
[MLKit](https://developers.google.com/ml-kit)

## Instruction

Create a Python 3 virtual environment for this example and activate the
`virtualenv`:

```
virtualenv -p python3.7 cifar10
source ./cifar10/bin/activate
```

Then, clone the tfx repo and copy cifar10/ folder to home directory:

```
git clone https://github.com/tensorflow/tfx ~/tfx-source && pushd ~/tfx-source
cp -r ~/tfx-source/tfx/examples/cifar10 ~/
```

Next, install the dependencies required by the CIFAR-10 example (appropriate
version of TF2 will be installed automatically).

```
pip install -e cifar10/
# The following is needed until tensorflow-model-analysis 0.23.0 is released
pip uinstall tensorflow-model-analysis
pip install git+https://github.com/tensorflow/model-analysis.git#egg=tensorflow_model_analysis
```

### Dataset

There is a subset of CIFAR10 (128 images) available in the data folder. To
prepare the whole dataset, first create a script and run the following Python
code: `import tensorflow_datasets as tfds ds = tfds.load('cifar10',
data_dir='./cifar10/data/',split=['train', 'test'])` Then, create sub-folders
for different dataset splits and move different splits to corresponding folders.
`cd cifar10/data mkdir train_whole mkdir test_whole mv
cifar10/3.0.2/cifar10-train.tfrecord-00000-of-00001 train_whole mv
cifar10/3.0.2/cifar10-test.tfrecord-00000-of-00001 test_whole` You'll find the
final dataset under `train_whole` and `test_whole` folders. Finally, clean up
the data folder. `rm -r cifar10`

### Train the model

Execute the pipeline python file : `python
~/cifar10/cifar_pipeline_native_keras.py` The trained model is located at
`~/cifar10/serving_model_lite/tflite`

This model is ready to be used for object detection with MLKit. Follow MLKit's
[documentation](https://developers.google.com/ml-kit/vision/object-detection/custom-models/android)
to set up an App and use it.

## Acknowledge Data Source

```
@TECHREPORT{Krizhevsky09learningmultiple,
    author = {Alex Krizhevsky},
    title = {Learning multiple layers of features from tiny images},
    institution = {},
    year = {2009}
}
```
