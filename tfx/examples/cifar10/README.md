
# CIFAR-10 Transfer Learning Example
This example illustrates how to use Transfer Learning for image classification with TFX.

## Instruction

Create a Python 3 virtual environment for this example and activate the
`virtualenv`:

```
virtualenv -p python3.5 cifar10
source ./cifar10/bin/activate
```

Next, install the dependencies required by the CIFAR-10 example (appropriate
version of TF2 will be installed automatically).

```
pip install tfx
```

Then, clone the tfx repo and copy cifar10/ folder to home directory:

```
git clone https://github.com/tensorflow/tfx ~/tfx-source && pushd ~/tfx-source
cp -r ~/tfx-source/tfx/examples/cifar10 ~/
```
Next, download the dataset (in Python):
```
import tensorflow_datasets as tfds
ds = tfds.load('cifar10', data_dir='./cifar10/data/',split=['train', 'test'])
```
Then, create sub-folders for different dataset splits and move different splits to corresponding folders.
```
cd cifar10/data
mkdir train
mkdir test
mv ciar10/3.0.2/cifar10-train.tfrecord-00000-of-00001 train/
mv ciar10/3.0.2/cifar10-test.tfrecord-00000-of-00001 test/
```
Execute the pipeline python file and output can be found at `~/tfx`:

```
python ~/cifar10/cifar_pipeline_native_keras.py
```
## Acknowledge Data Source
```
@TECHREPORT{Krizhevsky09learningmultiple,
    author = {Alex Krizhevsky},
    title = {Learning multiple layers of features from tiny images},
    institution = {},
    year = {2009}
}
```