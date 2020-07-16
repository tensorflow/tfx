# CIFAR-10 Transfer Learning and MLKit integration Example
This example illustrates how to use Transfer Learning for image classification with TFX, and use trained model to do object detection with [MLKit](https://developers.google.com/ml-kit)

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
### Dataset
There is a subset of CIFAR10 (100 images) available in the data folder. To prepare the whole dataset, first create a script and run the following Python code:
```
import tensorflow_datasets as tfds
ds = tfds.load('cifar10', data_dir='./cifar10/data/',split=['train', 'test'])
```
Then, create sub-folders for different dataset splits and move different splits to corresponding folders.
```
cd cifar10/data
mkdir train_whole
mkdir test_whole
mv cifar10/3.0.2/cifar10-train.tfrecord-00000-of-00001 train_whole/
mv cifar10/3.0.2/cifar10-test.tfrecord-00000-of-00001 test_whole/
```
### Train the model
Execute the pipeline python file :
```
python ~/cifar10/cifar_pipeline_native_keras.py
```
The trained model is located at `~/cifar10/serving_model_lite/tflite`

### Add MetaData to the trained model
To use our trained model with MLKit, we need to add metadata to our model specifying the input's normalization strategy and output's label map. To do so, we need to 
first create a folder to store the final model with metadata:
```
mkdir ~/cifar10/exported
```	
Then, specify model's input and output information in _MODEL_INFO inside the `metadate_writer.py` script. In our case, it is 
```
_MODEL_INFO = {
    "cifar10.tflite":
        ModelSpecificInfo(
            name="MobileNetV1 image classifier on cifar 10",
            version="v1",
            image_width=224,
            image_height=224,
            image_min=0,
            image_max=255,
            mean=[127.5],
            std=[127.5],
            num_classes=10),
}
```
Finally, run the `metadata_writer.py` script to write the metadata into model
```
python ~/cifar10/meta_data_writer -model_file PATH_TO_MODEL -label_file data/labels.txt -export_directory exported
```	
The exported model with metadata can be find in the `exported` folder. This model is ready to be used for object detection with MLKit. Follow MLKit's [documentation](https://developers.google.com/ml-kit/vision/object-detection/custom-models/android)  to set up an App and use it. 
## Acknowledge Data Source
```
@TECHREPORT{Krizhevsky09learningmultiple,
    author = {Alex Krizhevsky},
    title = {Learning multiple layers of features from tiny images},
    institution = {},
    year = {2009}
}
```