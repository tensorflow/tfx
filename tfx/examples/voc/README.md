
#  VOC 2007 Object Detection Example with TF Object Detection API & MediaPipe integration

This example illustrates how to use Tensorflow Object Detection API for object detection with TFX, and use trained model to do object detection with
[MediaPipe]([https://google.github.io/mediapipe/](https://google.github.io/mediapipe/))

## Instruction

Create a Python 3 virtual environment for this example and activate the
`virtualenv`:

```
virtualenv -p python3.7 voc
source ./voc/bin/activate
```
Next, install the dependencies required by the VOC example (appropriate version of TF2 will be installed automatically).
```
pip install tfx
```
Then, clone the tfx repo and copy voc/ folder to home directory:

```
git clone https://github.com/tensorflow/tfx ~/tfx-source && pushd ~/tfx-source
cp -r ~/tfx-source/tfx/examples/voc ~/
```

Next, install the Tensorflow Object Detection API following this  [installation guide](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md#installation). 

Then, add the voc folder to the PYTHONPATH environment variable.
```
export PYTHONPATH=$PYTHONPATH:/path/to/voc/folder
```

### Dataset

There is a subset of VOC 2007 (256 train images, 100 test images) available in the data folder. To prepare the whole dataset, first create a script and run the following Python
code inside `voc` folder: 
```
import tensorflow_datasets as tfds 
ds = tfds.load('voc', data_dir='./data/',split=['train', 'validation'])
``` 
Then, create sub-folders
for different dataset splits and move different splits to corresponding folders.
```
cd data 
mkdir train_whole 
mkdir val_whole 
mv voc/2007/4.0.0/voc-train.tfrecord-00000-of-00002 train_whole/ 
mv voc/2007/4.0.0/voc-train.tfrecord-00001-of-00002 train_whole/
mv voc/2007/4.0.0/voc-validation.tfrecord-00000-of-00002 val_whole/ 
mv voc/2007/4.0.0/voc-validation.tfrecord-00000-of-00002 val_whole/ 
```
You'll find the final dataset under `train_whole` and `val_whole` folders. Finally, clean up the data folder. 
```rm -r voc```

The `pascal_label_map.pbtxt` file in the data folder comes from [this folder](https://github.com/tensorflow/models/tree/master/research/object_detection/data) in  the Tensorflow Object Detection repo, and you can grab label map files for other datasets if you want to train on them.

### Train the model
Execute the pipeline python file : `python
~/voc/voc_pipeline_native_keras.py` 
The trained model is located at
`~/voc/serving_model_lite/{MODEL_IDENTIFIER}/tflite`

This model is ready to be used for object detection with MediaPipe. Follow the "With a TFLite Model" section in [MediaPipe's object detection documentation](https://google.github.io/mediapipe/solutions/object_detection.html#video-file-input)  and use the graph configuration file in `mediapipe_graph` folder to run the trained model in MediaPipe. 

## Acknowledge Data Source

```
@misc{pascal-voc-2007,  
	author = "Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.",
	title = "The {PASCAL} {V}isual {O}bject {C}lasses {C}hallenge 2007 {(VOC2007)} {R}esults",
	howpublished = "http://www.pascal-network.org/challenges/VOC/voc2007/workshop/index.html"}
```