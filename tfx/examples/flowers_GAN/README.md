# Flowers GAN creation integration Example

This example illustrates how to set up a GAN (Generative Adversarial Network)
with TFX.

## Versions
The current version is a basic setup that shows how a GAN network can be 
trained with TFX. The implementation can help the user to see how a generator
and discriminator model can be constructed and combined into a GAN Tensorflow
class model. This can be found under `flowers_utils_keras.py`. Tensorboard
callbacks have also been implemented. The flowers dataset has been selected
as it is well-known and light-weight.

### TODOs
Several TODOs are still implemented in the code to update the current version
with improvements. For example:
* Hyperparameter tuning has not yet been implemented
* Model blessing has not yet been implemented
* Performance metrics can be worked out in more detail
* Runners other than a Local runner can be added.

## Instruction

Create a Python 3 virtual environment for this example and activate the
`virtualenv`:

```
virtualenv -p python3 flowersgan
source ./flowersgan/bin/activate
```

Then, clone the tfx repo and copy flowers_GAN/ folder to home directory:

```
git clone https://github.com/tensorflow/tfx ~/tfx-source && pushd ~/tfx-source
cp -r ~/tfx-source/tfx/examples/flowers_GAN ~/
```

Next, install the dependencies required by the flowers GAN example (appropriate
version of TF2 will be installed automatically).

```
pip install -e flowers_GAN/
```

### Dataset

A tfrecords dataset has been generated from the [Tensorflow flowers dataset](https://www.tensorflow.org/datasets/catalog/tf_flowers).
This dataset can be found in the ~/tfx-source/tfx/examples/flowers_GAN folder.
### Train the model

Execute the pipeline python file : `python
~/flowers_GAN/flower_pipeline_local.py` The trained model will be located at
`~/flowers_GAN/serving_model`