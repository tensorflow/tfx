# Penguin Classification Example

The Penguin classification example introduces the TFX programming
environment and shows you how to solve a classification problem using
TFX.

The Penguin classification example demonstrates the end-to-end workflow
and steps of how to classify penguin species.
The [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/articles/intro.html) dataset can be found on [Github](https://github.com/allisonhorst/palmerpenguins).

The raw data contains some categorical data and has incomplete data rows.

This example will use
[processed Palmer Penguins dataset](https://storage.googleapis.com/download.tensorflow.org/data/palmer_penguins/penguins_processed.csv):
* Leaves out any incomplete records
* Drops `island` and `sex`, the two categorical columns
* Converts the labels to `int32`.

## Instruction

NOTE: This examples requires TFX 0.25.0 or later.

Create a Python 3 virtual environment for this example and activate the
`virtualenv`:

<pre class="devsite-terminal devsite-click-to-copy">
virtualenv -p python3.7 penguin
source ./penguin/bin/activate
</pre>

Next, install the dependencies required by the penguin example:

<pre class="devsite-terminal devsite-click-to-copy">
pip install -U tfx[examples]
</pre>

Then, clone the tfx repo and copy penguin/ folder to home directory:

<pre class="devsite-terminal devsite-click-to-copy">
git clone https://github.com/tensorflow/tfx ~/tfx-source && pushd ~/tfx-source
cp -r ~/tfx-source/tfx/examples/penguin ~/
</pre>

Execute the pipeline python file and output can be found at `~/tfx`:

<pre class="devsite-terminal devsite-click-to-copy">
python ~/penguin/penguin_pipeline_local.py
</pre>

## Instructions for using Flax

**The Flax support in TFX is experimental.**

The above instructions will use Keras for model definition and training.
Aditionally, you can use essentially the same TFX configuration with
[Flax](https://github.com/google/flax) models. The bulk of the differences
from using a Keras model are in `penguin_utils_flax_experimental.py`.

To execute the pipeline with a Flax model:
<pre class="devsite-terminal devsite-click-to-copy">
python ~/penguin/penguin_pipeline_local.py --model_framework=flax_experimental
</pre>
