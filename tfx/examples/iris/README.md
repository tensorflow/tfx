# Iris flowers classification Example

The Iris flowers classification example introduces the TFX programming
environment and shows you how to solve the classification problem in
TensorFlow TFX.

The Iris flowers classification example demonstrates the end-to-end workflow
and steps of how to classify Iris flower subspecies.

## Instruction

Create a Python 3 virtual environment for this example and activate the
`virtualenv`:

<pre class="devsite-terminal devsite-click-to-copy">
virtualenv -p python3.5 iris
source ./iris/bin/activate
</pre>

Next, install the dependencies required by the Iris example:

<pre class="devsite-terminal devsite-click-to-copy">
pip install tfx
</pre>

Then, clone the tfx repo and copy iris/ folder to home directory:

<pre class="devsite-terminal devsite-click-to-copy">
git clone https://github.com/tensorflow/tfx ~/tfx-source && pushd ~/tfx-source
cp -r ~/tfx-source/tfx/examples/iris ~/
</pre>

Execute the pipeline python file and output can be found at `~/tfx`:

<pre class="devsite-terminal devsite-click-to-copy">
python ~/iris/iris_pipeline_XXX.py
</pre>
