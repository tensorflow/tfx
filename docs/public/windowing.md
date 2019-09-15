# Window Preprocessing

_An explanation of a specific CUJ for windowing, and a general discussion of the
need_

_robertcrowe@ 12 September 2019_

## Dataset and Modeling Goal

As part of creating training for TFX we received a sample of data which contains
tracking of services by number of unique customers. The data forms a
discontinuous time series with each example aggregating a 1 minute window. The
data represents the demand for services, which drives the need for staffing. The
modeling goal is to predict the demand one hour from now on frequent basis
(running inference every 10 minutes as being a good goal).

## Model

The time series is discontinuous, which is a common problem for time series
data. The gaps in the time series are a problem for sequence-based models for at
least two reasons:

-   There is a big difference between 10 unique customers in 8 hours versus 10
    unique customers in 8 minutes.
-   Predicting 12 steps ahead in the sequence will correspond to varying
    lengths of clock time.

There are at least two commonly used ways of dealing with the discontinuities:

-   If the time series examples are uniform (start and end period is consistent)
and if the period is appropriate, then we can insert zeroed examples into the
sequence to make it continuous.
-   If the time series examples are not uniform (start and end period is
inconsistent) or if the period is not appropriate, then we can aggregate the
data into continuous sliding windows.

This second is the more general approach, and was the option chosen.
A fairly generic model architecture was chosen and delivered good results:

```python
def _build_model():
    inputs = tf.keras.Input(shape=(SEQUENCE, 1))

    x1 = tf.keras.layers.Dropout(rate=DROPOUT_RATE)(inputs)
    x1 = tf.keras.layers.LSTM(LSTM_HIDDEN,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',)(x1)

    x2 = tf.keras.layers.Conv1D(filters=CONV_FILTERS,
      kernel_size=CONV_KERNEL,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros')(inputs)

    x2 = tf.keras.layers.Flatten()(x2)
    x = tf.keras.layers.concatenate([x1, x2])

    x = tf.keras.layers.Dense(DENSE_HIDDEN,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      activation='relu')(x)

    predictions = tf.keras.layers.Dense(1,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros')(x)

    model = tf.keras.Model(inputs=inputs, outputs=predictions)

    return model
```

## Training - Options for Preprocessing

This kind of preprocessing appears to be beyond the capabilities of Transform.
For scalability, data processing should use Beam, and Beam does support sliding
window aggregations. That suggests at least 3 options:

1.  Preprocess the data using Beam into sliding windows before entering the TFX
    pipeline.
1.  Preprocess the data using Beam into sliding windows in a custom ExampleGen.
1.  Preprocess the data using Beam into sliding windows in a custom component
    downstream of the standard ExampleGen.

During development a pure Python approach was developed as a temporary way to
proceed with model development while Beam code was being developed. That code
aggregated the data into sliding windows before entering the TFX pipeline. Beam
code was developed, but for the training session we were developing this for we
ran out of time to integrate it into the pipeline and there seemed to be
performance problems, so we went with the pure Python approach for the training.
Subsequently it appears that the performance problems were probably caused by
using the DirectRunner, which has performance limitations, and not the Beam
code.

## Inference - Options for Preprocessing

It’s less clear how to integrate this preprocessing for inference, and there
appears to be a significant potential for training/serving skew.

### TFX Inference Pipeline

A TFX pipeline could be built, ending in a component making client calls to a
Serving instance and including the same component as #2 or #3 above. Assuming
that pipeline latency would be acceptable (?) this approach would rely on source
control and consistent deployment to avoid training/serving skew. While not
ideal, this is probably the best approach currently available.

### Preprocessing Beam Pipeline

A Beam pipeline could be implemented to preprocess the data before delivering it
to the Serving client. A separate instance of this same pipeline could be used
to generate the training dataset by feeding training examples into it and
archiving the output before sending it to a TFX training pipeline. Since the
same pipeline could be used for both training and serving, this would avoid
training/serving skew.

### Specific General Need for Windowing

Sliding windows are a well accepted technique for working with discontinuous
time series data:

*   https://towardsdatascience.com/ml-approaches-for-time-series-4d44722e48fe
*   https://blog.statsbot.co/time-series-prediction-using-recurrent-neural-networks-lstms-807fa6ca7f
*   https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
*   https://www.altumintelligence.com/articles/a/Time-Series-Prediction-Using-LSTM-Deep-Neural-Networks
*   https://www.business-science.io/timeseries-analysis/2018/07/01/keras-lstm-sunspots-part2.html

We’ve also seen CUJs at Google which demonstrate a need for windowing. For
example, in the TFX Iteration 32 meeting there was a guest from the perception
team who described using TFMA for Video Object Detection and Tracking and
applying windowing to his data. We also have a current bug from gTech asking for
windowing in TFX.

### Overall Need for Non-Transform Processing

Transform is a very strong part of the TFX offering because of the advanced
distributed data processing which it enables, and the integration of the
transformations into the SavedModel. However, since the transformations must be
included in a SavedModel they are limited to only those operations that can be
implemented in TensorFlow Ops. Integration of the transformations into the
SavedModel prevents training/serving skew. When transformations are required
which cannot be implemented in TensorFlow Ops we currently have no design
patterns, best practices, or methodologies for avoiding training/serving skew.
This will mean that there are CUJs for which TFX implementations will include
the potential for training/serving skew.

### A Proposal

One possible way to address this problem is to create a “SavedPipeline”
specification, which would include:

*   A set of SavedModels
*   A set of Beam transformations
*   A dependency graph of the SavedModels and Beam transformations
*   A PipelineSpec that defines the inputs and outputs of the SavedPipeline

For TF.Serving-style deployments this might be implemented with a TFX
architecture including components which run inference. For TF.Lite and TFJS
-style deployments it is less clear how this would be implemented.

#### Some notes:

*   The primary CUJ for SavedPipelines would be for specifying inference
    pipelines, but could potentially also be used for specifying training
    pipelines.
*   The SavedModels and Beam transformations could both be framed as TFX
    components.
*   Because SavedPipelines includes a list of potentially multiple SavedModels
    it can also be used to specify ensembles.
*   SavedModels could also be used in other contexts, including TF.Hub and Model
    Garden, to specify pre- and post-processing requirements as well as
    ensembling of models.
