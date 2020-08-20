# The ExampleGen TFX Pipeline Component

The ExampleGen TFX Pipeline component ingests data into TFX pipelines. It
consumes external files/services to generate Examples which will be read by
other TFX components. It also provides consistent and configurable partition,
and shuffles the dataset for ML best practice.

*   Consumes: Data from external data sources such as CSV, `TFRecord` and BigQuery
*   Emits: `tf.Example` records

## ExampleGen and Other Components

ExampleGen provides data to components that make use of the
[TensorFlow Data Validation](tfdv.md) library, such as
[SchemaGen](schemagen.md), [StatisticsGen](statsgen.md), and
[Example Validator](exampleval.md). It also provides data to
[Transform](transform.md), which makes use of the [TensorFlow Transform](tft.md)
library, and ultimately to deployment targets during inference.

## How to use an ExampleGen Component

For supported data sources (currently, CSV files, TFRecord files with TF Example
data format, and results of BigQuery queries) the ExampleGen pipeline component
is typically very easy to deploy and requires little customization. Typical code
looks like this:

```python
from tfx.utils.dsl_utils import csv_input
from tfx.components.example_gen.csv_example_gen.component import CsvExampleGen

examples = csv_input(os.path.join(base_dir, 'data/simple'))
example_gen = CsvExampleGen(input=examples)
```

or like below for importing external tf Examples directly:

```python
from tfx.utils.dsl_utils import tfrecord_input
from tfx.components.example_gen.import_example_gen.component import ImportExampleGen

examples = tfrecord_input(path_to_tfrecord_dir)
example_gen = ImportExampleGen(input=examples)
```

## Span, Version and Split

A Span is a grouping of training examples. If your data is persisted on a
filesystem, each Span may be stored in a separate directory. The semantics of a
Span are not hardcoded into TFX; a Span may correspond to a day of data, an hour
of data, or any other grouping that is meaningful to your task.

Each Span can hold multiple Versions of data. To give an example, if you remove
some examples from a Span to clean up poor quality data, this could result in a
new Version of that Span. By default, TFX components operate on the latest
Version within a Span.

Each Version within a Span can further be subdivided into multiple Splits. The
most common use-case for splitting a Span is to split it into training and eval
data.

![Spans and Splits](images/spans_splits.png)

### Custom input/output split

Note: this feature is only available after TFX 0.14.

To customize the train/eval split ratio which ExampleGen will output, set the
`output_config` for ExampleGen component. For example:

```python
from  tfx.proto import example_gen_pb2

# Input has a single split 'input_dir/*'.
# Output 2 splits: train:eval=3:1.
output = example_gen_pb2.Output(
             split_config=example_gen_pb2.SplitConfig(splits=[
                 example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=3),
                 example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1)
             ]))
examples = csv_input(input_dir)
example_gen = CsvExampleGen(input=examples, output_config=output)
```

Notice how the `hash_buckets` were set in this example.

For an input source which has already been split, set the `input_config` for
ExampleGen component:

```python
from  tfx.proto import example_gen_pb2

# Input train split is 'input_dir/train/*', eval split is 'input_dir/eval/*'.
# Output splits are generated one-to-one mapping from input splits.
input = example_gen_pb2.Input(splits=[
                example_gen_pb2.Input.Split(name='train', pattern='train/*'),
                example_gen_pb2.Input.Split(name='eval', pattern='eval/*')
            ])
examples = csv_input(input_dir)
example_gen = CsvExampleGen(input=examples, input_config=input)
```

For file based example gen (e.g. CsvExampleGen and ImportExampleGen), `pattern`
is a glob relative file pattern that maps to input files with root directory
given by input base path. For query-based example gen (e.g. BigQueryExampleGen,
PrestoExampleGen), `pattern` is a SQL query.

By default, the entire input base dir is treated as a single input split, and
the train and eval output split is generated with a 2:1 ratio.

Please refer to
[proto/example_gen.proto](https://github.com/tensorflow/tfx/blob/master/tfx/proto/example_gen.proto)
for details.

### Splitting Method

When using `hash_buckets` splitting method, instead of the entire record, one
can use a feature for partitioning the examples. If a feature is present,
ExampleGen will use a fingerprint of that feature as the partition key.

This feature can be used to maintain a stable split w.r.t. certain properties of
examples: for example, a user will always be put in the same split if "user_id"
were selected as the partition feature name.

The interpretation of what a "feature" means and how to match a "feature" with
the specified name depends on the ExampleGen implementation and the type of the
examples.

For ready-made ExampleGen implementations:

*   If it generates tf.Example, then a "feature" means an entry in
    tf.Example.features.feature.
*   If it generates tf.SequenceExample, then a "feature" means an entry in
    tf.SequenceExample.context.feature.
*   Only int64 and bytes features are supported.

In the following cases, ExampleGen throws runtime errors:

*   Specified feature name does not exist in the example.
*   Empty feature: `tf.train.Feature()`.
*   Non supported feature types, e.g., float features.

To output the train/eval split based on a feature in the examples, set the
`output_config` for ExampleGen component. For example:

```python
from  tfx.proto import example_gen_pb2

# Input has a single split 'input_dir/*'.
# Output 2 splits based on 'user_id' features: train:eval=3:1.
output = example_gen_pb2.Output(
             split_config=example_gen_pb2.SplitConfig(splits=[
                 example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=3),
                 example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1)
             ],
             partition_feature_name='user_id'))
examples = csv_input(input_dir)
example_gen = CsvExampleGen(input=examples, output_config=output)
```

Notice how the `partition_feature_name` was set in this example.

### Span

Note: this feature is only available after TFX 0.15.

Span can be retrieved by using '{SPAN}' spec in the
[input glob pattern](https://github.com/tensorflow/tfx/blob/master/tfx/proto/example_gen.proto):

*   This spec matches digits and maps the data into the relevant SPAN numbers.
    For example, 'data_{SPAN}-*.tfrecord' will collect files like
    'data_12-a.tfrecord', 'date_12-b.tfrecord'.
*   When SPAN spec is missing, it's assumed to be always Span '0'.
*   If SPAN is specified, pipeline will process the latest span, and store the
    span number in metadata

For example, let's assume there are input data:

*   '/tmp/span-01/train/data'
*   '/tmp/span-01/eval/data'
*   '/tmp/span-02/train/data'
*   '/tmp/span-02/eval/data'

and the input config is shown as below:

```python
splits {
  name: 'train'
  pattern: 'span-{SPAN}/train/*'
}
splits {
  name: 'eval'
  pattern: 'span-{SPAN}/eval/*'
}
```

when triggering the pipeline, it will process:

*   '/tmp/span-02/train/data' as train split
*   '/tmp/span-02/eval/data' as eval split

with span number as '02'. If later on '/tmp/span-03/...' are ready, simply
trigger the pipeline again and it will pick up span '03' for processing. Below
shows the code example for using span spec:

```python
from  tfx.proto import example_gen_pb2

input = example_gen_pb2.Input(splits=[
                example_gen_pb2.Input.Split(name='train',
                                            pattern='span-{SPAN}/train/*'),
                example_gen_pb2.Input.Split(name='eval',
                                            pattern='span-{SPAN}/eval/*')
            ])
examples = csv_input('/tmp')
example_gen = CsvExampleGen(input=examples, input_config=input)
```

Note: Retrieving a certain span is not supported yet. You can only fix the
pattern for now (for example, 'span-2/eval/*' instead of 'span-{SPAN}/eval/*'),
but by doing this, span number stored in metadata will be zero.

### Version

Note: Version is not supported yet

## Custom ExampleGen

Note: this feature is only available after TFX 0.14.

If the currently available ExampleGen components don't fit your needs, create
a custom ExampleGen, which will include a new executor extended from BaseExampleGenExecutor.

### File-Based ExampleGen

First, extend BaseExampleGenExecutor with a custom Beam PTransform, which
provides the conversion from your train/eval input split to TF examples. For
example, the
[CsvExampleGen executor](https://github.com/tensorflow/tfx/blob/master/tfx/components/example_gen/csv_example_gen/executor.py)
provides the conversion from an input CSV split to TF examples.

Then, create a component with above executor, as done in [CsvExampleGen component](https://github.com/tensorflow/tfx/blob/master/tfx/components/example_gen/csv_example_gen/component.py).
Alternatively, pass a custom executor into the standard
ExampleGen component as shown below.

```python
from tfx.components.base import executor_spec
from tfx.components.example_gen.component import FileBasedExampleGen
from tfx.components.example_gen.csv_example_gen import executor
from tfx.utils.dsl_utils import external_input

examples = external_input(os.path.join(base_dir, 'data/simple'))
example_gen = FileBasedExampleGen(
    input=examples,
    custom_executor_spec=executor_spec.ExecutorClassSpec(executor.Executor))
```

Now, we also support reading Avro and Parquet files using this
[method](https://github.com/tensorflow/tfx/blob/master/tfx/components/example_gen/custom_executors/avro_component_test.py).

### Query-Based ExampleGen

First, extend BaseExampleGenExecutor with a custom Beam PTransform, which reads
from the external data source. Then, create a simple component by
extending QueryBasedExampleGen.

This may or may not require additional connection configurations. For example,
the
[BigQuery executor](https://github.com/tensorflow/tfx/blob/master/tfx/components/example_gen/big_query_example_gen/executor.py)
reads using a default beam.io connector, which abstracts the connection
configuration details. The
[Presto executor](https://github.com/tensorflow/tfx/blob/master/tfx/examples/custom_components/presto_example_gen/presto_component/executor.py),
requires a custom Beam PTransform and a
[custom connection configuration protobuf](https://github.com/tensorflow/tfx/blob/master/tfx/examples/custom_components/presto_example_gen/proto/presto_config.proto)
as input.

If a connection configuration is required for a custom ExampleGen component, create
a new protobuf and pass it in through custom_config, which is now an optional
execution parameter. Below is an example of how to use a configured component.

```python
from tfx.examples.custom_components.presto_example_gen.proto import presto_config_pb2
from tfx.examples.custom_components.presto_example_gen.presto_component.component import PrestoExampleGen

presto_config = presto_config_pb2.PrestoConnConfig(host='localhost', port=8080)
example_gen = PrestoExampleGen(presto_config, query='SELECT * FROM chicago_taxi_trips')
```
