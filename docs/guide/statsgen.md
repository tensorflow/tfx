# The StatisticsGen TFX Pipeline Component

The StatisticsGen TFX pipeline component generates features statistics
over both training and serving data, which can be used by other pipeline
components.
StatisticsGen uses Beam to scale to large datasets.

* Consumes: datasets created by an ExampleGen pipeline component.
* Emits: Dataset statistics.

## StatisticsGen and TensorFlow Data Validation

StatisticsGen makes extensive use of [TensorFlow Data Validation](tfdv.md) for
generating statistics from your dataset.

## Using the StatsGen Component

A StatisticsGen pipeline component is typically very easy to deploy and requires
little customization. Typical code looks like this:

```python
compute_eval_stats = StatisticsGen(
      examples=example_gen.outputs['examples'],
      name='compute-eval-stats'
      )
```

## Using the StatsGen Component With a Schema

For the first run of a pipeline, the output of StatisticsGen will be used to
infer a schema. However, on subsequent runs you may have a manually curated
schema that contains additional information about your data set. By providing
this schema to StatisticsGen, TFDV can provide more useful statistics based on
declared properties of your data set.

In this setting, you will invoke StatisticsGen with a curated schema that has
been imported by an ImporterNode like this:

```python
user_schema_importer = Importer(
    source_uri=user_schema_dir, # directory containing only schema text proto
    artifact_type=standard_artifacts.Schema).with_id('schema_importer')

compute_eval_stats = StatisticsGen(
      examples=example_gen.outputs['examples'],
      schema=user_schema_importer.outputs['result'],
      name='compute-eval-stats'
      )
```

### Creating a Curated Schema

`Schema` in TFX is an instance of the TensorFlow Metadata
[`Schema` proto](https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto).
This can be composed in
[text format](https://googleapis.dev/python/protobuf/latest/google/protobuf/text_format.html)
from scratch. However, it is easier to use the inferred schema produced by
`SchemaGen` as a starting point. Once the `SchemaGen` component has executed,
the schema will be located under the pipeline root in the following path:

    <pipeline_root>/SchemaGen/schema/<artifact_id>/schema.pbtxt

Where `<artifact_id>` represents a unique ID for this version of the schema in
MLMD. This schema proto can then be modified to communicate information about
the dataset which cannot be reliably inferred, which will make the output of
`StatisticsGen` more useful and the validation performed in the
[`ExampleValidator`](https://www.tensorflow.org/tfx/guide/exampleval) component
more stringent.

More details are available in the
[StatisticsGen API reference](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/StatisticsGen).
