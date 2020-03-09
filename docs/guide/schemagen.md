# The SchemaGen TFX Pipeline Component

Some TFX components use a description of your input data called a *schema*. The
schema is an instance of
[schema.proto](
https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto).
It can specify data types for feature values,
whether a feature has to be present in all examples, allowed value ranges, and
other properties.  A SchemaGen pipeline component will automatically generate a
schema by inferring types, categories, and ranges from the training data.

*   Consumes: statistics from a StatisticsGen component
*   Emits: Data schema proto

Here's an excerpt from a schema proto:

```proto
...
feature {
  name: "age"
  value_count {
    min: 1
    max: 1
  }
  type: FLOAT
  presence {
    min_fraction: 1
    min_count: 1
  }
}
feature {
  name: "capital-gain"
  value_count {
    min: 1
    max: 1
  }
  type: FLOAT
  presence {
    min_fraction: 1
    min_count: 1
  }
}
...
```

The following TFX libraries use the schema:

*   TensorFlow Data Validation
*   TensorFlow Transform
*   TensorFlow Model Analysis

In a typical TFX pipeline SchemaGen generates a schema, which is consumed by the
other pipeline components.

Note: The auto-generated schema is best-effort and only tries to infer basic
properties of the data. It is expected that developers review and modify it as
needed.

## SchemaGen and TensorFlow Data Validation

SchemaGen makes extensive use of [TensorFlow Data Validation](tfdv.md) for inferring a schema.

## Using the SchemaGen Component

A SchemaGen pipeline component is typically very easy to deploy and requires little
customization. Typical code looks like this:

```python
from tfx import components

...

infer_schema = components.SchemaGen(
    statistics=compute_training_stats.outputs['statistics'])
```
