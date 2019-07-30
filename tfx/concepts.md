# Core Concepts

This document describes core concepts defined and implmented in the TFX system.

## Pipeline

TODO: To be added

## Component

TODO: To be added

## Orchestrator

TODO: To be added

## Runner

TODO: To be added

## Artifacts

In a Pipeline, an **Artifact**[^1] represents actual data that flow through
between Components. All Components take Artifacts as inputs and outputs.
Generally, Components must have at least one input Artifact and one output
Artifact. Artifact must be strongly typed, in a way as is registered in
[ML Metadata](https://www.tensorflow.org/tfx/guide/mlmd). In other words,
all Artifacts must have associated **Metadata**, which defines **Type** and
**Properties**. The concepts of **Artifact** and **Artifact Type** originate
from the [ML Metadata](https://github.com/google/ml-metadata) project,
as detailed in [this document](https://github.com/google/ml-metadata/blob/master/g3doc/get_started.md#concepts).
TFX implements a particular use of Artifact and Metadata as a key to realize its
high-level functionalities.

The **Artifact Type** is determined by how the Artifact is used by Components
in the Pipeline,  but not necessarily by what the Artifact physically is on
a filesystem.

For instance, the *ExamplesPath* artifact type may represent Examples
materialized in TFRecord of `tensorflow::Example` protocol buffer, CSV, JSON,
or any other physical format. Nevertheless, the way how Examples are used in
a Pipeline is exactly the same; being analyzed to generate statistics, being
validated against expected schema, being pre-processed in advance to training,
and being fed into training models. Likewise, the *ModelExportPath* artifact
type may representstrained model objects exported in various physical formats
such as TensorFlow SavedModel, ONNX, PMML or PKL (of various types of model
objects in Python). Nevertheless, models are always to be evaluated, analyzed
and deployed for serving in Pipelines.

As of TFX 0.13.0, *ExamplesPath* is assumed to be `tensorflow::Example`
protocol buffer in TFRecord format. *ModelExportResult* is assumed to be
TensorFlow SavedModel. Future versions of TFX may expand those artifact types
with more variants.

In order to differentiate such different variants of the same **Artifact Type**,
the Metadata defines a set of **Artifact Properties**. For instance, one such
**Artifact Property** for an *ExamplesPath* artifact may be *format*, whose
values may be one of `TFRecord`, `json`, `csv`, etc. No matter what the value of
the format property is, it is legitimate to pass an artifact of type
*ExamplesPath* to a Component that is designed to take Examples as an input
Artifact (e.g. a Trainer) and expect it to work. However, the actual
implementation of the consuming Component may adjust its behavior in response to
a particular value of the *format* property, or simply raise runtime error if
it doesn’t have implementation to process the particular format of the
Artifact Type.

In summary, **Artifact Type**s define the ontology of **Artifact**s in the
entire Pipeline system, whereas **Artifact Properties** define the ontology
specific to an **Artifact Type**. Users of the Pipeline system can choose to
extend such ontology locally to their Pipeline applications (by defining and
populating new custom properties). Or, users can choose to extend the ontology
globally for the Pipeline system (by introducing new Artifact Types and
modifying predefined type-properties of tfx Artifact Types), in which case such
extension must be contributed back to the master repository of the Pipeline
system (the TFX repository).
