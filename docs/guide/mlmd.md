# ML Metadata

[ML Metadata (MLMD)](https://github.com/google/ml-metadata) is a library for
recording and retrieving metadata associated with ML developer and data
scientist workflows. MLMD is an integral part of
[TensorFlow Extended (TFX)](https://www.tensorflow.org/tfx), but is designed so
that it can be used independently. As part of the broader TFX platform, most
users only interact with MLMD when examining the results of pipeline components,
for example in notebooks or in TensorBoard.

The graph below shows the components that are part of MLMD. The storage backend
is pluggable and can be extended. MLMD provides reference implementations for
SQLite (which supports in-memory and disk) and MySQL out of the box. The
MetadataStore provides APIs to record and retrieve metadata to and from the
storage backend. MLMD can register:

-   metadata about the artifacts generated through the components/steps of the
    pipelines
-   metadata about the executions of these components/steps
-   metadata about the pipeline and associated lineage information

The concepts are explained in more detail below.

![ML Metadata Overview](images/mlmd_overview.png)

## Functionality Enabled by MLMD

Tracking the inputs and outputs of all components/steps in an ML workflow and
their lineage allows ML platforms to enable several important features. The
following list provides a non-exhaustive overview of some of the major benefits.

*   **List all Artifacts of a specific type.** Example: all Models that have
    been trained.
*   **Load two Artifacts of the same type for comparison.** Example: compare
    results from two experiments.
*   **Show a DAG of all related executions and their input and output artifacts
    of a context.** Example: visualize the workflow of an experiment for
    debugging and discovery.
*   **Recurse back through all events to see how an artifact was created.**
    Examples: see what data went into a model; enforce data retention plans.
*   **Identify all artifacts that were created using a given artifact.**
    Examples: see all Models trained from a specific dataset; mark models based
    upon bad data.
*   **Determine if an execution has been run on the same inputs before.**
    Example: determine whether a component/step has already completed the same
    work and the previous output can just be reused.
*   **Record and query context of workflow runs.** Examples: track the owner and
    changelist used for a workflow run; group the lineage by experiments; manage
    artifacts by projects.

## Metadata Storage Backends and Store Connection Configuration

The MetadataStore object receives a connection configuration that corresponds to
the storage backend used.

*   **Fake Database** provides an in-memory DB (using SQLite) for fast
    experimentation and local runs. Database is deleted when store object is
    destroyed.

```python
connection_config = metadata_store_pb2.ConnectionConfig()
connection_config.fake_database.SetInParent() # Sets an empty fake database proto.
store = metadata_store.MetadataStore(connection_config)
```

*   **SQLite** reads and writes files from disk.

```python
connection_config = metadata_store_pb2.ConnectionConfig()
connection_config.sqlite.filename_uri = '...'
connection_config.sqlite.connection_mode = 3 # READWRITE_OPENCREATE
store = metadata_store.MetadataStore(connection_config)
```

*   **MySQL** connects to a MySQL server.

```python
connection_config = metadata_store_pb2.ConnectionConfig()
connection_config.mysql.host = '...'
connection_config.mysql.port = '...'
connection_config.mysql.database = '...'
connection_config.mysql.user = '...'
connection_config.mysql.password = '...'
store = metadata_store.MetadataStore(connection_config)
```

## Metadata Store

### Concepts

The Metadata Store uses the following data model to record and retrieve metadata
from the storage backend.

*   `ArtifactType` describes an artifact's type and its properties that are
    stored in the Metadata Store. These types can be registered on-the-fly with
    the Metadata Store in code, or they can be loaded in the store from a
    serialized format. Once a type is registered, its definition is available
    throughout the lifetime of the store.
*   `Artifact` describes a specific instances of an `ArtifactType`, and its
    properties that are written to the Metadata Store.
*   `ExecutionType` describes a type of component or step in a workflow, and its
    runtime parameters.
*   `Execution` is a record of a component run or a step in an ML workflow and
    the runtime parameters. An Execution can be thought of as an instance of an
    `ExecutionType`. Every time a developer runs an ML pipeline or step,
    executions are recorded for each step.
*   `Event` is a record of the relationship between an `Artifact` and
    `Executions`. When an `Execution` happens, `Event`s record every Artifact
    that was used by the `Execution`, and every `Artifact` that was produced.
    These records allow for provenance tracking throughout a workflow. By
    looking at all Events MLMD knows what Executions happened, what Artifacts
    were created as a result, and can recurse back from any `Artifact` to all of
    its upstream inputs.
*   `ContextType` describes a type of conceptual group of `Artifacts` and
    `Executions` in a workflow, and its structural properties. For example:
    projects, pipeline runs, experiments, owners.
*   `Context` is an instances of a `ContextType`. It captures the shared
    information within the group. For example: project name, changelist commit
    id, experiment annotations. It has a user-defined unique name within its
    `ContextType`.
*   `Attribution` is a record of the relationship between Artifacts and
    Contexts.
*   `Association` is a record of the relationship between Executions and
    Contexts.

### Tracking ML Workflows with ML Metadata

Below is a graph depicting how the low-level ML Metadata APIs can be used to
track the execution of a training task, followed by code examples. Note that the
code in this section shows the ML Metadata APIs to be used by ML platform
developers to integrate their platform with ML Metadata, and not directly by
developers. In addition, we will provide higher-level Python APIs that can be
used by data scientists in notebook environments to record their experiment
metadata.

![ML Metadata Example Flow](images/mlmd_flow.png)

1) Before executions can be recorded, ArtifactTypes have to be registered.

```python
# Create ArtifactTypes, e.g., Data and Model
data_type = metadata_store_pb2.ArtifactType()
data_type.name = "DataSet"
data_type.properties["day"] = metadata_store_pb2.INT
data_type.properties["split"] = metadata_store_pb2.STRING
data_type_id = store.put_artifact_type(data_type)

model_type = metadata_store_pb2.ArtifactType()
model_type.name = "SavedModel"
model_type.properties["version"] = metadata_store_pb2.INT
model_type.properties["name"] = metadata_store_pb2.STRING
model_type_id = store.put_artifact_type(model_type)
```

2) Before executions can be recorded, ExecutionTypes have to be registered for
all steps in our ML workflow.

```python
# Create ExecutionType, e.g., Trainer
trainer_type = metadata_store_pb2.ExecutionType()
trainer_type.name = "Trainer"
trainer_type.properties["state"] = metadata_store_pb2.STRING
trainer_type_id = store.put_execution_type(trainer_type)
```

3) Once types are registered, we create a DataSet Artifact.

```python
# Declare input artifact of type DataSet
data_artifact = metadata_store_pb2.Artifact()
data_artifact.uri = 'path/to/data'
data_artifact.properties["day"].int_value = 1
data_artifact.properties["split"].string_value = 'train'
data_artifact.type_id = data_type_id
data_artifact_id = store.put_artifacts([data_artifact])
```

4) With the DataSet Artifact created, we can create the Execution for a Trainer
run

```python
# Register the Execution of a Trainer run
trainer_run = metadata_store_pb2.Execution()
trainer_run.type_id = trainer_type_id
trainer_run.properties["state"].string_value = "RUNNING"
run_id = store.put_executions([trainer_run])
```

5) Declare input event and read data.

```python
# Declare the input event
input_event = metadata_store_pb2.Event()
input_event.artifact_id = data_artifact_id
input_event.execution_id = run_id
input_event.type = metadata_store_pb2.Event.DECLARED_INPUT

# Submit input event to the Metadata Store
store.put_events([input_event])
```

6) Now that the input is read, we declare the output artifact.

```python
# Declare output artifact of type SavedModel
model_artifact = metadata_store_pb2.Artifact()
model_artifact.uri = 'path/to/model/file'
model_artifact.properties["version"].int_value = 1
model_artifact.properties["name"].string_value = 'MNIST-v1'
model_artifact.type_id = model_type_id
model_artifact_id = store.put_artifacts(model_artifact)
```

7) With the Model Artifact created, we can record the output event.

```python
# Declare the output event
output_event = metadata_store_pb2.Event()
output_event.artifact_id = model_artifact_id
output_event.execution_id = run_id
output_event.type = metadata_store_pb2.Event.DECLARED_OUTPUT

# Submit output event to the Metadata Store
store.put_events([output_event])
```

8) Now that everything is recorded, the Execution can be marked as completed.

```python
trainer_run.id = run_id
trainer_run.id.properties["state"].string_value = "COMPLETED"
store.put_executions([trainer_run])
```

9) Then the artifacts and executions can be grouped to a Context (e.g.,
experiment).

```python
# Similarly, create a ContextType, e.g., Experiment with a `note` property
experiment_type = metadata_store_pb2.ContextType()
experiment_type.name = "Experiment"
experiment_type.properties["note"] = metadata_store_pb2.STRING
experiment_type_id = store.put_context_type(experiment_type)

# Group the model and the trainer run to an experiment.
my_experiment = metadata_store_pb2.Context()
my_experiment.type_id = experiment_type_id
# Give the experiment a name
my_experiment.name = "exp1"
my_experiment.properties["note"].string_value = "My first experiment."
experiment_id = store.put_contexts([my_experiment])

attribution = metadata_store_pb2.Attribution()
attribution.artifact_id = model_artifact_id
attribution.context_id = experiment_id

association = metadata_store_pb2.Association()
association.execution_id = run_id
attribution.context_id = experiment_id

store.put_attributions_and_associations([attribution], [association])
```

### With remote grpc server

1) Start a server with

```bash
bazel run -c opt --define grpc_no_ares=true  //ml_metadata/metadata_store:metadata_store_server
```

2) Create the client stub and use it in python

```python
from grpc import insecure_channel
from ml_metadata.proto import metadata_store_pb2
from ml_metadata.proto import metadata_store_service_pb2
from ml_metadata.proto import metadata_store_service_pb2_grpc
channel = insecure_channel('localhost:8080')
stub = metadata_store_service_pb2_grpc.MetadataStoreServiceStub(channel)
```

3) Use MLMD with RPC calls

```python
# Create ArtifactTypes, e.g., Data and Model
data_type = metadata_store_pb2.ArtifactType()
data_type.name = "DataSet"
data_type.properties["day"] = metadata_store_pb2.INT
data_type.properties["split"] = metadata_store_pb2.STRING
request = metadata_store_service_pb2.PutArtifactTypeRequest()
request.all_fields_match = True
request.artifact_type.CopyFrom(data_type)
stub.PutArtifactType(request)
model_type = metadata_store_pb2.ArtifactType()
model_type.name = "SavedModel"
model_type.properties["version"] = metadata_store_pb2.INT
model_type.properties["name"] = metadata_store_pb2.STRING
request.artifact_type.CopyFrom(model_type)
stub.PutArtifactType(request)
```

