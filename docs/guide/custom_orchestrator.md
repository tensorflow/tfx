# Orchestrating TFX Pipelines

## Custom Orchestrator

TFX is designed to be portable to multiple environments and orchestration
frameworks. Developers can create custom orchestrators or add additional
orchestrators in addition to the default orchestrators that are supported by
TFX, namely [Airflow](airflow.md), [Beam](beam_orchestrator.md) and
[Kubeflow](kubeflow.md).

All orchestrators must inherit from
[TfxRunner](https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/tfx_runner.py).
TFX orchestrators take the logical pipeline object, which contains pipeline
args, components, and DAG, and are responsible for scheduling components of the
TFX pipeline based on the dependencies defined by the DAG.

For example, let's look at how to create a custom orchestrator with
[ComponentLauncher](https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/component_launcher.py).
ComponentLauncher already handles driver, executor, and publisher of a single
component. The new orchestrator just needs to schedule ComponentLaunchers based
on the DAG. A simple orchestrator is provided as the [LocalDagRunner]
(https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/local/local_dag_runner.py),
which runs the components one by one in DAG's topological order.

This orchestrator can be used in the Python DSL:

```python
def _create_pipeline(...) -> dsl.Pipeline:
  ...
  return dsl.Pipeline(...)

if __name__ == '__main__':
  orchestration.LocalDagRunner().run(_create_pipeline(...))
```

To run above Python DSL file (assuming it is named dsl.py), simply do the
following:

```bash
python dsl.py
```
