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
on the DAG. The following example shows a simple toy orchestrator, which runs
the components one by one in DAG's topological order.

```python
import datetime

from tfx.orchestration import component_launcher
from tfx.orchestration import data_types
from tfx.orchestration import tfx_runner

class DirectDagRunner(tfx_runner.TfxRunner):
  """Tfx direct DAG runner."""

  def run(self, pipeline):
    """Directly run components in topological order."""
    # Run id is needed for each run.
    pipeline.pipeline_info.run_id = datetime.datetime.now().isoformat()

    # pipeline.components are in topological order already.
    for component in pipeline.components:
      component_launcher.ComponentLauncher(
          component=component,
          pipeline_info=pipeline.pipeline_info,
          driver_args=data_types.DriverArgs(
              enable_cache=pipeline.enable_cache),
          metadata_connection_config=pipeline.metadata_connection_config,
          additional_pipeline_args=pipeline.additional_pipeline_args
      ).launch()
```

The above orchestrator can be used in the Python DSL:

```python
import direct_runner
from tfx.orchestration import pipeline

def _create_pipeline(...) -> pipeline.Pipeline:
  ...
  return pipeline.Pipeline(...)

if __name__ == '__main__':
  direct_runner.DirectDagRunner().run(_create_pipeline(...))
```

To run above Python DSL file (assuming it is named dsl.py), simply do the
following:

```bash
python dsl.py
```
