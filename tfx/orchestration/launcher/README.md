Launcher is a layer in the TFX component design that is responsible for
communicating between the Executor and the underlying TFX metadata layer.

TFX is migrating to use the tech stack built on top of
[IR](https://github.com/tensorflow/tfx/blob/master/tfx/proto/orchestration/pipeline.proto).
This is a major step towards supporting the more flexible TFX DSL
[semantic](https://github.com/tensorflow/community/blob/master/rfcs/20200601-tfx-udsl-semantics.md).
Please refer to the
[RFC](https://github.com/tensorflow/community/blob/master/rfcs/20200705-tfx-ir.md)
for the IR to learn more details.

While this migration is no-op to most of the users, users who needed to
customize the old style Launcher under this package can follow these
instructions to migrate to move to the new IR stack.

#   Migrate from custom Launcher to custom ExecutorOperator
## Background:
`ExecutorOperator` is a new abstraction in the IR based execution stack.
In this stack, developers are not expected to define a subclass of
`BaseComponentLauncher`, but instead to define a subclass of
`BaseExecutorOperator`, and the SDK compiler will inject it into the
[Launcher](https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/portable/launcher.py#L91).

[This commit](https://github.com/tensorflow/tfx/search?q=343405676&type=commits)
includes an example change.
## Steps for migration

1. Change the custom launcher implementation to custom executor operator implementation.

  a. Delete the can_launch method. The new Launcher guarantees that given
     an executor spec, it uses the corresponding executor operator,
     so the verification in `can_launch` is not needed anymore: [link](https://github.com/tensorflow/tfx/blob/r0.27.0/tfx/orchestration/portable/launcher.py#L156-L157)

  b. Define the constructor following the exact argument list from the base
     class. Here is an example implementation:

  ```python
  def __init__(self,
      executor_spec: message.Message,
      platform_config: Optional[message.Message] = None):
      super().__init__(executor_spec, platform_config)
      self._container_executor_spec = cast(
          executable_spec_pb2.ContainerExecutableSpec, self._executor_spec)
  ```

  c. Change the `_run_executor` method

  ```python
  def _run_executor(
      self, execution_id: int,
      input_dict: Dict[Text, List[types.Artifact]],
      output_dict: Dict[Text, List[types.Artifact]],
      exec_properties: Dict[Text, Any]) -> None:
  ```

  to the new run_executor method:

  ```python
  def run_executor(
    self, execution_info: data_types.ExecutionInfo
  ) -> execution_result_pb2.ExecutorOutput:
  ```
  please note that the new `execution_info argument` is a superset of the
  argument list of the old `_run_executor` method.

  d. Update unit test accordingly, if any.

2. Make the generic launcher be aware of this new Executor operator.

  a. All first-party `ExecutorOperator`s developed by TFX team should already be
       in the [DEFAULT_EXECUTOR_OPERATORS](https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/portable/launcher.py#L58-L61)
       dictionary in the Launcher module.

  b. For custom ExecutorOperator not in above dict, please inject it into the
       launcher constructor via the
       [custom_executor_operators](https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/portable/launcher.py#L107-L108)
       argument.

## Appendix

This section is for advanced developers who deeply customized the legacy
Launcher only, You can stop reading if you only customize the `_run_executor()`
method:

Where all the old functionality of the legacy Launcher went:

* The `create()` method:
    There is only one Launcher class going forward, so no polymorphism support
    is needed. You don’t need to maintain this logic anymore. But you need to
    make sure the ExecutorOperator’s constructor uses the exact function
    signature of its base class.

* The `can_launch()` method:
    This method is not needed any more for the reason mentioned above.

* The `_run_executor()` method:
    This is abstracted as the ExecutorOperator layer mentioned above.

* The `_run_driver()` method:
    Most of the driver logic is implemented generically in the new generic
    Launcher and is hidden from developers. Developers only need to customize
    this layer when they want to define and customize the output of a
    component. This is abstracted as the Driver/DriveOperator layer, a layer
    that is very similar to the Executor/ExecutorOperator. If you override
    this method, you can follow the migration instruction above to migrate it
    to the new stack, with the following difference:
    * Drivers should derive from the [BaseDriver](https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/portable/base_driver.py)
    * DriverOperators should derive from the [BaseDriverOperator](https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/portable/base_driver_operator.py#L27)
    * Custom DriverOperators should be registered to the
      [DEFUALT_DRIVER_OPERATOR](https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/portable/launcher.py#L58)
      map or injected into the launcher via the
      [custom_driver_operators](https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/portable/launcher.py#L109-L110)
      argument.

* The `_run_publisher()` method:
    This is implemented generically in the new generic Launcher and is hidden from developers.
    Note that post-executor customizations can be added to ExecutorOperator.
    The generic Publisher will commit those change.

* The `launch()` method
    This is implemented in the new generic Launcher and is hidden from the developers.
