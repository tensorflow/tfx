"""MLMD representations for TFX Pipeline. References objects from MLMD to inspect their status.
"""

from typing import Dict, List, Optional

from ml_metadata.metadata_store import MetadataStore
from ml_metadata.proto import metadata_store_pb2
from tfx.orchestration.metadata import (
    _CONTEXT_TYPE_PIPELINE,
    _CONTEXT_TYPE_PIPELINE_RUN,
    _EXECUTION_TYPE_KEY_COMPONENT_ID,
    _EXECUTION_TYPE_KEY_PIPELINE_NAME,
    _EXECUTION_TYPE_KEY_RUN_ID,
    _EXECUTION_TYPE_KEY_STATE,
)
from tfx.orchestration.pipeline import Pipeline


class _MetadataClientBase:
    def __init__(self, metadata_connection_config: metadata_store_pb2.ConnectionConfig):
        self._metadata_connection_config = metadata_connection_config
        self._store = None

    @property
    def _mlmd(self):
        if not self._store:
            self._store = MetadataStore(self._metadata_connection_config)
        return self._store

    @property
    def connection_config(self):
        return self._metadata_connection_config


class PipelineRunContext(_MetadataClientBase):
    """Context for a specific synchronous pipeline run.

    Unique pipeline_name + run_id.
    """

    def __init__(
        self,
        pipeline_run_context: metadata_store_pb2.Context,
        metadata_connection_config: metadata_store_pb2.ConnectionConfig,
    ) -> None:
        super().__init__(metadata_connection_config)
        self._context = pipeline_run_context

    def get_status(self) -> Dict[str, str]:
        """Retrieve status of component executions for current pipeline run.

        Status can be one of the following:
        - 'new': Component execution has started.
        - 'completed': Component has executed end to end producing new artifacts
        - 'cached': Component execution skipped to reuse previous results from cache.

        Returns:
            Dict[str, str]: Dictionary with component id and their state.
        """
        executions = self._mlmd.get_executions_by_context(self._context.id)
        return {
            e.properties[_EXECUTION_TYPE_KEY_COMPONENT_ID]
            .string_value: e.properties[_EXECUTION_TYPE_KEY_STATE]
            .string_value
            for e in executions
        }

    @property
    def pipeline_name(self) -> str:
        """Get pipeline name

        Returns:
            str: Pipeline name for current pipeline run
        """
        return self._context.properties[_EXECUTION_TYPE_KEY_PIPELINE_NAME].string_value

    @property
    def run_id(self) -> str:
        """Run identifier for pipeline run

        Returns:
            str: String representing run_id
        """
        return self._context.properties[_EXECUTION_TYPE_KEY_RUN_ID].string_value

    @property
    def pipeline(self) -> "PipelineContext":
        """Retrieve Pipeline context for this pipeline run.

        Returns:
            PipelineContext: Context for pipeline from this pipeline_run
        """
        context = PipelineContext.get_pipeline_context(self._mlmd, self.pipeline_name)
        return PipelineContext(context, self.connection_config)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(pipeline_name="{self.pipeline_name}", run_id="{self.run_id}"")'


class PipelineContext(_MetadataClientBase):
    """Reference to pipeline context in MLMD."""

    def __init__(
        self,
        pipeline_context: metadata_store_pb2.Context,
        metadata_connection_config: metadata_store_pb2.ConnectionConfig,
    ) -> None:
        """Creates instance of PipelineContext

        Args:
            pipeline_context: MLMD Context for a pipeline.
            metadata_connection_config: MLMD connection config
        """
        super().__init__(metadata_connection_config)
        self._context = pipeline_context

    @staticmethod
    def get_pipeline_context(
        mlmd_client: MetadataStore, pipeline_name: str
    ) -> metadata_store_pb2.Context:
        """Static method to get pipeline context.

        Args:
            mlmd_client: MLMD MetadataStore instance
            pipeline_name: Pipeline name to retrieve

        Returns:
            metadata_store_pb2.Context: Returns None if context doesn't exist in MLMD yet. This can happen if pipeline has not been executed yet.
        """
        return mlmd_client.get_context_by_type_and_name(
            _CONTEXT_TYPE_PIPELINE, pipeline_name
        )

    def _get_runs(self) -> List[PipelineRunContext]:
        runs = []
        contexts = self._mlmd.get_contexts_by_type(_CONTEXT_TYPE_PIPELINE_RUN)
        for context in contexts:
            if (
                context.properties[_EXECUTION_TYPE_KEY_PIPELINE_NAME].string_value
                == self.name
            ):
                runs.append(PipelineRunContext(context, self.connection_config))
        return runs

    @property
    def runs(self) -> List[PipelineRunContext]:
        """Retrieve pipeline runs for current pipeline.

        Returns:
            List[PipelineRunContext]: List of pipeline runs for current pipeline
        """
        return self._get_runs()

    @property
    def name(self) -> str:
        return self._context.name

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name="{self.name}")'


class MetadataClient(_MetadataClientBase):
    """MLMD read client."""

    def __init__(
        self,
        metadata_connection_config: metadata_store_pb2.ConnectionConfig = None,
    ):
        """Creates instance of MetadataClient.

        Args:
            metadata_config: Configuration to connect to MLMD. Defaults to staging DB connection.
        """
        super().__init__(metadata_connection_config)

    @staticmethod
    def from_pipeline(pipeline: Pipeline) -> Optional[PipelineContext]:
        """Retrieve PipelineContext given a pipeline.

        Helper method to make this recurring task easier.

        Args:
            pipeline: Pipeline to retrieve context from.

        Returns:
            Optional[PipelineContext]: Returns None if context doesn't exist in MLMD yet. This can happen if pipeline has not been executed yet.
        """
        return MetadataClient(pipeline.metadata_connection_config).get_pipeline(
            pipeline.pipeline_info.pipeline_name
        )

    def get_pipeline(self, pipeline_name: str) -> Optional[PipelineContext]:
        """Retrieve PipelineContext given a pipeline name.

        Args:
            pipeline_name: Name of pipeline to retrieve

        Returns:
            Optional[PipelineContext]: Returns None if context doesn't exist in MLMD yet. This can happen if pipeline has not been executed yet.
        """
        context = PipelineContext.get_pipeline_context(self._mlmd, pipeline_name)
        if not context:
            return None
        return PipelineContext(context, self.connection_config)

    def __repr__(self) -> str:
        connection_str = str(self.connection_config).replace("\n", ",")
        return f"{self.__class__.__name__}(connection_config={{{connection_str}}})"
