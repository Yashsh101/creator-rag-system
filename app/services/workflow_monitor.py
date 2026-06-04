"""
Workflow monitoring service for tracking LangGraph execution performance and metrics.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class WorkflowExecutionMetrics:
    """Metrics for a single workflow execution."""
    workflow_name: str
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    status: str = "running"  # running, completed, failed
    error_message: Optional[str] = None
    node_timings: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class WorkflowNodeMetrics:
    """Metrics for a single node execution within a workflow."""
    node_name: str
    execution_count: int = 0
    total_duration: float = 0.0
    average_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    failure_count: int = 0


class WorkflowMonitor:
    """Monitor for tracking workflow execution metrics."""

    def __init__(self):
        self._executions: Dict[str, WorkflowExecutionMetrics] = {}
        self._node_metrics: Dict[str, WorkflowNodeMetrics] = defaultdict(WorkflowNodeMetrics)
        self._lock = Lock()
        self._execution_counter = 0

    def start_workflow_execution(
        self,
        workflow_name: str,
        execution_id: Optional[str] = None,
        metadata: Optional[Dict[str, any]] = None
    ) -> str:
        """Start tracking a workflow execution."""
        if execution_id is None:
            self._execution_counter += 1
            execution_id = f"{workflow_name}_{self._execution_counter}_{int(time.time())}"

        with self._lock:
            metrics = WorkflowExecutionMetrics(
                workflow_name=workflow_name,
                execution_id=execution_id,
                start_time=datetime.now(),
                metadata=metadata or {}
            )
            self._executions[execution_id] = metrics

        logger.info(
            "workflow_execution_started",
            extra={
                "event": "workflow_execution_started",
                "workflow_name": workflow_name,
                "execution_id": execution_id
            }
        )

        return execution_id

    def end_workflow_execution(
        self,
        execution_id: str,
        status: str = "completed",
        error_message: Optional[str] = None
    ) -> Optional[WorkflowExecutionMetrics]:
        """End tracking a workflow execution."""
        with self._lock:
            if execution_id not in self._executions:
                logger.warning(
                    "workflow_execution_not_found",
                    extra={
                        "event": "workflow_execution_not_found",
                        "execution_id": execution_id
                    }
                )
                return None

            metrics = self._executions[execution_id]
            metrics.end_time = datetime.now()
            metrics.status = status
            metrics.error_message = error_message

            if metrics.start_time and metrics.end_time:
                metrics.duration_seconds = (metrics.end_time - metrics.start_time).total_seconds()

            # Move to completed executions (optional: keep in memory or archive)
            logger.info(
                "workflow_execution_ended",
                extra={
                    "event": "workflow_execution_ended",
                    "execution_id": execution_id,
                    "workflow_name": metrics.workflow_name,
                    "status": status,
                    "duration_seconds": metrics.duration_seconds
                }
            )

            return metrics

    def record_node_execution(
        self,
        workflow_name: str,
        node_name: str,
        duration_seconds: float,
        success: bool = True
    ):
        """Record metrics for a node execution."""
        with self._lock:
            node_key = f"{workflow_name}:{node_name}"
            node_metrics = self._node_metrics[node_key]

            node_metrics.execution_count += 1
            node_metrics.total_duration += duration_seconds
            node_metrics.average_duration = node_metrics.total_duration / node_metrics.execution_count
            node_metrics.min_duration = min(node_metrics.min_duration, duration_seconds)
            node_metrics.max_duration = max(node_metrics.max_duration, duration_seconds)

            if not success:
                node_metrics.failure_count += 1

            logger.debug(
                "workflow_node_executed",
                extra={
                    "event": "workflow_node_executed",
                    "workflow_name": workflow_name,
                    "node_name": node_name,
                    "duration_seconds": duration_seconds,
                    "success": success
                }
            )

    def get_workflow_execution(self, execution_id: str) -> Optional[WorkflowExecutionMetrics]:
        """Get metrics for a specific workflow execution."""
        with self._lock:
            return self._executions.get(execution_id)

    def get_recent_executions(
        self,
        workflow_name: Optional[str] = None,
        limit: int = 100
    ) -> List[WorkflowExecutionMetrics]:
        """Get recent workflow executions."""
        with self._lock:
            executions = list(self._executions.values())

            if workflow_name:
                executions = [e for e in executions if e.workflow_name == workflow_name]

            # Sort by start time descending (most recent first)
            executions.sort(key=lambda e: e.start_time, reverse=True)

            return executions[:limit]

    def get_node_metrics(self, workflow_name: Optional[str] = None) -> Dict[str, WorkflowNodeMetrics]:
        """Get node execution metrics."""
        with self._lock:
            if workflow_name:
                return {
                    k: v for k, v in self._node_metrics.items()
                    if k.startswith(f"{workflow_name}:")
                }
            return dict(self._node_metrics)

    def get_workflow_summary(self, workflow_name: str) -> Dict[str, any]:
        """Get summary statistics for a workflow."""
        with self._lock:
            executions = [
                e for e in self._execceptions.values()
                if e.workflow_name == workflow_name
            ]

            if not executions:
                return {"workflow_name": workflow_name, "total_executions": 0}

            completed_executions = [e for e in executions if e.status == "completed"]
            failed_executions = [e for e in executions if e.status == "failed"]

            avg_duration = None
            if completed_executions:
                durations = [e.duration_seconds for e in completed_executions if e.duration_seconds is not None]
                if durations:
                    avg_duration = sum(durations) / len(durations)

            return {
                "workflow_name": workflow_name,
                "total_executions": len(executions),
                "completed_executions": len(completed_executions),
                "failed_executions": len(failed_executions),
                "success_rate": len(completed_executions) / len(executions) if executions else 0,
                "average_duration_seconds": avg_duration,
                "recent_executions": [
                    {
                        "execution_id": e.execution_id,
                        "start_time": e.start_time.isoformat(),
                        "duration_seconds": e.duration_seconds,
                        "status": e.status
                    }
                    for e in sorted(executions, key=lambda x: x.start_time, reverse=True)[:10]
                ]
            }

    def reset_metrics(self):
        """Reset all collected metrics."""
        with self._lock:
            self._executions.clear()
            self._node_metrics.clear()
            self._execution_counter = 0
            logger.info("workflow_metrics_reset", extra={"event": "workflow_metrics_reset"})


# Global workflow monitor instance
workflow_monitor = WorkflowMonitor()


# Decorator for automatic workflow tracking
def track_workflow(workflow_name: str):
    """Decorator to automatically track workflow execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            execution_id = workflow_monitor.start_workflow_execution(workflow_name)
            try:
                result = func(*args, **kwargs)
                workflow_monitor.end_workflow_execution(execution_id, "completed")
                return result
            except Exception as e:
                workflow_monitor.end_workflow_execution(execution_id, "failed", str(e))
                raise
        return wrapper
    return decorator


# Context manager for tracking node execution
class track_node_execution:
    """Context manager for tracking node execution time."""

    def __init__(self, workflow_name: str, node_name: str):
        self.workflow_name = workflow_name
        self.node_name = node_name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            success = exc_type is None
            workflow_monitor.record_node_execution(
                self.workflow_name,
                self.node_name,
                duration,
                success
            )