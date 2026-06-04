"""
API endpoints for workflow monitoring and management.
"""

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.workflow_monitor import (
    WorkflowExecutionMetrics,
    WorkflowNodeMetrics,
    workflow_monitor
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/workflows", tags=["workflows"])


@router.get("/executions/{execution_id}")
def get_workflow_execution(
    execution_id: str,
    db: Session = Depends(get_db),
) -> Dict:
    """Get details for a specific workflow execution."""
    execution = workflow_monitor.get_workflow_execution(execution_id)
    if not execution:
        raise HTTPException(status_code=404, detail="Workflow execution not found")

    return {
        "execution_id": execution.execution_id,
        "workflow_name": execution.workflow_name,
        "start_time": execution.start_time.isoformat() if execution.start_time else None,
        "end_time": execution.end_time.isoformat() if execution.end_time else None,
        "duration_seconds": execution.duration_seconds,
        "status": execution.status,
        "error_message": execution.error_message,
        "node_timings": execution.node_timings,
        "metadata": execution.metadata
    }


@router.get("/executions")
def list_workflow_executions(
    workflow_name: Optional[str] = Query(None, description="Filter by workflow name"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of executions to return"),
    db: Session = Depends(get_db),
) -> List[Dict]:
    """List recent workflow executions."""
    executions = workflow_monitor.get_recent_executions(workflow_name=workflow_name, limit=limit)

    return [
        {
            "execution_id": execution.execution_id,
            "workflow_name": execution.workflow_name,
            "start_time": execution.start_time.isoformat() if execution.start_time else None,
            "end_time": execution.end_time.isoformat() if execution.end_time else None,
            "duration_seconds": execution.duration_seconds,
            "status": execution.status,
            "error_message": execution.error_message
        }
        for execution in executions
    ]


@router.get("/nodes")
def get_workflow_node_metrics(
    workflow_name: Optional[str] = Query(None, description="Filter by workflow name"),
    db: Session = Depends(get_db),
) -> Dict[str, Dict]:
    """Get workflow node execution metrics."""
    node_metrics = workflow_monitor.get_node_metrics(workflow_name=workflow_name)

    return {
        node_key: {
            "node_name": metrics.node_name,
            "execution_count": metrics.execution_count,
            "total_duration": metrics.total_duration,
            "average_duration": metrics.average_duration,
            "min_duration": metrics.min_duration if metrics.min_duration != float('inf') else 0,
            "max_duration": metrics.max_duration,
            "failure_count": metrics.failure_count,
            "success_rate": (metrics.execution_count - metrics.failure_count) / metrics.execution_count
            if metrics.execution_count > 0 else 0
        }
        for node_key, metrics in node_metrics.items()
    }


@router.get("/summary/{workflow_name}")
def get_workflow_summary(
    workflow_name: str,
    db: Session = Depends(get_db),
) -> Dict:
    """Get summary statistics for a specific workflow."""
    return workflow_monitor.get_workflow_summary(workflow_name)


@router.post("/reset")
def reset_workflow_metrics(
    db: Session = Depends(get_db),
) -> Dict:
    """Reset all workflow monitoring metrics (admin only in production)."""
    workflow_monitor.reset_metrics()
    return {"message": "Workflow metrics reset successfully"}


@router.get("/health")
def workflow_health_check() -> Dict:
    """Health check for workflow monitoring service."""
    return {
        "status": "healthy",
        "service": "workflow_monitor",
        "tracked_executions": len(workflow_monitor._executions),
        "tracked_nodes": len(workflow_monitor._node_metrics)
    }