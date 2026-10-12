"""Foreground operation service; terminal and MCP are separate adapters."""

from optimizer.operations.contracts import ObjectRef, OperationRequest, OperationResponse
from optimizer.operations.service import OperationService
from optimizer.session import Session
from optimizer.catalog import exports

__all__ = exports("operations")
