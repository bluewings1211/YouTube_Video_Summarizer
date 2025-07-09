"""
Enhanced transaction manager for complex database operations.

This module provides advanced transaction management capabilities
including rollback mechanisms, savepoints, and operation tracking.
"""

import logging
from contextlib import contextmanager
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text

from .exceptions import (
    DatabaseError, DatabaseConnectionError, DatabaseQueryError,
    classify_database_error, is_recoverable_error
)

logger = logging.getLogger(__name__)


class TransactionStatus(Enum):
    """Transaction status enumeration."""
    ACTIVE = "active"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class OperationType(Enum):
    """Operation type enumeration."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    BATCH_DELETE = "batch_delete"
    CACHE_CLEAR = "cache_clear"
    REPROCESS = "reprocess"


@dataclass
class TransactionOperation:
    """Represents a single operation within a transaction."""
    id: str
    operation_type: OperationType
    description: str
    target_table: str
    target_id: Optional[int] = None
    target_ids: Optional[List[int]] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    executed_at: Optional[datetime] = None
    success: bool = False
    error_message: Optional[str] = None
    affected_rows: Optional[int] = None
    rollback_data: Optional[Dict[str, Any]] = None


@dataclass
class SavepointInfo:
    """Information about a database savepoint."""
    name: str
    created_at: datetime
    operations_count: int
    description: Optional[str] = None


@dataclass
class TransactionResult:
    """Result of a transaction operation."""
    success: bool
    transaction_id: str
    status: TransactionStatus
    operations: List[TransactionOperation]
    savepoints: List[SavepointInfo]
    start_time: datetime
    end_time: Optional[datetime] = None
    execution_time_seconds: Optional[float] = None
    error_message: Optional[str] = None
    rollback_reason: Optional[str] = None


class TransactionManager:
    """
    Enhanced transaction manager with rollback capabilities.
    
    Provides advanced transaction management including:
    - Operation tracking and logging
    - Savepoint management
    - Automated rollback on errors
    - Transaction result reporting
    """
    
    def __init__(self, session: Session, auto_commit: bool = True):
        """
        Initialize transaction manager.
        
        Args:
            session: Database session
            auto_commit: Whether to auto-commit successful transactions
        """
        self.session = session
        self.auto_commit = auto_commit
        self.transaction_id = f"txn_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.logger = logging.getLogger(f"{__name__}.TransactionManager")
        
        # Transaction state
        self.status = TransactionStatus.ACTIVE
        self.operations: List[TransactionOperation] = []
        self.savepoints: List[SavepointInfo] = []
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self.rollback_reason: Optional[str] = None
        
        # Operation counter for unique IDs
        self._operation_counter = 0

    def _generate_operation_id(self) -> str:
        """Generate unique operation ID."""
        self._operation_counter += 1
        return f"{self.transaction_id}_op_{self._operation_counter:03d}"

    def _generate_savepoint_name(self) -> str:
        """Generate unique savepoint name."""
        return f"sp_{self.transaction_id}_{len(self.savepoints) + 1}"

    def create_savepoint(self, description: Optional[str] = None) -> str:
        """
        Create a database savepoint.
        
        Args:
            description: Optional description for the savepoint
            
        Returns:
            Savepoint name
            
        Raises:
            DatabaseError: If savepoint creation fails
        """
        try:
            savepoint_name = self._generate_savepoint_name()
            
            # Create savepoint
            self.session.execute(text(f"SAVEPOINT {savepoint_name}"))
            
            # Record savepoint info
            savepoint_info = SavepointInfo(
                name=savepoint_name,
                created_at=datetime.now(),
                operations_count=len(self.operations),
                description=description
            )
            self.savepoints.append(savepoint_info)
            
            self.logger.info(f"Created savepoint {savepoint_name}: {description or 'No description'}")
            return savepoint_name
            
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self.logger.error(f"Failed to create savepoint: {db_error}")
            raise DatabaseError(f"Savepoint creation failed: {db_error.message}")

    def rollback_to_savepoint(self, savepoint_name: str, reason: str = "Manual rollback") -> bool:
        """
        Rollback to a specific savepoint.
        
        Args:
            savepoint_name: Name of the savepoint to rollback to
            reason: Reason for rollback
            
        Returns:
            True if rollback was successful
            
        Raises:
            DatabaseError: If rollback fails
        """
        try:
            # Find the savepoint
            savepoint = None
            for sp in self.savepoints:
                if sp.name == savepoint_name:
                    savepoint = sp
                    break
            
            if not savepoint:
                raise DatabaseError(f"Savepoint {savepoint_name} not found")
            
            # Rollback to savepoint
            self.session.execute(text(f"ROLLBACK TO SAVEPOINT {savepoint_name}"))
            
            # Remove operations after this savepoint
            self.operations = self.operations[:savepoint.operations_count]
            
            # Remove savepoints after this one
            self.savepoints = [sp for sp in self.savepoints if sp.created_at <= savepoint.created_at]
            
            self.logger.info(f"Rolled back to savepoint {savepoint_name}: {reason}")
            return True
            
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self.logger.error(f"Failed to rollback to savepoint: {db_error}")
            raise DatabaseError(f"Savepoint rollback failed: {db_error.message}")

    def execute_operation(
        self,
        operation_type: OperationType,
        description: str,
        target_table: str,
        operation_func: Callable,
        target_id: Optional[int] = None,
        target_ids: Optional[List[int]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        rollback_data: Optional[Dict[str, Any]] = None
    ) -> TransactionOperation:
        """
        Execute an operation within the transaction.
        
        Args:
            operation_type: Type of operation
            description: Description of the operation
            target_table: Target table name
            operation_func: Function to execute the operation
            target_id: Single target ID (for single record operations)
            target_ids: Multiple target IDs (for batch operations)
            parameters: Operation parameters
            rollback_data: Data needed for rollback
            
        Returns:
            TransactionOperation with results
            
        Raises:
            DatabaseError: If operation fails
        """
        operation_id = self._generate_operation_id()
        
        operation = TransactionOperation(
            id=operation_id,
            operation_type=operation_type,
            description=description,
            target_table=target_table,
            target_id=target_id,
            target_ids=target_ids,
            parameters=parameters or {},
            rollback_data=rollback_data
        )
        
        try:
            self.logger.info(f"Executing operation {operation_id}: {description}")
            
            # Execute the operation
            result = operation_func()
            
            # Update operation with results
            operation.executed_at = datetime.now()
            operation.success = True
            
            # Try to extract affected rows count
            if hasattr(result, 'rowcount'):
                operation.affected_rows = result.rowcount
            elif isinstance(result, int):
                operation.affected_rows = result
            
            self.operations.append(operation)
            
            self.logger.info(f"Operation {operation_id} completed successfully")
            return operation
            
        except Exception as e:
            operation.executed_at = datetime.now()
            operation.success = False
            operation.error_message = str(e)
            
            self.operations.append(operation)
            
            self.logger.error(f"Operation {operation_id} failed: {e}")
            raise

    def commit_transaction(self) -> TransactionResult:
        """
        Commit the transaction.
        
        Returns:
            TransactionResult with transaction details
            
        Raises:
            DatabaseError: If commit fails
        """
        try:
            if self.auto_commit:
                self.session.commit()
            
            self.status = TransactionStatus.COMMITTED
            self.end_time = datetime.now()
            
            self.logger.info(f"Transaction {self.transaction_id} committed successfully")
            
            return self._create_transaction_result()
            
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self.logger.error(f"Failed to commit transaction: {db_error}")
            
            # Attempt rollback
            try:
                self.session.rollback()
                self.status = TransactionStatus.ROLLED_BACK
                self.rollback_reason = f"Auto-rollback after commit failure: {db_error.message}"
            except Exception as rollback_error:
                self.logger.error(f"Rollback after commit failure also failed: {rollback_error}")
                self.status = TransactionStatus.FAILED
            
            raise DatabaseError(f"Transaction commit failed: {db_error.message}")

    def rollback_transaction(self, reason: str = "Manual rollback") -> TransactionResult:
        """
        Rollback the entire transaction.
        
        Args:
            reason: Reason for rollback
            
        Returns:
            TransactionResult with transaction details
        """
        try:
            self.session.rollback()
            self.status = TransactionStatus.ROLLED_BACK
            self.rollback_reason = reason
            self.end_time = datetime.now()
            
            self.logger.info(f"Transaction {self.transaction_id} rolled back: {reason}")
            
            return self._create_transaction_result()
            
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self.logger.error(f"Failed to rollback transaction: {db_error}")
            
            self.status = TransactionStatus.FAILED
            self.error_message = f"Rollback failed: {db_error.message}"
            
            return self._create_transaction_result()

    def _create_transaction_result(self) -> TransactionResult:
        """Create transaction result object."""
        execution_time = None
        if self.end_time:
            execution_time = (self.end_time - self.start_time).total_seconds()
        
        return TransactionResult(
            success=self.status == TransactionStatus.COMMITTED,
            transaction_id=self.transaction_id,
            status=self.status,
            operations=self.operations.copy(),
            savepoints=self.savepoints.copy(),
            start_time=self.start_time,
            end_time=self.end_time,
            execution_time_seconds=execution_time,
            error_message=self.error_message,
            rollback_reason=self.rollback_reason
        )

    def get_transaction_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the transaction.
        
        Returns:
            Dictionary with transaction summary
        """
        successful_ops = sum(1 for op in self.operations if op.success)
        failed_ops = len(self.operations) - successful_ops
        
        return {
            "transaction_id": self.transaction_id,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "operations": {
                "total": len(self.operations),
                "successful": successful_ops,
                "failed": failed_ops,
                "by_type": {
                    op_type.value: sum(1 for op in self.operations if op.operation_type == op_type)
                    for op_type in OperationType
                }
            },
            "savepoints": {
                "total": len(self.savepoints),
                "names": [sp.name for sp in self.savepoints]
            },
            "error_message": self.error_message,
            "rollback_reason": self.rollback_reason
        }


@contextmanager
def managed_transaction(
    session: Session,
    auto_commit: bool = True,
    description: Optional[str] = None
):
    """
    Context manager for managed transactions.
    
    Args:
        session: Database session
        auto_commit: Whether to auto-commit on success
        description: Optional description for the transaction
        
    Yields:
        TransactionManager instance
        
    Example:
        with managed_transaction(session, description="Delete video with rollback") as txn:
            # Create savepoint before critical operation
            savepoint = txn.create_savepoint("before_delete")
            
            # Execute operations
            txn.execute_operation(
                OperationType.DELETE,
                "Delete video record",
                "videos",
                lambda: session.delete(video),
                target_id=video.id
            )
            
            # If something goes wrong, rollback to savepoint
            if some_condition:
                txn.rollback_to_savepoint(savepoint, "Condition failed")
    """
    manager = TransactionManager(session, auto_commit)
    
    try:
        if description:
            manager.logger.info(f"Starting transaction {manager.transaction_id}: {description}")
        
        yield manager
        
        # Auto-commit if configured
        if auto_commit:
            result = manager.commit_transaction()
            if not result.success:
                raise DatabaseError(f"Transaction failed: {result.error_message}")
        
    except Exception as e:
        # Auto-rollback on any exception
        manager.rollback_transaction(f"Auto-rollback due to exception: {str(e)}")
        raise
    
    finally:
        # Log transaction summary
        summary = manager.get_transaction_summary()
        manager.logger.info(f"Transaction {manager.transaction_id} ended: {summary}")


# Utility functions for common transaction patterns
def execute_with_rollback(
    session: Session,
    operations: List[Callable],
    operation_descriptions: List[str],
    rollback_on_error: bool = True
) -> TransactionResult:
    """
    Execute multiple operations with automatic rollback on error.
    
    Args:
        session: Database session
        operations: List of operation functions
        operation_descriptions: Descriptions for each operation
        rollback_on_error: Whether to rollback on error
        
    Returns:
        TransactionResult with operation results
    """
    with managed_transaction(session, auto_commit=False) as txn:
        try:
            for i, (operation, description) in enumerate(zip(operations, operation_descriptions)):
                txn.execute_operation(
                    OperationType.UPDATE,  # Generic type
                    description,
                    "multiple",  # Generic table
                    operation
                )
            
            return txn.commit_transaction()
            
        except Exception as e:
            if rollback_on_error:
                return txn.rollback_transaction(f"Operation failed: {str(e)}")
            else:
                raise