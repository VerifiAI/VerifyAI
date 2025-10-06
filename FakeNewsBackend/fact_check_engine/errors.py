import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import traceback
import json

logger = logging.getLogger(__name__)


class ErrorCode(Enum):
    """Standardized error codes for the fact-checking system."""
    
    # General errors (1000-1099)
    UNKNOWN_ERROR = "FC1000"
    INVALID_INPUT = "FC1001"
    MISSING_PARAMETER = "FC1002"
    INVALID_PARAMETER = "FC1003"
    CONFIGURATION_ERROR = "FC1004"
    
    # Authentication/Authorization errors (1100-1199)
    AUTHENTICATION_FAILED = "FC1100"
    AUTHORIZATION_FAILED = "FC1101"
    INVALID_TOKEN = "FC1102"
    TOKEN_EXPIRED = "FC1103"
    INSUFFICIENT_PERMISSIONS = "FC1104"
    
    # API Client errors (1200-1299)
    API_CONNECTION_ERROR = "FC1200"
    API_AUTHENTICATION_ERROR = "FC1201"
    API_RATE_LIMIT_EXCEEDED = "FC1202"
    API_QUOTA_EXCEEDED = "FC1203"
    API_INVALID_RESPONSE = "FC1204"
    API_TIMEOUT = "FC1205"
    API_SERVICE_UNAVAILABLE = "FC1206"
    
    # Claim Processing errors (1300-1399)
    CLAIM_PARSING_FAILED = "FC1300"
    CLAIM_TOO_SHORT = "FC1301"
    CLAIM_TOO_LONG = "FC1302"
    CLAIM_INVALID_FORMAT = "FC1303"
    NO_CLAIMS_FOUND = "FC1304"
    ENTITY_EXTRACTION_FAILED = "FC1305"
    
    # Search errors (1400-1499)
    SEARCH_FAILED = "FC1400"
    NO_SOURCES_FOUND = "FC1401"
    SEARCH_TIMEOUT = "FC1402"
    SEARCH_QUOTA_EXCEEDED = "FC1403"
    INVALID_SEARCH_PARAMETERS = "FC1404"
    
    # Evidence Processing errors (1500-1599)
    EVIDENCE_COLLECTION_FAILED = "FC1500"
    CONTENT_EXTRACTION_FAILED = "FC1501"
    EVIDENCE_VALIDATION_FAILED = "FC1502"
    INSUFFICIENT_EVIDENCE = "FC1503"
    EVIDENCE_QUALITY_TOO_LOW = "FC1504"
    
    # Validation errors (1600-1699)
    PROOF_VALIDATION_FAILED = "FC1600"
    SCORING_FAILED = "FC1601"
    CONSENSUS_BUILDING_FAILED = "FC1602"
    VERDICT_DETERMINATION_FAILED = "FC1603"
    CONFIDENCE_CALCULATION_FAILED = "FC1604"
    
    # System errors (1700-1799)
    DATABASE_ERROR = "FC1700"
    CACHE_ERROR = "FC1701"
    FILE_SYSTEM_ERROR = "FC1702"
    MEMORY_ERROR = "FC1703"
    NETWORK_ERROR = "FC1704"
    
    # Processing errors (1800-1899)
    PROCESSING_TIMEOUT = "FC1800"
    PROCESSING_CANCELLED = "FC1801"
    PROCESSING_FAILED = "FC1802"
    BATCH_PROCESSING_FAILED = "FC1803"
    CONCURRENT_PROCESSING_ERROR = "FC1804"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FactCheckError(Exception):
    """Base exception class for fact-checking system."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        request_id: Optional[str] = None
    ):
        """
        Initialize fact-check error.
        
        Args:
            message: Human-readable error message
            error_code: Standardized error code
            severity: Error severity level
            details: Additional error details
            cause: Original exception that caused this error
            request_id: Associated request ID
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.details = details or {}
        self.cause = cause
        self.request_id = request_id
        self.timestamp = datetime.now()
        
        # Add stack trace if available
        if cause:
            self.details["cause_type"] = type(cause).__name__
            self.details["cause_message"] = str(cause)
            self.details["stack_trace"] = traceback.format_exc()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation."""
        return {
            "error_code": self.error_code.value,
            "message": self.message,
            "severity": self.severity.value,
            "details": self.details,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat()
        }
    
    def to_json(self) -> str:
        """Convert error to JSON string."""
        return json.dumps(self.to_dict(), default=str)


# Specific exception classes
class ConfigurationError(FactCheckError):
    """Configuration-related errors."""
    
    def __init__(self, message: str, config_key: str = None, **kwargs):
        super().__init__(
            message,
            error_code=ErrorCode.CONFIGURATION_ERROR,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        if config_key:
            self.details["config_key"] = config_key


class AuthenticationError(FactCheckError):
    """Authentication-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code=ErrorCode.AUTHENTICATION_FAILED,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class AuthorizationError(FactCheckError):
    """Authorization-related errors."""
    
    def __init__(self, message: str, required_permission: str = None, **kwargs):
        super().__init__(
            message,
            error_code=ErrorCode.AUTHORIZATION_FAILED,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        if required_permission:
            self.details["required_permission"] = required_permission


class APIError(FactCheckError):
    """API-related errors."""
    
    def __init__(
        self,
        message: str,
        api_name: str,
        error_code: ErrorCode = ErrorCode.API_CONNECTION_ERROR,
        status_code: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, error_code=error_code, **kwargs)
        self.details["api_name"] = api_name
        if status_code:
            self.details["status_code"] = status_code


class RateLimitError(APIError):
    """Rate limit exceeded errors."""
    
    def __init__(
        self,
        message: str,
        api_name: str,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            message,
            api_name=api_name,
            error_code=ErrorCode.API_RATE_LIMIT_EXCEEDED,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        if retry_after:
            self.details["retry_after"] = retry_after


class QuotaExceededError(APIError):
    """API quota exceeded errors."""
    
    def __init__(
        self,
        message: str,
        api_name: str,
        quota_limit: Optional[int] = None,
        quota_used: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            message,
            api_name=api_name,
            error_code=ErrorCode.API_QUOTA_EXCEEDED,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        if quota_limit:
            self.details["quota_limit"] = quota_limit
        if quota_used:
            self.details["quota_used"] = quota_used


class ClaimProcessingError(FactCheckError):
    """Claim processing errors."""
    
    def __init__(
        self,
        message: str,
        claim: str = None,
        error_code: ErrorCode = ErrorCode.CLAIM_PARSING_FAILED,
        **kwargs
    ):
        super().__init__(message, error_code=error_code, **kwargs)
        if claim:
            self.details["claim"] = claim[:200]  # Truncate for logging


class SearchError(FactCheckError):
    """Search-related errors."""
    
    def __init__(
        self,
        message: str,
        query: str = None,
        error_code: ErrorCode = ErrorCode.SEARCH_FAILED,
        **kwargs
    ):
        super().__init__(message, error_code=error_code, **kwargs)
        if query:
            self.details["query"] = query


class EvidenceError(FactCheckError):
    """Evidence processing errors."""
    
    def __init__(
        self,
        message: str,
        source_url: str = None,
        error_code: ErrorCode = ErrorCode.EVIDENCE_COLLECTION_FAILED,
        **kwargs
    ):
        super().__init__(message, error_code=error_code, **kwargs)
        if source_url:
            self.details["source_url"] = source_url


class ValidationError(FactCheckError):
    """Validation-related errors."""
    
    def __init__(
        self,
        message: str,
        validation_step: str = None,
        error_code: ErrorCode = ErrorCode.PROOF_VALIDATION_FAILED,
        **kwargs
    ):
        super().__init__(message, error_code=error_code, **kwargs)
        if validation_step:
            self.details["validation_step"] = validation_step


class ProcessingError(FactCheckError):
    """Processing-related errors."""
    
    def __init__(
        self,
        message: str,
        processing_stage: str = None,
        error_code: ErrorCode = ErrorCode.PROCESSING_FAILED,
        **kwargs
    ):
        super().__init__(message, error_code=error_code, **kwargs)
        if processing_stage:
            self.details["processing_stage"] = processing_stage


class TimeoutError(FactCheckError):
    """Timeout-related errors."""
    
    def __init__(
        self,
        message: str,
        timeout_duration: float = None,
        operation: str = None,
        **kwargs
    ):
        super().__init__(
            message,
            error_code=ErrorCode.PROCESSING_TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        if timeout_duration:
            self.details["timeout_duration"] = timeout_duration
        if operation:
            self.details["operation"] = operation


class SystemError(FactCheckError):
    """System-level errors."""
    
    def __init__(
        self,
        message: str,
        system_component: str = None,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        **kwargs
    ):
        super().__init__(
            message,
            error_code=error_code,
            severity=ErrorSeverity.CRITICAL,
            **kwargs
        )
        if system_component:
            self.details["system_component"] = system_component


# Error handling utilities
class ErrorHandler:
    """Centralized error handling utilities."""
    
    def __init__(self):
        """Initialize error handler."""
        self.error_counts = {}
        self.error_patterns = {}
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ) -> FactCheckError:
        """
        Handle and convert exceptions to FactCheckError.
        
        Args:
            error: Original exception
            context: Additional context information
            request_id: Associated request ID
            
        Returns:
            FactCheckError instance
        """
        # If already a FactCheckError, return as-is
        if isinstance(error, FactCheckError):
            return error
        
        # Convert common exceptions
        if isinstance(error, ValueError):
            return FactCheckError(
                message=str(error),
                error_code=ErrorCode.INVALID_PARAMETER,
                cause=error,
                request_id=request_id,
                details=context
            )
        elif isinstance(error, KeyError):
            return FactCheckError(
                message=f"Missing required parameter: {str(error)}",
                error_code=ErrorCode.MISSING_PARAMETER,
                cause=error,
                request_id=request_id,
                details=context
            )
        elif isinstance(error, ConnectionError):
            return FactCheckError(
                message=f"Network connection error: {str(error)}",
                error_code=ErrorCode.NETWORK_ERROR,
                cause=error,
                request_id=request_id,
                details=context
            )
        elif isinstance(error, TimeoutError):
            return FactCheckError(
                message=f"Operation timed out: {str(error)}",
                error_code=ErrorCode.PROCESSING_TIMEOUT,
                cause=error,
                request_id=request_id,
                details=context
            )
        else:
            # Generic error handling
            return FactCheckError(
                message=f"Unexpected error: {str(error)}",
                error_code=ErrorCode.UNKNOWN_ERROR,
                severity=ErrorSeverity.HIGH,
                cause=error,
                request_id=request_id,
                details=context
            )
    
    def log_error(self, error: FactCheckError):
        """Log error with appropriate level based on severity."""
        error_dict = error.to_dict()
        
        if error.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL ERROR: {error.message}", extra=error_dict)
        elif error.severity == ErrorSeverity.HIGH:
            logger.error(f"HIGH SEVERITY ERROR: {error.message}", extra=error_dict)
        elif error.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"MEDIUM SEVERITY ERROR: {error.message}", extra=error_dict)
        else:
            logger.info(f"LOW SEVERITY ERROR: {error.message}", extra=error_dict)
        
        # Track error patterns
        self._track_error(error)
    
    def _track_error(self, error: FactCheckError):
        """Track error for pattern analysis."""
        error_key = error.error_code.value
        
        if error_key not in self.error_counts:
            self.error_counts[error_key] = 0
        
        self.error_counts[error_key] += 1
        
        # Store recent error pattern
        if error_key not in self.error_patterns:
            self.error_patterns[error_key] = []
        
        self.error_patterns[error_key].append({
            "timestamp": error.timestamp,
            "request_id": error.request_id,
            "details": error.details
        })
        
        # Keep only recent errors (last 100)
        if len(self.error_patterns[error_key]) > 100:
            self.error_patterns[error_key] = self.error_patterns[error_key][-100:]
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and patterns."""
        return {
            "error_counts": dict(self.error_counts),
            "total_errors": sum(self.error_counts.values()),
            "most_common_errors": sorted(
                self.error_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            "error_patterns_tracked": len(self.error_patterns)
        }
    
    def check_error_patterns(self) -> List[Dict[str, Any]]:
        """Check for concerning error patterns."""
        alerts = []
        
        # Check for high error rates
        for error_code, count in self.error_counts.items():
            if count > 50:  # Threshold for concern
                alerts.append({
                    "type": "high_error_rate",
                    "error_code": error_code,
                    "count": count,
                    "severity": "high"
                })
        
        # Check for recent spikes
        now = datetime.now()
        recent_threshold = now - timedelta(minutes=15)
        
        for error_code, patterns in self.error_patterns.items():
            recent_errors = [
                p for p in patterns
                if p["timestamp"] > recent_threshold
            ]
            
            if len(recent_errors) > 10:  # More than 10 errors in 15 minutes
                alerts.append({
                    "type": "error_spike",
                    "error_code": error_code,
                    "recent_count": len(recent_errors),
                    "severity": "medium"
                })
        
        return alerts


# Global error handler instance
error_handler = ErrorHandler()


# Decorator for error handling
def handle_errors(request_id_param: str = "request_id"):
    """
    Decorator for automatic error handling in functions.
    
    Args:
        request_id_param: Name of parameter containing request ID
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            request_id = kwargs.get(request_id_param)
            
            try:
                return await func(*args, **kwargs)
            except FactCheckError:
                # Re-raise FactCheckErrors as-is
                raise
            except Exception as e:
                # Convert other exceptions
                fact_check_error = error_handler.handle_error(
                    e,
                    context={"function": func.__name__, "args": str(args)[:200]},
                    request_id=request_id
                )
                error_handler.log_error(fact_check_error)
                raise fact_check_error
        
        return wrapper
    return decorator


# Utility functions
def create_error_response(error: FactCheckError) -> Dict[str, Any]:
    """Create standardized error response."""
    return {
        "success": False,
        "error": {
            "code": error.error_code.value,
            "message": error.message,
            "severity": error.severity.value,
            "timestamp": error.timestamp.isoformat(),
            "request_id": error.request_id
        },
        "details": error.details
    }


def is_retryable_error(error: FactCheckError) -> bool:
    """Check if error is retryable."""
    retryable_codes = [
        ErrorCode.API_TIMEOUT,
        ErrorCode.API_SERVICE_UNAVAILABLE,
        ErrorCode.NETWORK_ERROR,
        ErrorCode.PROCESSING_TIMEOUT
    ]
    
    return error.error_code in retryable_codes


def get_retry_delay(error: FactCheckError, attempt: int) -> float:
    """Get retry delay for retryable errors."""
    if error.error_code == ErrorCode.API_RATE_LIMIT_EXCEEDED:
        # Use retry_after if available, otherwise exponential backoff
        return error.details.get("retry_after", 2 ** attempt)
    else:
        # Exponential backoff with jitter
        import random
        base_delay = 2 ** attempt
        jitter = random.uniform(0.1, 0.5)
        return base_delay + jitter


## Suggestions for Upgrade:
# 1. Integrate with external error tracking services like Sentry, Rollbar, or Bugsnag
# 2. Add machine learning-based error pattern detection and prediction
# 3. Implement automatic error recovery and self-healing mechanisms
# 4. Add comprehensive error documentation and troubleshooting guides