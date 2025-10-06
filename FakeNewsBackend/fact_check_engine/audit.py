import asyncio
import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from pathlib import Path

try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False
    logging.warning("aiofiles not available. File-based audit logging will be limited.")

from .config import config

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events."""
    FACT_CHECK_REQUEST = "fact_check_request"
    CLAIM_PARSED = "claim_parsed"
    SEARCH_PERFORMED = "search_performed"
    EVIDENCE_COLLECTED = "evidence_collected"
    PROOF_VALIDATED = "proof_validated"
    CONSENSUS_BUILT = "consensus_built"
    VERDICT_DETERMINED = "verdict_determined"
    API_CALL = "api_call"
    ERROR_OCCURRED = "error_occurred"
    SYSTEM_EVENT = "system_event"


class AuditLevel(Enum):
    """Audit logging levels."""
    MINIMAL = "minimal"      # Only critical events
    STANDARD = "standard"    # Standard operations
    DETAILED = "detailed"    # Detailed analysis steps
    COMPREHENSIVE = "comprehensive"  # All events including debug


@dataclass
class AuditEvent:
    """Audit event data structure."""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    request_id: Optional[str]
    user_id: Optional[str]
    session_id: Optional[str]
    event_data: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "request_id": self.request_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "event_data": self.event_data,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


@dataclass
class EvidenceTrail:
    """Evidence trail for a fact-checking operation."""
    request_id: str
    claim: str
    sources_accessed: List[Dict[str, Any]]
    evidence_collected: List[Dict[str, Any]]
    validation_steps: List[Dict[str, Any]]
    decision_factors: List[Dict[str, Any]]
    final_verdict: Dict[str, Any]
    processing_time: float
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def get_evidence_hash(self) -> str:
        """Generate hash of evidence for integrity verification."""
        evidence_str = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(evidence_str.encode()).hexdigest()


class AuditLogger:
    """Comprehensive audit logging system."""
    
    def __init__(self, audit_level: AuditLevel = AuditLevel.STANDARD):
        """Initialize audit logger."""
        self.audit_level = audit_level
        self.audit_dir = Path("logs/audit")
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        
        # Audit configuration
        self.config = {
            "max_file_size": 100 * 1024 * 1024,  # 100MB
            "max_files": 30,  # Keep 30 days of logs
            "compress_old_files": True,
            "enable_real_time_alerts": True,
            "retention_days": 365
        }
        
        # Event counters
        self.event_counters = {event_type: 0 for event_type in AuditEventType}
        
        # Evidence trails storage
        self.evidence_trails: Dict[str, EvidenceTrail] = {}
        
        logger.info(f"AuditLogger initialized with level: {audit_level.value}")
    
    async def log_event(
        self,
        event_type: AuditEventType,
        event_data: Dict[str, Any],
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log an audit event.
        
        Args:
            event_type: Type of event
            event_data: Event-specific data
            request_id: Associated request ID
            user_id: User who triggered the event
            session_id: Session ID
            metadata: Additional metadata
            
        Returns:
            Event ID
        """
        # Check if event should be logged based on audit level
        if not self._should_log_event(event_type):
            return ""
        
        event_id = str(uuid.uuid4())
        
        # Create audit event
        audit_event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.now(),
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            event_data=event_data,
            metadata=metadata or {}
        )
        
        # Log to file
        await self._write_audit_log(audit_event)
        
        # Update counters
        self.event_counters[event_type] += 1
        
        # Check for alerts
        await self._check_alerts(audit_event)
        
        logger.debug(f"Audit event logged: {event_type.value} ({event_id})")
        return event_id
    
    def _should_log_event(self, event_type: AuditEventType) -> bool:
        """Determine if event should be logged based on audit level."""
        if self.audit_level == AuditLevel.MINIMAL:
            return event_type in [
                AuditEventType.FACT_CHECK_REQUEST,
                AuditEventType.VERDICT_DETERMINED,
                AuditEventType.ERROR_OCCURRED
            ]
        elif self.audit_level == AuditLevel.STANDARD:
            return event_type not in [AuditEventType.SYSTEM_EVENT]
        else:
            return True  # Log all events for DETAILED and COMPREHENSIVE
    
    async def _write_audit_log(self, event: AuditEvent):
        """Write audit event to log file."""
        try:
            # Generate filename based on date
            log_file = self.audit_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
            
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(log_file, 'a') as f:
                    await f.write(event.to_json() + '\n')
            else:
                # Fallback to synchronous writing
                with open(log_file, 'a') as f:
                    f.write(event.to_json() + '\n')
            
            # Check file rotation
            await self._rotate_logs_if_needed()
            
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    async def _rotate_logs_if_needed(self):
        """Rotate log files if they exceed size limits."""
        try:
            current_log = self.audit_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
            
            if current_log.exists() and current_log.stat().st_size > self.config["max_file_size"]:
                # Rotate current log
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                rotated_name = self.audit_dir / f"audit_{timestamp}.jsonl"
                current_log.rename(rotated_name)
                
                logger.info(f"Rotated audit log: {rotated_name}")
            
            # Clean up old logs
            await self._cleanup_old_logs()
            
        except Exception as e:
            logger.error(f"Log rotation failed: {e}")
    
    async def _cleanup_old_logs(self):
        """Clean up old audit logs based on retention policy."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config["retention_days"])
            
            for log_file in self.audit_dir.glob("audit_*.jsonl"):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    log_file.unlink()
                    logger.info(f"Deleted old audit log: {log_file}")
                    
        except Exception as e:
            logger.error(f"Log cleanup failed: {e}")
    
    async def _check_alerts(self, event: AuditEvent):
        """Check for conditions that should trigger alerts."""
        if not self.config["enable_real_time_alerts"]:
            return
        
        # Alert conditions
        alert_conditions = [
            # High error rate
            (event.event_type == AuditEventType.ERROR_OCCURRED and 
             self.event_counters[AuditEventType.ERROR_OCCURRED] % 10 == 0),
            
            # Suspicious activity patterns
            (event.event_type == AuditEventType.FACT_CHECK_REQUEST and
             event.user_id and self._check_rate_limit_exceeded(event.user_id)),
        ]
        
        for condition in alert_conditions:
            if condition:
                await self._send_alert(event)
                break
    
    def _check_rate_limit_exceeded(self, user_id: str) -> bool:
        """Check if user has exceeded rate limits."""
        # This would implement actual rate limit checking
        # For now, return False
        return False
    
    async def _send_alert(self, event: AuditEvent):
        """Send alert for suspicious or error conditions."""
        alert_data = {
            "alert_type": "audit_alert",
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "timestamp": event.timestamp.isoformat(),
            "details": event.event_data
        }
        
        logger.warning(f"AUDIT ALERT: {json.dumps(alert_data)}")
        
        # In production, this would send to monitoring systems
        # like Slack, PagerDuty, or email notifications
    
    async def start_evidence_trail(
        self,
        request_id: str,
        claim: str,
        user_id: Optional[str] = None
    ) -> str:
        """Start tracking evidence trail for a fact-checking request."""
        trail = EvidenceTrail(
            request_id=request_id,
            claim=claim,
            sources_accessed=[],
            evidence_collected=[],
            validation_steps=[],
            decision_factors=[],
            final_verdict={},
            processing_time=0.0,
            created_at=datetime.now()
        )
        
        self.evidence_trails[request_id] = trail
        
        await self.log_event(
            AuditEventType.FACT_CHECK_REQUEST,
            {
                "claim": claim,
                "claim_length": len(claim),
                "trail_started": True
            },
            request_id=request_id,
            user_id=user_id
        )
        
        return request_id
    
    async def add_source_access(
        self,
        request_id: str,
        source_info: Dict[str, Any]
    ):
        """Add source access to evidence trail."""
        if request_id in self.evidence_trails:
            self.evidence_trails[request_id].sources_accessed.append({
                **source_info,
                "accessed_at": datetime.now().isoformat()
            })
            
            await self.log_event(
                AuditEventType.SEARCH_PERFORMED,
                source_info,
                request_id=request_id
            )
    
    async def add_evidence_collection(
        self,
        request_id: str,
        evidence_info: Dict[str, Any]
    ):
        """Add evidence collection to trail."""
        if request_id in self.evidence_trails:
            self.evidence_trails[request_id].evidence_collected.append({
                **evidence_info,
                "collected_at": datetime.now().isoformat()
            })
            
            await self.log_event(
                AuditEventType.EVIDENCE_COLLECTED,
                evidence_info,
                request_id=request_id
            )
    
    async def add_validation_step(
        self,
        request_id: str,
        validation_info: Dict[str, Any]
    ):
        """Add validation step to trail."""
        if request_id in self.evidence_trails:
            self.evidence_trails[request_id].validation_steps.append({
                **validation_info,
                "validated_at": datetime.now().isoformat()
            })
            
            await self.log_event(
                AuditEventType.PROOF_VALIDATED,
                validation_info,
                request_id=request_id
            )
    
    async def add_decision_factor(
        self,
        request_id: str,
        factor_info: Dict[str, Any]
    ):
        """Add decision factor to trail."""
        if request_id in self.evidence_trails:
            self.evidence_trails[request_id].decision_factors.append({
                **factor_info,
                "recorded_at": datetime.now().isoformat()
            })
    
    async def complete_evidence_trail(
        self,
        request_id: str,
        final_verdict: Dict[str, Any],
        processing_time: float
    ):
        """Complete evidence trail with final verdict."""
        if request_id in self.evidence_trails:
            trail = self.evidence_trails[request_id]
            trail.final_verdict = final_verdict
            trail.processing_time = processing_time
            
            # Log completion
            await self.log_event(
                AuditEventType.VERDICT_DETERMINED,
                {
                    "verdict": final_verdict.get("verdict"),
                    "confidence": final_verdict.get("confidence_score"),
                    "processing_time": processing_time,
                    "sources_count": len(trail.sources_accessed),
                    "evidence_count": len(trail.evidence_collected),
                    "evidence_hash": trail.get_evidence_hash()
                },
                request_id=request_id
            )
            
            # Archive trail
            await self._archive_evidence_trail(request_id, trail)
    
    async def _archive_evidence_trail(self, request_id: str, trail: EvidenceTrail):
        """Archive evidence trail to permanent storage."""
        try:
            archive_file = self.audit_dir / f"evidence_trails_{datetime.now().strftime('%Y%m')}.jsonl"
            
            trail_data = {
                "trail_id": request_id,
                "trail_data": trail.to_dict(),
                "evidence_hash": trail.get_evidence_hash(),
                "archived_at": datetime.now().isoformat()
            }
            
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(archive_file, 'a') as f:
                    await f.write(json.dumps(trail_data, default=str) + '\n')
            else:
                with open(archive_file, 'a') as f:
                    f.write(json.dumps(trail_data, default=str) + '\n')
            
            # Remove from memory
            del self.evidence_trails[request_id]
            
            logger.debug(f"Evidence trail archived: {request_id}")
            
        except Exception as e:
            logger.error(f"Failed to archive evidence trail: {e}")
    
    async def get_audit_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get audit summary for the specified number of days."""
        try:
            summary = {
                "period_days": days,
                "event_counts": dict(self.event_counters),
                "active_trails": len(self.evidence_trails),
                "total_events": sum(self.event_counters.values()),
                "generated_at": datetime.now().isoformat()
            }
            
            # Calculate rates
            if days > 0:
                daily_average = sum(self.event_counters.values()) / days
                summary["daily_average_events"] = daily_average
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate audit summary: {e}")
            return {"error": str(e)}
    
    async def search_audit_logs(
        self,
        event_type: Optional[AuditEventType] = None,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search audit logs with filters."""
        try:
            results = []
            
            # This is a simplified implementation
            # In production, you'd use a proper search index or database
            
            # Search through recent log files
            search_date = start_date or (datetime.now() - timedelta(days=7))
            
            while search_date <= (end_date or datetime.now()):
                log_file = self.audit_dir / f"audit_{search_date.strftime('%Y%m%d')}.jsonl"
                
                if log_file.exists():
                    try:
                        with open(log_file, 'r') as f:
                            for line in f:
                                if len(results) >= limit:
                                    break
                                
                                try:
                                    event = json.loads(line.strip())
                                    
                                    # Apply filters
                                    if event_type and event.get("event_type") != event_type.value:
                                        continue
                                    if request_id and event.get("request_id") != request_id:
                                        continue
                                    if user_id and event.get("user_id") != user_id:
                                        continue
                                    
                                    results.append(event)
                                    
                                except json.JSONDecodeError:
                                    continue
                    except Exception as e:
                        logger.error(f"Error reading log file {log_file}: {e}")
                
                search_date += timedelta(days=1)
            
            return results
            
        except Exception as e:
            logger.error(f"Audit log search failed: {e}")
            return []
    
    async def verify_evidence_integrity(self, request_id: str) -> Dict[str, Any]:
        """Verify integrity of evidence trail."""
        try:
            # Search for archived trail
            current_month = datetime.now().strftime('%Y%m')
            archive_file = self.audit_dir / f"evidence_trails_{current_month}.jsonl"
            
            if not archive_file.exists():
                return {"error": "Evidence trail not found"}
            
            with open(archive_file, 'r') as f:
                for line in f:
                    try:
                        trail_data = json.loads(line.strip())
                        if trail_data.get("trail_id") == request_id:
                            # Verify hash
                            stored_hash = trail_data.get("evidence_hash")
                            trail = EvidenceTrail(**trail_data["trail_data"])
                            calculated_hash = trail.get_evidence_hash()
                            
                            return {
                                "request_id": request_id,
                                "integrity_verified": stored_hash == calculated_hash,
                                "stored_hash": stored_hash,
                                "calculated_hash": calculated_hash,
                                "verified_at": datetime.now().isoformat()
                            }
                    except json.JSONDecodeError:
                        continue
            
            return {"error": "Evidence trail not found in archives"}
            
        except Exception as e:
            logger.error(f"Evidence integrity verification failed: {e}")
            return {"error": str(e)}


# Global audit logger instance
audit_logger = AuditLogger(AuditLevel.STANDARD)


# Convenience functions
async def log_fact_check_request(claim: str, request_id: str, user_id: str = None):
    """Log fact-check request."""
    await audit_logger.start_evidence_trail(request_id, claim, user_id)


async def log_api_call(api_name: str, endpoint: str, response_code: int, request_id: str = None):
    """Log API call."""
    await audit_logger.log_event(
        AuditEventType.API_CALL,
        {
            "api_name": api_name,
            "endpoint": endpoint,
            "response_code": response_code,
            "timestamp": datetime.now().isoformat()
        },
        request_id=request_id
    )


async def log_error(error_type: str, error_message: str, request_id: str = None, **kwargs):
    """Log error event."""
    await audit_logger.log_event(
        AuditEventType.ERROR_OCCURRED,
        {
            "error_type": error_type,
            "error_message": error_message,
            "additional_info": kwargs
        },
        request_id=request_id
    )


## Suggestions for Upgrade:
# 1. Integrate with enterprise audit systems like Splunk, ELK stack, or cloud logging services
# 2. Add blockchain-based evidence integrity verification for tamper-proof audit trails
# 3. Implement real-time audit dashboards with anomaly detection and alerting
# 4. Add GDPR compliance features for data retention, anonymization, and right-to-be-forgotten