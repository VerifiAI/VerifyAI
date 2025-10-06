import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import json

from .cache import cache, CacheKey
from .config import config
from .errors import QuotaExceededError, RateLimitError, ErrorCode

logger = logging.getLogger(__name__)


class QuotaType(Enum):
    """Types of quota limits."""
    REQUESTS_PER_SECOND = "requests_per_second"
    REQUESTS_PER_MINUTE = "requests_per_minute"
    REQUESTS_PER_HOUR = "requests_per_hour"
    REQUESTS_PER_DAY = "requests_per_day"
    REQUESTS_PER_MONTH = "requests_per_month"
    TOTAL_REQUESTS = "total_requests"
    DATA_TRANSFER = "data_transfer"
    CONCURRENT_REQUESTS = "concurrent_requests"


class QuotaStatus(Enum):
    """Quota status levels."""
    AVAILABLE = "available"
    WARNING = "warning"      # 80-95% used
    CRITICAL = "critical"    # 95-100% used
    EXCEEDED = "exceeded"    # Over 100%


@dataclass
class QuotaLimit:
    """Quota limit configuration."""
    quota_type: QuotaType
    limit: int
    window_seconds: int
    warning_threshold: float = 0.8  # 80%
    critical_threshold: float = 0.95  # 95%
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "quota_type": self.quota_type.value,
            "limit": self.limit,
            "window_seconds": self.window_seconds,
            "warning_threshold": self.warning_threshold,
            "critical_threshold": self.critical_threshold
        }


@dataclass
class QuotaUsage:
    """Current quota usage information."""
    api_name: str
    quota_type: QuotaType
    current_usage: int
    limit: int
    remaining: int
    reset_time: datetime
    status: QuotaStatus
    usage_percentage: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "api_name": self.api_name,
            "quota_type": self.quota_type.value,
            "current_usage": self.current_usage,
            "limit": self.limit,
            "remaining": self.remaining,
            "reset_time": self.reset_time.isoformat(),
            "status": self.status.value,
            "usage_percentage": self.usage_percentage
        }


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration."""
    identifier: str  # API name, user ID, IP address, etc.
    requests_per_window: int
    window_seconds: int
    burst_allowance: int = 0  # Additional requests allowed in burst
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "identifier": self.identifier,
            "requests_per_window": self.requests_per_window,
            "window_seconds": self.window_seconds,
            "burst_allowance": self.burst_allowance
        }


class QuotaManager:
    """Comprehensive quota management and rate limiting system."""
    
    def __init__(self):
        """Initialize quota manager."""
        # API quota configurations
        self.api_quotas = self._load_api_quotas()
        
        # Rate limiting rules
        self.rate_limit_rules = self._load_rate_limit_rules()
        
        # Usage tracking
        self.usage_stats = {}
        
        # Alert thresholds
        self.alert_thresholds = {
            "warning": 0.8,
            "critical": 0.95
        }
        
        # Quota refresh intervals
        self.refresh_intervals = {
            QuotaType.REQUESTS_PER_SECOND: 1,
            QuotaType.REQUESTS_PER_MINUTE: 60,
            QuotaType.REQUESTS_PER_HOUR: 3600,
            QuotaType.REQUESTS_PER_DAY: 86400,
            QuotaType.REQUESTS_PER_MONTH: 2592000  # 30 days
        }
        
        logger.info("QuotaManager initialized")
    
    def _load_api_quotas(self) -> Dict[str, List[QuotaLimit]]:
        """Load API quota configurations."""
        return {
            "newsapi": [
                QuotaLimit(QuotaType.REQUESTS_PER_DAY, config.api.news_api_rate_limit, 86400),
                QuotaLimit(QuotaType.REQUESTS_PER_SECOND, 5, 1)  # Conservative rate limit
            ],
            "serper": [
                QuotaLimit(QuotaType.REQUESTS_PER_MONTH, config.api.serper_api_rate_limit, 2592000),
                QuotaLimit(QuotaType.REQUESTS_PER_SECOND, 10, 1)
            ],
            "newsdata": [
                QuotaLimit(QuotaType.REQUESTS_PER_DAY, config.api.newsdata_api_rate_limit, 86400),
                QuotaLimit(QuotaType.REQUESTS_PER_SECOND, 3, 1)
            ]
        }
    
    def _load_rate_limit_rules(self) -> Dict[str, RateLimitRule]:
        """Load rate limiting rules."""
        return {
            "global": RateLimitRule("global", 1000, 3600),  # 1000 requests per hour globally
            "per_user": RateLimitRule("user", 100, 3600),   # 100 requests per hour per user
            "per_ip": RateLimitRule("ip", 200, 3600),       # 200 requests per hour per IP
            "fact_check": RateLimitRule("fact_check", 50, 3600),  # 50 fact-checks per hour per user
            "search": RateLimitRule("search", 200, 3600)    # 200 searches per hour per user
        }
    
    async def check_quota(
        self,
        api_name: str,
        quota_type: QuotaType,
        requested_amount: int = 1
    ) -> Tuple[bool, QuotaUsage]:
        """
        Check if quota allows the requested usage.
        
        Args:
            api_name: Name of the API
            quota_type: Type of quota to check
            requested_amount: Amount of quota requested
            
        Returns:
            Tuple of (allowed, quota_usage)
        """
        try:
            # Get quota limit
            quota_limit = self._get_quota_limit(api_name, quota_type)
            if not quota_limit:
                # No quota limit configured, allow request
                return True, self._create_unlimited_quota_usage(api_name, quota_type)
            
            # Get current usage
            current_usage = await self._get_current_usage(api_name, quota_type)
            
            # Calculate remaining quota
            remaining = max(0, quota_limit.limit - current_usage)
            
            # Check if request can be fulfilled
            allowed = remaining >= requested_amount
            
            # Calculate usage percentage
            usage_percentage = (current_usage / quota_limit.limit) * 100
            
            # Determine status
            if usage_percentage >= 100:
                status = QuotaStatus.EXCEEDED
            elif usage_percentage >= quota_limit.critical_threshold * 100:
                status = QuotaStatus.CRITICAL
            elif usage_percentage >= quota_limit.warning_threshold * 100:
                status = QuotaStatus.WARNING
            else:
                status = QuotaStatus.AVAILABLE
            
            # Calculate reset time
            reset_time = self._calculate_reset_time(quota_type)
            
            quota_usage = QuotaUsage(
                api_name=api_name,
                quota_type=quota_type,
                current_usage=current_usage,
                limit=quota_limit.limit,
                remaining=remaining,
                reset_time=reset_time,
                status=status,
                usage_percentage=usage_percentage
            )
            
            return allowed, quota_usage
            
        except Exception as e:
            logger.error(f"Quota check failed for {api_name}/{quota_type.value}: {e}")
            # On error, allow request but log the issue
            return True, self._create_error_quota_usage(api_name, quota_type, str(e))
    
    async def consume_quota(
        self,
        api_name: str,
        quota_type: QuotaType,
        amount: int = 1
    ) -> bool:
        """
        Consume quota for an API request.
        
        Args:
            api_name: Name of the API
            quota_type: Type of quota to consume
            amount: Amount of quota to consume
            
        Returns:
            True if quota was consumed successfully
        """
        try:
            # Check quota first
            allowed, quota_usage = await self.check_quota(api_name, quota_type, amount)
            
            if not allowed:
                raise QuotaExceededError(
                    f"Quota exceeded for {api_name} ({quota_type.value})",
                    api_name=api_name,
                    quota_limit=quota_usage.limit,
                    quota_used=quota_usage.current_usage
                )
            
            # Consume quota
            await self._increment_usage(api_name, quota_type, amount)
            
            # Check for alerts
            await self._check_quota_alerts(quota_usage)
            
            return True
            
        except QuotaExceededError:
            raise
        except Exception as e:
            logger.error(f"Quota consumption failed for {api_name}/{quota_type.value}: {e}")
            return False
    
    async def check_rate_limit(
        self,
        rule_name: str,
        identifier: str,
        requested_amount: int = 1
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check rate limit for a specific identifier.
        
        Args:
            rule_name: Name of the rate limit rule
            identifier: Unique identifier (user ID, IP, etc.)
            requested_amount: Number of requests
            
        Returns:
            Tuple of (allowed, rate_limit_info)
        """
        try:
            rule = self.rate_limit_rules.get(rule_name)
            if not rule:
                # No rule configured, allow request
                return True, {"status": "no_limit"}
            
            # Get current usage in window
            cache_key = CacheKey.rate_limit(f"{rule_name}:{identifier}", str(rule.window_seconds))
            current_usage = await cache.get(cache_key, default=0)
            
            # Check if request is allowed
            effective_limit = rule.requests_per_window + rule.burst_allowance
            allowed = current_usage + requested_amount <= effective_limit
            
            # Calculate reset time
            ttl = await cache.get_ttl(cache_key)
            reset_time = datetime.now() + timedelta(seconds=max(0, ttl))
            
            rate_limit_info = {
                "rule_name": rule_name,
                "identifier": identifier,
                "current_usage": current_usage,
                "limit": rule.requests_per_window,
                "burst_allowance": rule.burst_allowance,
                "effective_limit": effective_limit,
                "remaining": max(0, effective_limit - current_usage),
                "reset_time": reset_time.isoformat(),
                "window_seconds": rule.window_seconds,
                "allowed": allowed
            }
            
            return allowed, rate_limit_info
            
        except Exception as e:
            logger.error(f"Rate limit check failed for {rule_name}/{identifier}: {e}")
            return True, {"status": "error", "error": str(e)}
    
    async def consume_rate_limit(
        self,
        rule_name: str,
        identifier: str,
        amount: int = 1
    ) -> bool:
        """
        Consume rate limit quota.
        
        Args:
            rule_name: Name of the rate limit rule
            identifier: Unique identifier
            amount: Amount to consume
            
        Returns:
            True if successful
        """
        try:
            # Check rate limit first
            allowed, rate_limit_info = await self.check_rate_limit(rule_name, identifier, amount)
            
            if not allowed:
                raise RateLimitError(
                    f"Rate limit exceeded for {rule_name}",
                    api_name=rule_name,
                    retry_after=rate_limit_info.get("window_seconds", 60)
                )
            
            # Increment usage
            rule = self.rate_limit_rules[rule_name]
            cache_key = CacheKey.rate_limit(f"{rule_name}:{identifier}", str(rule.window_seconds))
            await cache.increment(cache_key, amount, ttl=rule.window_seconds)
            
            return True
            
        except RateLimitError:
            raise
        except Exception as e:
            logger.error(f"Rate limit consumption failed for {rule_name}/{identifier}: {e}")
            return False
    
    async def get_quota_status(self, api_name: str) -> Dict[str, Any]:
        """
        Get comprehensive quota status for an API.
        
        Args:
            api_name: Name of the API
            
        Returns:
            Quota status information
        """
        try:
            quotas = self.api_quotas.get(api_name, [])
            quota_statuses = []
            
            for quota_limit in quotas:
                _, quota_usage = await self.check_quota(api_name, quota_limit.quota_type, 0)
                quota_statuses.append(quota_usage.to_dict())
            
            # Calculate overall status
            overall_status = QuotaStatus.AVAILABLE
            for quota_status in quota_statuses:
                if quota_status["status"] == QuotaStatus.EXCEEDED.value:
                    overall_status = QuotaStatus.EXCEEDED
                    break
                elif quota_status["status"] == QuotaStatus.CRITICAL.value:
                    overall_status = QuotaStatus.CRITICAL
                elif quota_status["status"] == QuotaStatus.WARNING.value and overall_status == QuotaStatus.AVAILABLE:
                    overall_status = QuotaStatus.WARNING
            
            return {
                "api_name": api_name,
                "overall_status": overall_status.value,
                "quotas": quota_statuses,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get quota status for {api_name}: {e}")
            return {
                "api_name": api_name,
                "overall_status": "error",
                "error": str(e)
            }
    
    async def get_all_quota_status(self) -> Dict[str, Any]:
        """Get quota status for all configured APIs."""
        try:
            all_status = {}
            
            for api_name in self.api_quotas.keys():
                all_status[api_name] = await self.get_quota_status(api_name)
            
            return {
                "quotas": all_status,
                "summary": self._generate_quota_summary(all_status),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get all quota status: {e}")
            return {"error": str(e)}
    
    async def reset_quota(self, api_name: str, quota_type: QuotaType) -> bool:
        """
        Reset quota usage for testing or emergency purposes.
        
        Args:
            api_name: Name of the API
            quota_type: Type of quota to reset
            
        Returns:
            True if successful
        """
        try:
            cache_key = self._get_usage_cache_key(api_name, quota_type)
            await cache.delete(cache_key)
            
            logger.info(f"Reset quota for {api_name}/{quota_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset quota for {api_name}/{quota_type.value}: {e}")
            return False
    
    async def get_usage_statistics(self, days: int = 7) -> Dict[str, Any]:
        """
        Get usage statistics for the specified period.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Usage statistics
        """
        try:
            # This is a simplified implementation
            # In production, you'd store historical data
            
            stats = {
                "period_days": days,
                "api_usage": {},
                "rate_limit_hits": 0,
                "quota_exceeded_count": 0,
                "generated_at": datetime.now().isoformat()
            }
            
            # Get current usage for all APIs
            for api_name in self.api_quotas.keys():
                api_stats = await self.get_quota_status(api_name)
                stats["api_usage"][api_name] = api_stats
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get usage statistics: {e}")
            return {"error": str(e)}
    
    def _get_quota_limit(self, api_name: str, quota_type: QuotaType) -> Optional[QuotaLimit]:
        """Get quota limit configuration for API and type."""
        quotas = self.api_quotas.get(api_name, [])
        for quota in quotas:
            if quota.quota_type == quota_type:
                return quota
        return None
    
    async def _get_current_usage(self, api_name: str, quota_type: QuotaType) -> int:
        """Get current usage for API and quota type."""
        cache_key = self._get_usage_cache_key(api_name, quota_type)
        return await cache.get(cache_key, default=0)
    
    async def _increment_usage(self, api_name: str, quota_type: QuotaType, amount: int):
        """Increment usage counter."""
        cache_key = self._get_usage_cache_key(api_name, quota_type)
        ttl = self.refresh_intervals.get(quota_type, 3600)
        await cache.increment(cache_key, amount, ttl=ttl)
    
    def _get_usage_cache_key(self, api_name: str, quota_type: QuotaType) -> str:
        """Generate cache key for usage tracking."""
        # Include time window in key for automatic expiration
        now = datetime.now()
        
        if quota_type == QuotaType.REQUESTS_PER_SECOND:
            window = now.strftime("%Y%m%d%H%M%S")
        elif quota_type == QuotaType.REQUESTS_PER_MINUTE:
            window = now.strftime("%Y%m%d%H%M")
        elif quota_type == QuotaType.REQUESTS_PER_HOUR:
            window = now.strftime("%Y%m%d%H")
        elif quota_type == QuotaType.REQUESTS_PER_DAY:
            window = now.strftime("%Y%m%d")
        elif quota_type == QuotaType.REQUESTS_PER_MONTH:
            window = now.strftime("%Y%m")
        else:
            window = "total"
        
        return f"quota:{api_name}:{quota_type.value}:{window}"
    
    def _calculate_reset_time(self, quota_type: QuotaType) -> datetime:
        """Calculate when quota will reset."""
        now = datetime.now()
        
        if quota_type == QuotaType.REQUESTS_PER_SECOND:
            return now.replace(microsecond=0) + timedelta(seconds=1)
        elif quota_type == QuotaType.REQUESTS_PER_MINUTE:
            return now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        elif quota_type == QuotaType.REQUESTS_PER_HOUR:
            return now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        elif quota_type == QuotaType.REQUESTS_PER_DAY:
            return now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        elif quota_type == QuotaType.REQUESTS_PER_MONTH:
            # First day of next month
            if now.month == 12:
                return datetime(now.year + 1, 1, 1)
            else:
                return datetime(now.year, now.month + 1, 1)
        else:
            return now + timedelta(hours=1)  # Default
    
    def _create_unlimited_quota_usage(self, api_name: str, quota_type: QuotaType) -> QuotaUsage:
        """Create quota usage for unlimited quota."""
        return QuotaUsage(
            api_name=api_name,
            quota_type=quota_type,
            current_usage=0,
            limit=-1,  # Unlimited
            remaining=-1,
            reset_time=datetime.now() + timedelta(hours=1),
            status=QuotaStatus.AVAILABLE,
            usage_percentage=0.0
        )
    
    def _create_error_quota_usage(self, api_name: str, quota_type: QuotaType, error: str) -> QuotaUsage:
        """Create quota usage for error cases."""
        return QuotaUsage(
            api_name=api_name,
            quota_type=quota_type,
            current_usage=0,
            limit=0,
            remaining=0,
            reset_time=datetime.now(),
            status=QuotaStatus.AVAILABLE,  # Allow on error
            usage_percentage=0.0
        )
    
    async def _check_quota_alerts(self, quota_usage: QuotaUsage):
        """Check if quota usage triggers alerts."""
        if quota_usage.status in [QuotaStatus.WARNING, QuotaStatus.CRITICAL, QuotaStatus.EXCEEDED]:
            alert_data = {
                "alert_type": "quota_alert",
                "api_name": quota_usage.api_name,
                "quota_type": quota_usage.quota_type.value,
                "status": quota_usage.status.value,
                "usage_percentage": quota_usage.usage_percentage,
                "remaining": quota_usage.remaining,
                "reset_time": quota_usage.reset_time.isoformat()
            }
            
            logger.warning(f"QUOTA ALERT: {json.dumps(alert_data)}")
            
            # In production, send to monitoring systems
    
    def _generate_quota_summary(self, all_status: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of all quota statuses."""
        summary = {
            "total_apis": len(all_status),
            "available": 0,
            "warning": 0,
            "critical": 0,
            "exceeded": 0,
            "errors": 0
        }
        
        for api_status in all_status.values():
            status = api_status.get("overall_status", "error")
            if status in summary:
                summary[status] += 1
            else:
                summary["errors"] += 1
        
        return summary


# Global quota manager instance
quota_manager = QuotaManager()


# Convenience functions
async def check_api_quota(api_name: str, quota_type: QuotaType = QuotaType.REQUESTS_PER_DAY) -> bool:
    """Check if API quota is available."""
    allowed, _ = await quota_manager.check_quota(api_name, quota_type)
    return allowed


async def consume_api_quota(api_name: str, quota_type: QuotaType = QuotaType.REQUESTS_PER_DAY) -> bool:
    """Consume API quota."""
    return await quota_manager.consume_quota(api_name, quota_type)


async def check_user_rate_limit(user_id: str, endpoint: str = "general") -> bool:
    """Check user rate limit."""
    allowed, _ = await quota_manager.check_rate_limit("per_user", f"{user_id}:{endpoint}")
    return allowed


async def consume_user_rate_limit(user_id: str, endpoint: str = "general") -> bool:
    """Consume user rate limit."""
    return await quota_manager.consume_rate_limit("per_user", f"{user_id}:{endpoint}")


## Suggestions for Upgrade:
# 1. Implement dynamic quota adjustment based on API performance and availability
# 2. Add machine learning-based usage prediction and optimization
# 3. Integrate with external quota management services and billing systems
# 4. Add support for quota sharing and pooling across multiple instances