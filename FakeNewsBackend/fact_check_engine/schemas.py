from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator, HttpUrl
from pydantic.types import constr, confloat, conint

# Enums for validation
class VerdictEnum(str, Enum):
    """Enumeration of possible verdicts."""
    TRUE = "TRUE"
    FALSE = "FALSE"
    MISLEADING = "MISLEADING"
    UNPROVEN = "UNPROVEN"
    MIXED = "MIXED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class ConfidenceLevelEnum(str, Enum):
    """Enumeration of confidence levels."""
    VERY_HIGH = "VERY_HIGH"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    VERY_LOW = "VERY_LOW"


class EvidenceTypeEnum(str, Enum):
    """Enumeration of evidence types."""
    SUPPORTING = "supporting"
    REFUTING = "refuting"
    NEUTRAL = "neutral"
    UNCLEAR = "unclear"


class SourceTypeEnum(str, Enum):
    """Enumeration of source types."""
    NEWS = "news"
    WEB = "web"
    FACT_CHECK = "fact_check"
    SOCIAL = "social"


# Request Models
class FactCheckRequest(BaseModel):
    """Request model for fact-checking a claim."""
    claim: constr(min_length=10, max_length=1000) = Field(
        ...,
        description="The claim to be fact-checked",
        example="The COVID-19 vaccine contains microchips for tracking people."
    )
    max_sources: conint(ge=5, le=100) = Field(
        default=50,
        description="Maximum number of sources to search"
    )
    include_fact_checks: bool = Field(
        default=True,
        description="Whether to include dedicated fact-checking sources"
    )
    time_range: Optional[str] = Field(
        default=None,
        description="Time range for search (h, d, w, m, y)",
        pattern=r"^[hdwmy]$"
    )
    languages: List[str] = Field(
        default=["en"],
        description="Languages to search in",
        max_items=5
    )
    priority_sources: Optional[List[str]] = Field(
        default=None,
        description="Priority source domains",
        max_items=10
    )
    
    @validator('languages')
    def validate_languages(cls, v):
        """Validate language codes."""
        valid_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ar']
        for lang in v:
            if lang not in valid_languages:
                raise ValueError(f"Unsupported language: {lang}")
        return v


class BulkFactCheckRequest(BaseModel):
    """Request model for bulk fact-checking multiple claims."""
    claims: List[constr(min_length=10, max_length=1000)] = Field(
        ...,
        description="List of claims to be fact-checked",
        min_items=1,
        max_items=10
    )
    max_sources_per_claim: conint(ge=5, le=50) = Field(
        default=25,
        description="Maximum sources per claim"
    )
    include_fact_checks: bool = Field(
        default=True,
        description="Whether to include fact-checking sources"
    )
    parallel_processing: bool = Field(
        default=True,
        description="Whether to process claims in parallel"
    )


class SearchRequest(BaseModel):
    """Request model for searching evidence about a claim."""
    query: constr(min_length=3, max_length=500) = Field(
        ...,
        description="Search query",
        example="COVID-19 vaccine safety"
    )
    max_results: conint(ge=1, le=100) = Field(
        default=20,
        description="Maximum number of results"
    )
    source_types: List[SourceTypeEnum] = Field(
        default=[SourceTypeEnum.NEWS, SourceTypeEnum.FACT_CHECK],
        description="Types of sources to search"
    )
    time_range: Optional[str] = Field(
        default=None,
        description="Time range filter"
    )


# Response Models
class EntityModel(BaseModel):
    """Model for named entities."""
    text: str = Field(..., description="Entity text")
    label: str = Field(..., description="Entity label/type")
    start: int = Field(..., description="Start position in text")
    end: int = Field(..., description="End position in text")
    confidence: confloat(ge=0.0, le=1.0) = Field(..., description="Entity confidence")


class ClaimModel(BaseModel):
    """Model for parsed claims."""
    text: str = Field(..., description="Claim text")
    confidence: confloat(ge=0.0, le=1.0) = Field(..., description="Claim confidence")
    entities: List[EntityModel] = Field(default=[], description="Named entities")
    claim_type: str = Field(..., description="Type of claim")
    keywords: List[str] = Field(default=[], description="Important keywords")
    position: List[int] = Field(..., description="Position in source text [start, end]")


class SourceModel(BaseModel):
    """Model for search sources."""
    title: str = Field(..., description="Source title")
    url: HttpUrl = Field(..., description="Source URL")
    snippet: str = Field(..., description="Source snippet/description")
    source_name: str = Field(..., description="Source name")
    published_date: Optional[datetime] = Field(None, description="Publication date")
    source_type: SourceTypeEnum = Field(..., description="Type of source")
    relevance_score: confloat(ge=0.0, le=1.0) = Field(..., description="Relevance score")
    credibility_score: confloat(ge=0.0, le=1.0) = Field(..., description="Credibility score")
    api_source: str = Field(..., description="API source name")


class EvidenceModel(BaseModel):
    """Model for evidence pieces."""
    content: str = Field(..., description="Evidence content")
    source_url: HttpUrl = Field(..., description="Source URL")
    source_name: str = Field(..., description="Source name")
    evidence_type: EvidenceTypeEnum = Field(..., description="Type of evidence")
    confidence: confloat(ge=0.0, le=1.0) = Field(..., description="Evidence confidence")
    relevance_score: confloat(ge=0.0, le=1.0) = Field(..., description="Relevance score")
    credibility_score: confloat(ge=0.0, le=1.0) = Field(..., description="Credibility score")
    extracted_at: datetime = Field(..., description="Extraction timestamp")
    context: Optional[str] = Field(None, description="Additional context")
    quotes: List[str] = Field(default=[], description="Relevant quotes")


class ValidationResultModel(BaseModel):
    """Model for validation results."""
    claim: str = Field(..., description="Original claim")
    verdict: str = Field(..., description="Validation verdict")
    confidence: confloat(ge=0.0, le=1.0) = Field(..., description="Validation confidence")
    evidence_score: confloat(ge=0.0, le=1.0) = Field(..., description="Evidence quality score")
    credibility_score: confloat(ge=0.0, le=1.0) = Field(..., description="Source credibility score")
    consistency_score: confloat(ge=0.0, le=1.0) = Field(..., description="Evidence consistency score")
    bias_score: confloat(ge=0.0, le=1.0) = Field(..., description="Bias detection score")
    temporal_score: confloat(ge=0.0, le=1.0) = Field(..., description="Temporal relevance score")
    supporting_strength: confloat(ge=0.0, le=1.0) = Field(..., description="Supporting evidence strength")
    refuting_strength: confloat(ge=0.0, le=1.0) = Field(..., description="Refuting evidence strength")
    validated_at: datetime = Field(..., description="Validation timestamp")


class VerdictExplanationModel(BaseModel):
    """Model for verdict explanations."""
    primary_factors: List[str] = Field(..., description="Primary factors in decision")
    supporting_evidence_summary: str = Field(..., description="Supporting evidence summary")
    refuting_evidence_summary: str = Field(..., description="Refuting evidence summary")
    key_sources: List[str] = Field(..., description="Key sources used")
    limitations: List[str] = Field(..., description="Analysis limitations")
    methodology_notes: List[str] = Field(..., description="Methodology notes")


class FinalVerdictModel(BaseModel):
    """Model for final verdicts."""
    claim: str = Field(..., description="Original claim")
    verdict: VerdictEnum = Field(..., description="Final verdict")
    confidence_score: confloat(ge=0.0, le=1.0) = Field(..., description="Confidence score")
    confidence_level: ConfidenceLevelEnum = Field(..., description="Confidence level")
    explanation: VerdictExplanationModel = Field(..., description="Detailed explanation")
    evidence_summary: Dict[str, Any] = Field(..., description="Evidence summary")
    source_analysis: Dict[str, Any] = Field(..., description="Source analysis")
    methodology_details: Dict[str, Any] = Field(..., description="Methodology details")
    quality_indicators: Dict[str, Any] = Field(..., description="Quality indicators")
    timestamp: datetime = Field(..., description="Analysis timestamp")
    processing_time: float = Field(..., description="Processing time in seconds")


class FactCheckResponse(BaseModel):
    """Response model for fact-checking requests."""
    request_id: str = Field(..., description="Unique request identifier")
    claim: str = Field(..., description="Original claim")
    verdict: FinalVerdictModel = Field(..., description="Final verdict and analysis")
    claims_parsed: List[ClaimModel] = Field(..., description="Parsed claims")
    sources_found: List[SourceModel] = Field(..., description="Sources found")
    evidence_collected: List[EvidenceModel] = Field(..., description="Evidence collected")
    processing_stats: Dict[str, Any] = Field(..., description="Processing statistics")
    api_usage: Dict[str, int] = Field(..., description="API usage statistics")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BulkFactCheckResponse(BaseModel):
    """Response model for bulk fact-checking requests."""
    request_id: str = Field(..., description="Unique request identifier")
    results: List[FactCheckResponse] = Field(..., description="Individual fact-check results")
    summary: Dict[str, Any] = Field(..., description="Bulk processing summary")
    total_processing_time: float = Field(..., description="Total processing time")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SearchResponse(BaseModel):
    """Response model for search requests."""
    query: str = Field(..., description="Original search query")
    sources: List[SourceModel] = Field(..., description="Search results")
    total_sources: int = Field(..., description="Total sources found")
    search_time: float = Field(..., description="Search time in seconds")
    api_usage: Dict[str, int] = Field(..., description="API usage statistics")
    confidence: confloat(ge=0.0, le=1.0) = Field(..., description="Search confidence")


class HealthCheckResponse(BaseModel):
    """Response model for health checks."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Service version")
    api_status: Dict[str, Dict[str, Any]] = Field(..., description="API status details")
    system_info: Dict[str, Any] = Field(..., description="System information")


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Status and Monitoring Models
class APIQuotaModel(BaseModel):
    """Model for API quota information."""
    api_name: str = Field(..., description="API name")
    remaining_quota: Union[int, str] = Field(..., description="Remaining quota")
    quota_limit: Union[int, str] = Field(..., description="Quota limit")
    reset_time: Optional[datetime] = Field(None, description="Quota reset time")
    usage_percentage: confloat(ge=0.0, le=100.0) = Field(..., description="Usage percentage")


class SystemStatusModel(BaseModel):
    """Model for system status."""
    service_status: str = Field(..., description="Overall service status")
    api_quotas: List[APIQuotaModel] = Field(..., description="API quota status")
    cache_status: Dict[str, Any] = Field(..., description="Cache status")
    database_status: Dict[str, Any] = Field(..., description="Database status")
    last_updated: datetime = Field(..., description="Last status update")


# Configuration Models
class AnalysisConfigModel(BaseModel):
    """Model for analysis configuration."""
    max_sources: conint(ge=1, le=200) = Field(default=50, description="Maximum sources")
    include_fact_checks: bool = Field(default=True, description="Include fact-checking sources")
    confidence_threshold: confloat(ge=0.0, le=1.0) = Field(default=0.6, description="Confidence threshold")
    enable_caching: bool = Field(default=True, description="Enable result caching")
    cache_ttl: conint(ge=300, le=86400) = Field(default=3600, description="Cache TTL in seconds")
    parallel_processing: bool = Field(default=True, description="Enable parallel processing")


# Webhook Models
class WebhookConfigModel(BaseModel):
    """Model for webhook configuration."""
    url: HttpUrl = Field(..., description="Webhook URL")
    events: List[str] = Field(..., description="Events to subscribe to")
    secret: Optional[str] = Field(None, description="Webhook secret")
    active: bool = Field(default=True, description="Whether webhook is active")


class WebhookEventModel(BaseModel):
    """Model for webhook events."""
    event_type: str = Field(..., description="Event type")
    timestamp: datetime = Field(..., description="Event timestamp")
    data: Dict[str, Any] = Field(..., description="Event data")
    request_id: str = Field(..., description="Associated request ID")


# Batch Processing Models
class BatchJobModel(BaseModel):
    """Model for batch processing jobs."""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status")
    claims: List[str] = Field(..., description="Claims to process")
    progress: confloat(ge=0.0, le=100.0) = Field(..., description="Job progress percentage")
    results: Optional[List[FactCheckResponse]] = Field(None, description="Job results")
    created_at: datetime = Field(..., description="Job creation time")
    completed_at: Optional[datetime] = Field(None, description="Job completion time")
    error_message: Optional[str] = Field(None, description="Error message if failed")


# Analytics Models
class AnalyticsModel(BaseModel):
    """Model for analytics data."""
    period: str = Field(..., description="Analytics period")
    total_requests: int = Field(..., description="Total requests")
    verdicts_distribution: Dict[str, int] = Field(..., description="Verdict distribution")
    average_confidence: float = Field(..., description="Average confidence score")
    top_claim_types: List[Dict[str, Any]] = Field(..., description="Top claim types")
    api_usage_stats: Dict[str, int] = Field(..., description="API usage statistics")
    performance_metrics: Dict[str, float] = Field(..., description="Performance metrics")


## Suggestions for Upgrade:
# 1. Add support for custom validation rules and business logic in Pydantic models
# 2. Implement automatic API documentation generation with rich examples and use cases
# 3. Add support for versioned schemas to handle API evolution and backward compatibility
# 4. Integrate with OpenAPI 3.0 specification for enhanced API documentation and client generation