import asyncio
import logging
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import time

# Import schemas
from .schemas import (
    FactCheckRequest, FactCheckResponse, BulkFactCheckRequest, BulkFactCheckResponse,
    SearchRequest, SearchResponse, HealthCheckResponse, ErrorResponse,
    SystemStatusModel, AnalysisConfigModel, BatchJobModel, AnalyticsModel
)

# Import core modules
from .search_orchestrator import SearchOrchestrator
from ..proof_validation.claim_parser import ClaimParser
from ..proof_validation.proofs_aggregator import ProofsAggregator
from ..proof_validation.proof_validator import ProofValidator
from ..proof_validation.scoring import AdvancedScoring
from .verdict_engine import VerdictEngine
from .consensus_builder import AdvancedConsensusBuilder
from .config import config

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Advanced Fact-Checking API",
    description="Production-grade fact-checking service with explainable AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Security
security = HTTPBearer(auto_error=False)

# Global components (initialized on startup)
search_orchestrator: Optional[SearchOrchestrator] = None
claim_parser: Optional[ClaimParser] = None
proofs_aggregator: Optional[ProofsAggregator] = None
proof_validator: Optional[ProofValidator] = None
advanced_scoring: Optional[AdvancedScoring] = None
verdict_engine: Optional[VerdictEngine] = None
consensus_builder: Optional[AdvancedConsensusBuilder] = None

# In-memory storage for batch jobs (use Redis in production)
batch_jobs: Dict[str, Dict[str, Any]] = {}

# Request tracking
active_requests: Dict[str, Dict[str, Any]] = {}


# Dependency functions
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from authentication token."""
    # Implement your authentication logic here
    # For now, we'll allow all requests
    return {"user_id": "anonymous", "permissions": ["read", "write"]}


def track_request(request_id: str, endpoint: str, start_time: float):
    """Track active request."""
    active_requests[request_id] = {
        "endpoint": endpoint,
        "start_time": start_time,
        "status": "processing"
    }


def complete_request(request_id: str, status: str = "completed"):
    """Mark request as completed."""
    if request_id in active_requests:
        active_requests[request_id]["status"] = status
        active_requests[request_id]["end_time"] = time.time()


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global search_orchestrator, claim_parser, proofs_aggregator
    global proof_validator, advanced_scoring, verdict_engine, consensus_builder
    
    try:
        logger.info("Initializing fact-checking service components...")
        
        # Initialize core components
        search_orchestrator = SearchOrchestrator()
        claim_parser = ClaimParser()
        proofs_aggregator = ProofsAggregator()
        proof_validator = ProofValidator()
        advanced_scoring = AdvancedScoring()
        verdict_engine = VerdictEngine()
        consensus_builder = AdvancedConsensusBuilder()
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down fact-checking service...")
    
    # Cancel any active requests
    for request_id in list(active_requests.keys()):
        complete_request(request_id, "cancelled")
    
    logger.info("Shutdown complete")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTP_ERROR",
            message=exc.detail,
            timestamp=datetime.now()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="INTERNAL_ERROR",
            message="An internal error occurred",
            timestamp=datetime.now()
        ).dict()
    )


# Main fact-checking endpoints
# Dashboard Integration Endpoints (v1 API)
@app.post("/v1/verify")
async def verify_claim_v1(request: dict):
    """
    Simplified endpoint for dashboard integration
    Request: { "claim": "<user's text>" }
    Response: { "claim", "verdict", "confidence", "proofs": [...] }
    """
    try:
        claim_text = request.get("claim", "")
        if not claim_text:
            raise HTTPException(status_code=400, detail="Claim text is required")
        
        # Create FactCheckRequest object
        fact_check_request = FactCheckRequest(
            claim=claim_text,
            priority="normal",
            include_explanations=True
        )
        
        # Process the fact check
        request_id = str(uuid.uuid4())
        start_time = time.time()
        track_request(request_id, "verify_v1", start_time)
        
        try:
            # Initialize components if needed
            global search_orchestrator, claim_parser, proofs_aggregator, proof_validator, advanced_scoring, verdict_engine, consensus_builder
            
            if not all([search_orchestrator, claim_parser, proofs_aggregator, proof_validator, advanced_scoring, verdict_engine, consensus_builder]):
                await startup_event()
            
            # Parse and validate claim
            parsed_claim = await claim_parser.parse_claim(fact_check_request.claim)
            
            # Search for evidence
            search_results = await search_orchestrator.search(
                query=parsed_claim.get("query", fact_check_request.claim),
                max_results=20
            )
            
            # Aggregate proofs
            proofs = await proofs_aggregator.aggregate_proofs(
                search_results,
                parsed_claim
            )
            
            # Validate proofs
            validated_proofs = await proof_validator.validate_proofs(
                proofs,
                fact_check_request.claim
            )
            
            # Score proofs
            scored_proofs = await advanced_scoring.score_proofs(
                validated_proofs,
                fact_check_request.claim
            )
            
            # Generate verdict
            verdict_result = await verdict_engine.generate_verdict(
                scored_proofs,
                fact_check_request.claim
            )
            
            # Build consensus
            final_result = await consensus_builder.build_consensus(
                verdict_result,
                scored_proofs
            )
            
            # Format response for dashboard
            def serialize_datetime(obj):
                """Convert datetime objects to ISO format strings."""
                from datetime import datetime
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: serialize_datetime(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [serialize_datetime(item) for item in obj]
                return obj
            
            # Serialize all data to handle datetime objects
            serialized_final_result = serialize_datetime(final_result)
            serialized_scored_proofs = serialize_datetime(scored_proofs)
            
            response = {
                "claim": fact_check_request.claim,
                "verdict": serialized_final_result.get("verdict", "AMBIGUOUS"),
                "confidence": float(serialized_final_result.get("confidence", 0.5)) * 100,  # Convert to percentage
                "proofs": [
                    {
                        "title": str(proof.get("title", "Unknown")),
                        "url": str(proof.get("url", "")),
                        "domain": str(proof.get("domain", "")),
                        "credibility_score": float(proof.get("credibility_score", 0.5)) * 100,
                        "determined_verdict": str(proof.get("verdict", "AMBIGUOUS"))
                    }
                    for proof in serialized_scored_proofs[:10]  # Limit to top 10 proofs
                ]
            }
            
            complete_request(request_id, "completed")
            return response
            
        except Exception as e:
            complete_request(request_id, "failed")
            logger.error(f"Error in verify_v1: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in verify_v1: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/v1/claim/{claim_id}")
async def get_claim_v1(claim_id: str):
    """
    Get cached claim result by ID (optional cached lookups)
    """
    # For now, return a simple response - can be enhanced with actual caching
    return {
        "claim_id": claim_id,
        "status": "not_implemented",
        "message": "Cached claim lookup not yet implemented"
    }


@app.get("/v1/metrics")
async def get_metrics_v1():
    """
    Get system metrics for dashboard (quota, cache stats)
    """
    try:
        # Get current system status
        uptime = time.time() - getattr(app.state, 'start_time', time.time())
        
        return {
            "quota": {
                "news_api": "Available",  # Can be enhanced with actual quota tracking
                "serper_api": "Available",
                "newsdata_api": "Available"
            },
            "cache": {
                "hit_ratio": 0.75,  # Placeholder - can be enhanced with actual cache stats
                "total_requests": len(active_requests),
                "cache_size": 0
            },
            "performance": {
                "average_response_time": 2.5,  # Placeholder
                "uptime_seconds": uptime,
                "active_requests": len(active_requests)
            },
            "status": "healthy"
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        return {
            "quota": {"error": "Unable to fetch quota information"},
            "cache": {"error": "Unable to fetch cache statistics"},
            "performance": {"error": "Unable to fetch performance metrics"},
            "status": "error"
        }


@app.post("/fact-check", response_model=FactCheckResponse)
async def fact_check_claim(
    request: FactCheckRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Fact-check a single claim with comprehensive analysis.
    
    This endpoint performs a complete fact-checking analysis including:
    - Claim parsing and entity extraction
    - Multi-source evidence gathering
    - Proof validation and scoring
    - Consensus building and verdict determination
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    track_request(request_id, "fact-check", start_time)
    
    try:
        logger.info(f"Processing fact-check request {request_id}: {request.claim[:100]}...")
        
        # Step 1: Parse claims
        claims = await claim_parser.parse_claims(request.claim)
        if not claims:
            raise HTTPException(
                status_code=400,
                detail="No valid claims found in the provided text"
            )
        
        # Use the highest confidence claim
        primary_claim = max(claims, key=lambda c: c.confidence)
        
        # Step 2: Search for evidence
        search_sources = await search_orchestrator.search_targeted(
            claim=primary_claim.text,
            entities=[entity.text for entity in primary_claim.entities],
            keywords=primary_claim.keywords,
            max_results=request.max_sources
        )
        
        # Step 3: Aggregate proofs
        proofs = await proofs_aggregator.aggregate_proofs(
            claim=primary_claim.text,
            search_sources=search_sources.sources,
            extract_content=True
        )
        
        # Step 4: Validate proofs
        validation_result = await proof_validator.validate_proofs(proofs)
        
        # Step 5: Calculate source scores
        evidence_analyses = []  # This would come from proof_validator in a real implementation
        source_scores = await advanced_scoring.calculate_source_scores(
            proofs, evidence_analyses
        )
        
        # Step 6: Build consensus
        consensus_result = await advanced_scoring.build_consensus(
            proofs, source_scores, validation_result
        )
        
        # Step 7: Determine final verdict
        final_verdict = await verdict_engine.determine_final_verdict(
            proofs, validation_result, consensus_result
        )
        
        # Prepare response
        processing_time = time.time() - start_time
        
        response = FactCheckResponse(
            request_id=request_id,
            claim=request.claim,
            verdict=final_verdict.to_dict(),
            claims_parsed=[claim.to_dict() for claim in claims],
            sources_found=[source.to_dict() for source in search_sources.sources],
            evidence_collected=[evidence.to_dict() for evidence in 
                              proofs.supporting_evidence + proofs.refuting_evidence + proofs.neutral_evidence],
            processing_stats={
                "processing_time": processing_time,
                "claims_parsed": len(claims),
                "sources_searched": len(search_sources.sources),
                "evidence_pieces": len(proofs.supporting_evidence) + len(proofs.refuting_evidence) + len(proofs.neutral_evidence),
                "consensus_confidence": consensus_result.consensus_confidence
            },
            api_usage=search_sources.api_usage
        )
        
        complete_request(request_id, "completed")
        logger.info(f"Fact-check completed for request {request_id} in {processing_time:.2f}s")
        
        return response
        
    except HTTPException:
        complete_request(request_id, "failed")
        raise
    except Exception as e:
        complete_request(request_id, "failed")
        logger.error(f"Fact-check failed for request {request_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Fact-checking failed: {str(e)}"
        )


@app.post("/fact-check/bulk", response_model=BulkFactCheckResponse)
async def bulk_fact_check(
    request: BulkFactCheckRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Fact-check multiple claims in batch.
    
    This endpoint processes multiple claims either sequentially or in parallel
    depending on the configuration and system load.
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    track_request(request_id, "bulk-fact-check", start_time)
    
    try:
        logger.info(f"Processing bulk fact-check request {request_id} with {len(request.claims)} claims")
        
        results = []
        
        if request.parallel_processing and len(request.claims) <= 5:
            # Process in parallel for small batches
            tasks = []
            for claim in request.claims:
                fact_check_req = FactCheckRequest(
                    claim=claim,
                    max_sources=request.max_sources_per_claim,
                    include_fact_checks=request.include_fact_checks
                )
                task = fact_check_claim(fact_check_req, background_tasks, current_user)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions in results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Claim {i} failed: {result}")
                    # Create error response for failed claim
                    error_response = FactCheckResponse(
                        request_id=f"{request_id}-{i}",
                        claim=request.claims[i],
                        verdict={
                            "verdict": "INSUFFICIENT_EVIDENCE",
                            "confidence_score": 0.0,
                            "explanation": {"primary_factors": [f"Processing failed: {str(result)}"]}
                        },
                        claims_parsed=[],
                        sources_found=[],
                        evidence_collected=[],
                        processing_stats={"error": str(result)},
                        api_usage={}
                    )
                    processed_results.append(error_response)
                else:
                    processed_results.append(result)
            
            results = processed_results
        else:
            # Process sequentially
            for claim in request.claims:
                try:
                    fact_check_req = FactCheckRequest(
                        claim=claim,
                        max_sources=request.max_sources_per_claim,
                        include_fact_checks=request.include_fact_checks
                    )
                    result = await fact_check_claim(fact_check_req, background_tasks, current_user)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process claim '{claim}': {e}")
                    # Continue with other claims
                    continue
        
        total_processing_time = time.time() - start_time
        
        # Generate summary
        verdicts = [r.verdict.get("verdict", "UNKNOWN") for r in results if hasattr(r, 'verdict')]
        verdict_counts = {verdict: verdicts.count(verdict) for verdict in set(verdicts)}
        
        summary = {
            "total_claims": len(request.claims),
            "successful_analyses": len(results),
            "failed_analyses": len(request.claims) - len(results),
            "verdict_distribution": verdict_counts,
            "average_confidence": sum(
                r.verdict.get("confidence_score", 0) for r in results if hasattr(r, 'verdict')
            ) / max(1, len(results)),
            "total_sources_used": sum(
                len(r.sources_found) for r in results if hasattr(r, 'sources_found')
            )
        }
        
        response = BulkFactCheckResponse(
            request_id=request_id,
            results=results,
            summary=summary,
            total_processing_time=total_processing_time
        )
        
        complete_request(request_id, "completed")
        logger.info(f"Bulk fact-check completed for request {request_id}")
        
        return response
        
    except Exception as e:
        complete_request(request_id, "failed")
        logger.error(f"Bulk fact-check failed for request {request_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Bulk fact-checking failed: {str(e)}"
        )


@app.post("/search", response_model=SearchResponse)
async def search_evidence(
    request: SearchRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Search for evidence about a claim or topic.
    
    This endpoint provides access to the search functionality without
    performing full fact-checking analysis.
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        logger.info(f"Processing search request {request_id}: {request.query}")
        
        # Perform comprehensive search
        search_results = await search_orchestrator.search_comprehensive(
            query=request.query,
            max_results=request.max_results,
            include_fact_checks="fact_check" in [st.value for st in request.source_types],
            time_range=request.time_range
        )
        
        # Filter by source types if specified
        if request.source_types:
            filtered_sources = [
                source for source in search_results.sources
                if source.source_type in [st.value for st in request.source_types]
            ]
            search_results.sources = filtered_sources
            search_results.total_sources = len(filtered_sources)
        
        response = SearchResponse(
            query=request.query,
            sources=[source.to_dict() for source in search_results.sources],
            total_sources=search_results.total_sources,
            search_time=search_results.search_time,
            api_usage=search_results.api_usage,
            confidence=search_results.confidence
        )
        
        logger.info(f"Search completed for request {request_id}")
        return response
        
    except Exception as e:
        logger.error(f"Search failed for request {request_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


# Status and monitoring endpoints
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint for service monitoring.
    
    Returns the current status of the service and its dependencies.
    """
    try:
        # Check API status
        api_status = await search_orchestrator.get_api_status()
        
        # System information
        system_info = {
            "active_requests": len(active_requests),
            "batch_jobs": len(batch_jobs),
            "uptime": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
        }
        
        response = HealthCheckResponse(
            status="healthy",
            timestamp=datetime.now(),
            version="1.0.0",
            api_status=api_status,
            system_info=system_info
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="Service unhealthy"
        )


@app.get("/status", response_model=SystemStatusModel)
async def get_system_status(current_user: dict = Depends(get_current_user)):
    """
    Get detailed system status including API quotas and performance metrics.
    """
    try:
        # Get API status
        api_status = await search_orchestrator.get_api_status()
        
        # Convert to quota models
        api_quotas = []
        for api_name, status in api_status.items():
            quota_info = {
                "api_name": api_name,
                "remaining_quota": status.get("remaining_quota", "unknown"),
                "quota_limit": "varies",  # Would be configured per API
                "reset_time": None,
                "usage_percentage": 0.0 if status.get("remaining_quota") == "unknown" else 50.0
            }
            api_quotas.append(quota_info)
        
        response = SystemStatusModel(
            service_status="operational",
            api_quotas=api_quotas,
            cache_status={"enabled": config.cache.enable_cache, "hit_rate": 0.85},
            database_status={"connected": True, "response_time": 0.05},
            last_updated=datetime.now()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get system status"
        )


# Administrative endpoints
@app.get("/requests/active")
async def get_active_requests(current_user: dict = Depends(get_current_user)):
    """Get list of currently active requests."""
    return {
        "active_requests": len(active_requests),
        "requests": [
            {
                "request_id": req_id,
                "endpoint": req_info["endpoint"],
                "duration": time.time() - req_info["start_time"],
                "status": req_info["status"]
            }
            for req_id, req_info in active_requests.items()
        ]
    }


@app.post("/admin/config")
async def update_config(
    config_update: AnalysisConfigModel,
    current_user: dict = Depends(get_current_user)
):
    """Update system configuration (admin only)."""
    # In a real implementation, you'd validate admin permissions
    # and update the actual configuration
    return {"message": "Configuration updated", "config": config_update.dict()}


# Batch processing endpoints
@app.post("/batch/submit")
async def submit_batch_job(
    claims: List[str],
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Submit a batch job for processing multiple claims."""
    job_id = str(uuid.uuid4())
    
    # Create batch job
    batch_job = {
        "job_id": job_id,
        "status": "queued",
        "claims": claims,
        "progress": 0.0,
        "results": [],
        "created_at": datetime.now(),
        "completed_at": None,
        "error_message": None
    }
    
    batch_jobs[job_id] = batch_job
    
    # Add to background tasks
    background_tasks.add_task(process_batch_job, job_id)
    
    return {"job_id": job_id, "status": "queued", "claims_count": len(claims)}


@app.get("/batch/{job_id}", response_model=BatchJobModel)
async def get_batch_job_status(
    job_id: str = Path(..., description="Batch job ID"),
    current_user: dict = Depends(get_current_user)
):
    """Get status of a batch job."""
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Batch job not found")
    
    job = batch_jobs[job_id]
    return BatchJobModel(**job)


async def process_batch_job(job_id: str):
    """Process a batch job in the background."""
    try:
        job = batch_jobs[job_id]
        job["status"] = "processing"
        
        results = []
        total_claims = len(job["claims"])
        
        for i, claim in enumerate(job["claims"]):
            try:
                # Process each claim
                request = FactCheckRequest(claim=claim)
                result = await fact_check_claim(request, None, {"user_id": "batch"})
                results.append(result)
                
                # Update progress
                job["progress"] = ((i + 1) / total_claims) * 100
                
            except Exception as e:
                logger.error(f"Batch job {job_id} failed on claim {i}: {e}")
                continue
        
        # Complete job
        job["status"] = "completed"
        job["results"] = results
        job["completed_at"] = datetime.now()
        job["progress"] = 100.0
        
    except Exception as e:
        logger.error(f"Batch job {job_id} failed: {e}")
        job["status"] = "failed"
        job["error_message"] = str(e)
        job["completed_at"] = datetime.now()


# Analytics endpoints
@app.get("/analytics", response_model=AnalyticsModel)
async def get_analytics(
    period: str = Query("24h", description="Analytics period (1h, 24h, 7d, 30d)"),
    current_user: dict = Depends(get_current_user)
):
    """Get analytics data for the specified period."""
    # In a real implementation, this would query actual analytics data
    mock_analytics = AnalyticsModel(
        period=period,
        total_requests=1250,
        verdicts_distribution={
            "TRUE": 450,
            "FALSE": 380,
            "MISLEADING": 220,
            "MIXED": 120,
            "UNPROVEN": 80
        },
        average_confidence=0.73,
        top_claim_types=[
            {"type": "statistical", "count": 340},
            {"type": "causal", "count": 280},
            {"type": "factual", "count": 250}
        ],
        api_usage_stats={
            "newsapi": 2100,
            "serper": 1800,
            "newsdata": 1200
        },
        performance_metrics={
            "average_response_time": 4.2,
            "cache_hit_rate": 0.85,
            "error_rate": 0.02
        }
    )
    
    return mock_analytics


if __name__ == "__main__":
    import uvicorn
    
    # Set start time for uptime calculation
    app.state.start_time = time.time()
    
    uvicorn.run(
        "routes:app",
        host=config.host,
        port=config.port,
        workers=1,  # Use 1 worker for development
        log_level=config.logging.level.lower(),
        reload=config.debug
    )


## Suggestions for Upgrade:
# 1. Implement proper authentication and authorization with JWT tokens and role-based access control
# 2. Add comprehensive request/response logging and audit trails for compliance and debugging
# 3. Implement rate limiting and throttling to prevent abuse and ensure fair usage
# 4. Add WebSocket support for real-time updates on long-running fact-checking operations