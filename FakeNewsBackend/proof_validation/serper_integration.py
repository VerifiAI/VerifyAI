import os
import requests
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class SerperEvidence:
    """Evidence found through Serper API search"""
    title: str
    url: str
    snippet: str
    credibility_level: str
    publication_date: Optional[str]
    relevance_score: float
    source_type: str  # 'news', 'academic', 'government', 'fact_check', 'other'

@dataclass
class SerperValidationResult:
    """Result of claim validation using Serper API"""
    claim: str
    validation_status: str  # 'VERIFIED', 'PARTIALLY_VERIFIED', 'UNVERIFIED', 'CONTRADICTED'
    confidence_score: float
    evidence_count: int
    supporting_evidence: List[SerperEvidence]
    contradicting_evidence: List[SerperEvidence]
    neutral_evidence: List[SerperEvidence]
    search_timestamp: str
    api_response_time: float

class SerperAPIClient:
    """Client for Serper API integration with proof validation system"""
    
    def __init__(self):
        self.api_key = os.getenv('SERPER_API_KEY')
        self.base_url = "https://google.serper.dev/search"
        self.news_url = "https://google.serper.dev/news"
        
        if not self.api_key:
            raise ValueError("SERPER_API_KEY not found in environment variables")
    
    def _classify_source_type(self, url: str) -> str:
        """Classify the type of source based on URL"""
        domain = url.split('/')[2].lower() if len(url.split('/')) > 2 else url.lower()
        
        if any(fact_check in domain for fact_check in ['factcheck', 'snopes', 'politifact', 'fullfact']):
            return 'fact_check'
        elif domain.endswith('.edu') or 'scholar' in domain or 'research' in domain:
            return 'academic'
        elif domain.endswith('.gov') or 'government' in domain:
            return 'government'
        elif any(news in domain for news in ['reuters', 'bbc', 'cnn', 'nytimes', 'washingtonpost', 'ap.org', 'npr']):
            return 'news'
        else:
            return 'other'
    
    def _calculate_credibility_level(self, url: str, source_type: str) -> str:
        """Calculate credibility level based on source characteristics"""
        domain = url.split('/')[2].lower() if len(url.split('/')) > 2 else url.lower()
        
        # High credibility sources
        high_credibility = [
            'reuters.com', 'bbc.com', 'ap.org', 'npr.org',
            'factcheck.org', 'snopes.com', 'politifact.com',
            'nature.com', 'science.org', 'nejm.org'
        ]
        
        if any(trusted in domain for trusted in high_credibility):
            return 'HIGH'
        elif source_type in ['academic', 'government', 'fact_check']:
            return 'HIGH'
        elif source_type == 'news':
            return 'MEDIUM'
        elif domain.endswith('.org'):
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _calculate_relevance_score(self, snippet: str, claim: str) -> float:
        """Calculate relevance score based on snippet content and claim"""
        claim_words = set(claim.lower().split())
        snippet_words = set(snippet.lower().split())
        
        # Simple relevance calculation based on word overlap
        overlap = len(claim_words.intersection(snippet_words))
        total_claim_words = len(claim_words)
        
        return min(overlap / total_claim_words, 1.0) if total_claim_words > 0 else 0.0
    
    async def search_claim_evidence_async(self, claim: str, search_type: str = 'search') -> Optional[Dict]:
        """Asynchronously search for evidence related to a claim"""
        
        url = self.news_url if search_type == 'news' else self.base_url
        
        # Enhanced query for better fact-checking results
        enhanced_query = f'"{claim}" fact check verification evidence truth'
        
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        
        payload = {
            'q': enhanced_query,
            'num': 10,
            'type': search_type
        }
        
        start_time = datetime.now()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=15) as response:
                    if response.status == 200:
                        result = await response.json()
                        end_time = datetime.now()
                        result['_response_time'] = (end_time - start_time).total_seconds()
                        return result
                    else:
                        return None
        except Exception as e:
            print(f"Async search failed: {e}")
            return None
    
    def search_claim_evidence(self, claim: str, search_type: str = 'search') -> Optional[Dict]:
        """Search for evidence related to a claim (synchronous version)"""
        return asyncio.run(self.search_claim_evidence_async(claim, search_type))
    
    def process_search_results(self, results: Dict, claim: str) -> List[SerperEvidence]:
        """Process search results into SerperEvidence objects"""
        evidence_list = []
        
        if 'organic' in results:
            for result in results['organic']:
                url = result.get('link', '')
                snippet = result.get('snippet', '')
                source_type = self._classify_source_type(url)
                credibility = self._calculate_credibility_level(url, source_type)
                relevance = self._calculate_relevance_score(snippet, claim)
                
                evidence = SerperEvidence(
                    title=result.get('title', 'No Title'),
                    url=url,
                    snippet=snippet,
                    credibility_level=credibility,
                    publication_date=result.get('date'),
                    relevance_score=relevance,
                    source_type=source_type
                )
                
                evidence_list.append(evidence)
        
        # Sort by relevance score and credibility
        evidence_list.sort(key=lambda x: (x.relevance_score, x.credibility_level == 'HIGH'), reverse=True)
        
        return evidence_list
    
    def validate_claim_comprehensive(self, claim: str) -> SerperValidationResult:
        """Perform comprehensive claim validation using Serper API"""
        
        start_time = datetime.now()
        
        # Search for general evidence
        general_results = self.search_claim_evidence(claim, 'search')
        news_results = self.search_claim_evidence(claim, 'news')
        
        all_evidence = []
        response_time = 0
        
        if general_results:
            all_evidence.extend(self.process_search_results(general_results, claim))
            response_time += general_results.get('_response_time', 0)
        
        if news_results:
            all_evidence.extend(self.process_search_results(news_results, claim))
            response_time += news_results.get('_response_time', 0)
        
        # Remove duplicates based on URL
        unique_evidence = {}
        for evidence in all_evidence:
            if evidence.url not in unique_evidence:
                unique_evidence[evidence.url] = evidence
        
        all_evidence = list(unique_evidence.values())
        
        # Classify evidence as supporting, contradicting, or neutral
        supporting_evidence = []
        contradicting_evidence = []
        neutral_evidence = []
        
        # Simple classification based on keywords (can be enhanced with NLP)
        supporting_keywords = ['true', 'confirmed', 'verified', 'accurate', 'correct', 'factual']
        contradicting_keywords = ['false', 'debunked', 'myth', 'incorrect', 'wrong', 'misleading']
        
        for evidence in all_evidence:
            snippet_lower = evidence.snippet.lower()
            
            if any(keyword in snippet_lower for keyword in contradicting_keywords):
                contradicting_evidence.append(evidence)
            elif any(keyword in snippet_lower for keyword in supporting_keywords):
                supporting_evidence.append(evidence)
            else:
                neutral_evidence.append(evidence)
        
        # Calculate confidence score
        total_evidence = len(all_evidence)
        high_credibility_count = sum(1 for e in all_evidence if e.credibility_level == 'HIGH')
        supporting_count = len(supporting_evidence)
        contradicting_count = len(contradicting_evidence)
        
        if total_evidence == 0:
            confidence_score = 0.0
            validation_status = 'UNVERIFIED'
        else:
            # Weighted confidence calculation
            credibility_weight = high_credibility_count / total_evidence
            evidence_balance = (supporting_count - contradicting_count) / total_evidence
            
            confidence_score = (credibility_weight * 0.6 + (evidence_balance + 1) / 2 * 0.4) * 100
            
            if confidence_score >= 75 and supporting_count > contradicting_count:
                validation_status = 'VERIFIED'
            elif confidence_score >= 50:
                validation_status = 'PARTIALLY_VERIFIED'
            elif contradicting_count > supporting_count and confidence_score >= 30:
                validation_status = 'CONTRADICTED'
            else:
                validation_status = 'UNVERIFIED'
        
        return SerperValidationResult(
            claim=claim,
            validation_status=validation_status,
            confidence_score=round(confidence_score, 2),
            evidence_count=total_evidence,
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
            neutral_evidence=neutral_evidence,
            search_timestamp=start_time.isoformat(),
            api_response_time=round(response_time, 3)
        )

# Integration function for existing proof validation system
def integrate_serper_validation(claim: str) -> Dict:
    """Integration function that can be called from existing proof validation modules"""
    
    try:
        client = SerperAPIClient()
        result = client.validate_claim_comprehensive(claim)
        
        # Convert to format compatible with existing system
        return {
            'validation_method': 'serper_api',
            'claim': result.claim,
            'status': result.validation_status,
            'confidence': result.confidence_score,
            'evidence_sources': len(result.supporting_evidence + result.contradicting_evidence + result.neutral_evidence),
            'supporting_sources': len(result.supporting_evidence),
            'contradicting_sources': len(result.contradicting_evidence),
            'api_response_time': result.api_response_time,
            'timestamp': result.search_timestamp,
            'detailed_evidence': {
                'supporting': [{
                    'title': e.title,
                    'url': e.url,
                    'credibility': e.credibility_level,
                    'relevance': e.relevance_score
                } for e in result.supporting_evidence],
                'contradicting': [{
                    'title': e.title,
                    'url': e.url,
                    'credibility': e.credibility_level,
                    'relevance': e.relevance_score
                } for e in result.contradicting_evidence]
            }
        }
    
    except Exception as e:
        return {
            'validation_method': 'serper_api',
            'status': 'ERROR',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    # Test the integration
    test_claim = "Artificial intelligence can help detect fake news"
    result = integrate_serper_validation(test_claim)
    
    print("🔍 SERPER API INTEGRATION TEST")
    print("="*50)
    print(f"Claim: {test_claim}")
    print(f"Status: {result.get('status', 'Unknown')}")
    print(f"Confidence: {result.get('confidence', 0)}%")
    print(f"Evidence Sources: {result.get('evidence_sources', 0)}")
    print(f"API Response Time: {result.get('api_response_time', 0)}s")
    print("✅ Integration test completed!")