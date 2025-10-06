#!/usr/bin/env python3
"""
Serper API Integration for Proof Validation
Demonstrates how Serper API can enhance the proof validation system
by providing real-time web search capabilities for fact-checking.
"""

import os
import requests
import json
from dotenv import load_dotenv
from typing import Dict, List, Optional
from datetime import datetime

# Load environment variables
load_dotenv()

class SerperProofValidator:
    """Enhanced proof validator using Serper API for real-time fact checking"""
    
    def __init__(self):
        self.api_key = os.getenv('SERPER_API_KEY')
        self.base_url = "https://google.serper.dev/search"
        
        if not self.api_key:
            raise ValueError("SERPER_API_KEY not found in environment variables")
    
    def search_claim_evidence(self, claim: str, num_results: int = 5) -> Optional[Dict]:
        """Search for evidence related to a specific claim"""
        
        # Enhance the search query for better fact-checking results
        enhanced_query = f'"{claim}" fact check verification evidence'
        
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        
        payload = {
            'q': enhanced_query,
            'num': num_results,
            'type': 'search'
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=15
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ API Error: HTTP {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            return None
    
    def analyze_source_credibility(self, url: str) -> str:
        """Analyze the credibility of a source URL"""
        
        # Simple credibility scoring based on domain
        trusted_domains = [
            'reuters.com', 'bbc.com', 'ap.org', 'npr.org',
            'factcheck.org', 'snopes.com', 'politifact.com',
            'cnn.com', 'nytimes.com', 'washingtonpost.com'
        ]
        
        domain = url.split('/')[2] if len(url.split('/')) > 2 else url
        
        if any(trusted in domain.lower() for trusted in trusted_domains):
            return "HIGH"
        elif domain.endswith('.edu') or domain.endswith('.gov'):
            return "HIGH"
        elif domain.endswith('.org'):
            return "MEDIUM"
        else:
            return "LOW"
    
    def validate_claim_with_evidence(self, claim: str) -> Dict:
        """Validate a claim by searching for supporting/contradicting evidence"""
        
        print(f"\n🔍 VALIDATING CLAIM: '{claim}'")
        print("="*80)
        
        # Search for evidence
        search_results = self.search_claim_evidence(claim)
        
        if not search_results:
            return {
                'claim': claim,
                'status': 'ERROR',
                'evidence_count': 0,
                'credibility_score': 0,
                'sources': []
            }
        
        evidence_sources = []
        total_credibility = 0
        
        if 'organic' in search_results:
            for result in search_results['organic']:
                source_url = result.get('link', '')
                credibility = self.analyze_source_credibility(source_url)
                
                evidence_sources.append({
                    'title': result.get('title', 'No Title'),
                    'url': source_url,
                    'snippet': result.get('snippet', 'No snippet'),
                    'credibility': credibility,
                    'date': result.get('date', 'Unknown')
                })
                
                # Add to credibility score
                if credibility == 'HIGH':
                    total_credibility += 3
                elif credibility == 'MEDIUM':
                    total_credibility += 2
                else:
                    total_credibility += 1
        
        # Calculate overall credibility score (0-100)
        max_possible_score = len(evidence_sources) * 3
        credibility_score = (total_credibility / max_possible_score * 100) if max_possible_score > 0 else 0
        
        # Determine validation status
        if credibility_score >= 70:
            status = 'VERIFIED'
        elif credibility_score >= 40:
            status = 'PARTIALLY_VERIFIED'
        else:
            status = 'UNVERIFIED'
        
        return {
            'claim': claim,
            'status': status,
            'evidence_count': len(evidence_sources),
            'credibility_score': round(credibility_score, 2),
            'sources': evidence_sources,
            'timestamp': datetime.now().isoformat()
        }
    
    def display_validation_results(self, results: Dict):
        """Display validation results in a formatted way"""
        
        print(f"\n📊 VALIDATION RESULTS")
        print(f"Claim: {results['claim']}")
        print(f"Status: {results['status']}")
        print(f"Credibility Score: {results['credibility_score']}/100")
        print(f"Evidence Sources Found: {results['evidence_count']}")
        print(f"Validation Time: {results['timestamp']}")
        
        print(f"\n📚 EVIDENCE SOURCES:")
        for i, source in enumerate(results['sources'], 1):
            print(f"\n[{i}] {source['title']}")
            print(f"    🔗 URL: {source['url']}")
            print(f"    📝 Snippet: {source['snippet'][:100]}...")
            print(f"    🏆 Credibility: {source['credibility']}")
            print(f"    📅 Date: {source['date']}")
        
        print("\n" + "="*80)

def main():
    """Demonstrate Serper API integration with proof validation"""
    
    print("🚀 SERPER API PROOF VALIDATION DEMO")
    print("="*80)
    
    try:
        validator = SerperProofValidator()
        print(f"✅ Serper API initialized successfully")
        
        # Test claims for validation
        test_claims = [
            "COVID-19 vaccines are effective in preventing severe illness",
            "Climate change is caused by human activities",
            "The Earth is flat"
        ]
        
        validation_results = []
        
        for claim in test_claims:
            result = validator.validate_claim_with_evidence(claim)
            validation_results.append(result)
            validator.display_validation_results(result)
            
            # Small delay between requests
            import time
            time.sleep(2)
        
        # Summary
        print(f"\n🎯 VALIDATION SUMMARY")
        print("="*80)
        
        for result in validation_results:
            status_emoji = {
                'VERIFIED': '✅',
                'PARTIALLY_VERIFIED': '⚠️',
                'UNVERIFIED': '❌',
                'ERROR': '🚫'
            }.get(result['status'], '❓')
            
            print(f"{status_emoji} {result['status']} - Score: {result['credibility_score']}/100")
            print(f"   Claim: {result['claim'][:60]}...")
        
        print(f"\n🎉 Proof validation demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()