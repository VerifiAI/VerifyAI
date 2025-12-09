#!/usr/bin/env python3
"""
Cross-Verification API Endpoint
Provides REST API for cross-verification functionality
"""

from flask import Blueprint, request, jsonify
import logging
import traceback
from cross_verification_engine import CrossVerificationEngine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Blueprint
cross_verify_bp = Blueprint('cross_verify', __name__)

# Initialize cross-verification engine
verification_engine = CrossVerificationEngine()

@cross_verify_bp.route('/api/cross-verify', methods=['POST'])
def cross_verify_content():
    """
    Cross-verify content using multiple sources and evidence analysis
    
    Expected JSON payload:
    {
        "content": "Content to verify",
        "perform_serper_analysis": true/false
    }
    
    Returns:
    {
        "success": true/false,
        "verification_result": {...},
        "error": "Error message if any"
    }
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        content = data.get('content', '').strip()
        perform_serper = data.get('perform_serper_analysis', True)
        
        if not content:
            return jsonify({
                'success': False,
                'error': 'Content is required for verification'
            }), 400
        
        logger.info(f"Cross-verification request for content: {content[:100]}...")
        
        # For now, simulate SerperAPI analysis with demo data
        # In production, this would call actual SerperAPI
        simulated_serper_report = generate_simulated_serper_report(content)
        
        # Perform cross-verification
        verification_result = verification_engine.cross_verify_content(simulated_serper_report)
        
        # Format result for frontend
        formatted_result = format_verification_result(verification_result, content)
        
        logger.info(f"Cross-verification completed. Verdict: {verification_result['final_verdict']['verdict']}")
        
        return jsonify({
            'success': True,
            'verification_result': formatted_result
        })
        
    except Exception as e:
        logger.error(f"Cross-verification error: {str(e)}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': f'Verification failed: {str(e)}'
        }), 500

def generate_simulated_serper_report(content: str) -> dict:
    """
    Generate simulated SerperAPI report for testing
    In production, this would call actual SerperAPI
    """
    
    # Simulate the user's scenario: 0 supporting, 25 contradicting, 25 neutral
    supporting_evidence = []
    
    contradicting_evidence = [
        {
            'title': f'Fact-Check Analysis {i+1}: Content Disputed',
            'snippet': f'Our investigation reveals that this claim lacks credible evidence and contradicts established facts from reliable sources.',
            'link': f'https://factcheck-site-{i+1}.com/analysis',
            'source': f'FactCheck Site {i+1}',
            'date': '2024-01-15'
        }
        for i in range(25)
    ]
    
    neutral_evidence = [
        {
            'title': f'Research Report {i+1}: Neutral Analysis',
            'snippet': f'This research provides contextual information without taking a definitive stance on the claim.',
            'link': f'https://research-institute-{i+1}.org/report',
            'source': f'Research Institute {i+1}',
            'date': '2024-01-10'
        }
        for i in range(25)
    ]
    
    return {
        'query': content[:100],
        'analysis': {
            'supportingEvidence': supporting_evidence,
            'contradictingEvidence': contradicting_evidence,
            'neutralEvidence': neutral_evidence
        },
        'metadata': {
            'totalSources': 50,
            'analysisDate': '2024-01-20',
            'confidence': 0.10
        }
    }

def format_verification_result(verification_result: dict, original_content: str) -> dict:
    """
    Format verification result for frontend consumption
    """
    
    final_verdict = verification_result['final_verdict']
    evidence_analysis = verification_result['evidence_analysis']
    
    # Map verdict to frontend format
    verdict_mapping = {
        'FAKE': 'FAKE',
        'LIKELY FAKE': 'FAKE',
        'REAL': 'REAL',
        'LIKELY REAL': 'REAL',
        'INCONCLUSIVE (LEAN FAKE)': 'FAKE',
        'UNKNOWN': 'UNKNOWN'
    }
    
    mapped_verdict = verdict_mapping.get(final_verdict['verdict'], 'UNKNOWN')
    
    # Calculate confidence percentage
    confidence_percentage = int(final_verdict['confidence'] * 100)
    
    # Create evidence breakdown for frontend
    evidence_breakdown = []
    
    # Add contradicting evidence summary
    if evidence_analysis['contradicting']['count'] > 0:
        evidence_breakdown.append({
            'type': 'Contradicting',
            'source': f"{evidence_analysis['contradicting']['count']} Fact-Checking Sources",
            'credibility': 'High' if evidence_analysis['contradicting']['average_credibility'] > 0.7 else 'Medium',
            'summary': f"Cross-verification found {evidence_analysis['contradicting']['count']} sources that contradict the claim with an average credibility of {evidence_analysis['contradicting']['average_credibility']:.1%}."
        })
    
    # Add supporting evidence summary
    if evidence_analysis['supporting']['count'] > 0:
        evidence_breakdown.append({
            'type': 'Supporting',
            'source': f"{evidence_analysis['supporting']['count']} Supporting Sources",
            'credibility': 'High' if evidence_analysis['supporting']['average_credibility'] > 0.7 else 'Medium',
            'summary': f"Found {evidence_analysis['supporting']['count']} sources that support the claim with an average credibility of {evidence_analysis['supporting']['average_credibility']:.1%}."
        })
    
    # Add neutral evidence summary
    if evidence_analysis['neutral']['count'] > 0:
        evidence_breakdown.append({
            'type': 'Neutral',
            'source': f"{evidence_analysis['neutral']['count']} Neutral Sources",
            'credibility': 'Medium',
            'summary': f"Found {evidence_analysis['neutral']['count']} neutral sources that provide context without taking a stance."
        })
    
    # Add analysis summary
    evidence_breakdown.append({
        'type': 'Analysis',
        'source': 'Cross-Verification Engine',
        'credibility': 'High',
        'summary': final_verdict['reasoning']
    })
    
    return {
        'verification': {
            'isVerified': mapped_verdict == 'REAL',
            'confidence': final_verdict['confidence'],
            'claim': original_content[:100] + ('...' if len(original_content) > 100 else '')
        },
        'analysis': {
            'supportingEvidence': [],  # Simplified for demo
            'contradictingEvidence': [],  # Simplified for demo
            'neutralEvidence': []  # Simplified for demo
        },
        'crossVerification': {
            'verdict': mapped_verdict,
            'confidence': confidence_percentage,
            'reasoning': final_verdict['reasoning'],
            'evidenceBreakdown': evidence_breakdown,
            'analysisDetails': {
                'supportingSources': evidence_analysis['supporting']['count'],
                'contradictingSources': evidence_analysis['contradicting']['count'],
                'neutralSources': evidence_analysis['neutral']['count'],
                'totalSources': (
                    evidence_analysis['supporting']['count'] +
                    evidence_analysis['contradicting']['count'] +
                    evidence_analysis['neutral']['count']
                )
            }
        }
    }

# Health check endpoint
@cross_verify_bp.route('/api/cross-verify/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for the cross-verification service
    """
    return jsonify({
        'status': 'healthy',
        'service': 'cross-verification-api',
        'version': '1.0.0'
    })

# Test endpoint
@cross_verify_bp.route('/api/cross-verify/test', methods=['GET'])
def test_verification():
    """
    Test endpoint with sample data
    """
    try:
        # Test with sample content
        test_content = "Sample news content for testing cross-verification"
        simulated_report = generate_simulated_serper_report(test_content)
        verification_result = verification_engine.cross_verify_content(simulated_report)
        formatted_result = format_verification_result(verification_result, test_content)
        
        return jsonify({
            'success': True,
            'message': 'Cross-verification test completed successfully',
            'test_result': formatted_result
        })
        
    except Exception as e:
        logger.error(f"Test verification error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Test failed: {str(e)}'
        }), 500
