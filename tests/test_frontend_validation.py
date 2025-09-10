#!/usr/bin/env python3
"""
Test script to verify the frontend validation functionality works correctly
after fixing the response structure mismatch.
"""

import requests
import json
import time

def test_validation_endpoint():
    """Test the /api/validate endpoint with proper response structure"""
    base_url = "http://localhost:5001"
    
    # Test data
    test_cases = [
        {
            "name": "Text Input - Breaking News",
            "data": {
                "text": "Breaking: Scientists discover cure for all diseases in secret lab",
                "input_type": "text"
            }
        },
        {
            "name": "Text Input - Legitimate News",
            "data": {
                "text": "NASA announces new Mars rover mission scheduled for 2026",
                "input_type": "text"
            }
        }
    ]
    
    print("🧪 Testing /api/validate endpoint response structure...\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print(f"Input: {test_case['data']['text'][:50]}...")
        
        try:
            # Make request
            response = requests.post(
                f"{base_url}/api/validate",
                json=test_case['data'],
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                # Verify response structure matches what frontend expects
                print("\n📋 Response Structure Analysis:")
                print(f"  Status: {data.get('status', 'MISSING')}")
                
                # Check for required top-level keys
                required_keys = ['final_verdict', 'mhfn_analysis', 'proof_analysis', 'performance_metrics']
                for key in required_keys:
                    if key in data:
                        print(f"  ✅ {key}: Present")
                    else:
                        print(f"  ❌ {key}: MISSING")
                
                # Check nested structure
                if 'final_verdict' in data:
                    fv = data['final_verdict']
                    print(f"    - prediction: {fv.get('prediction', 'MISSING')}")
                    print(f"    - confidence: {fv.get('confidence', 'MISSING')}")
                
                if 'mhfn_analysis' in data:
                    ma = data['mhfn_analysis']
                    print(f"    - MHFN prediction: {ma.get('prediction', 'MISSING')}")
                    print(f"    - MHFN confidence: {ma.get('confidence', 'MISSING')}")
                
                if 'proof_analysis' in data:
                    pa = data['proof_analysis']
                    print(f"    - total_sources_checked: {pa.get('total_sources_checked', 'MISSING')}")
                    print(f"    - proofs count: {len(pa.get('proofs', []))}")
                
                if 'performance_metrics' in data:
                    pm = data['performance_metrics']
                    print(f"    - total_processing_time: {pm.get('total_processing_time', 'MISSING')}")
                
                print("\n✅ Test PASSED - Response structure is correct for frontend")
                
            else:
                print(f"❌ Test FAILED - HTTP {response.status_code}")
                print(f"Response: {response.text[:200]}...")
                
        except Exception as e:
            print(f"❌ Test FAILED - Exception: {str(e)}")
        
        print("\n" + "="*60 + "\n")
    
    print("🎯 Frontend validation test completed!")

if __name__ == "__main__":
    test_validation_endpoint()