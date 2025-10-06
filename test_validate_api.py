#!/usr/bin/env python3
import requests
import json
import time

def test_validate_endpoint():
    """Test the /api/validate endpoint with different input types"""
    base_url = "http://localhost:5001"
    
    # Test cases
    test_cases = [
        {
            "name": "Text Input - Suspicious News",
            "data": {
                "text": "Breaking: Scientists discover cure for all diseases in secret lab",
                "type": "text"
            }
        },
        {
            "name": "Text Input - Legitimate News",
            "data": {
                "text": "NASA announces new Mars rover mission scheduled for 2026",
                "type": "text"
            }
        },
        {
            "name": "URL Input",
            "data": {
                "url": "https://www.bbc.com/news",
                "type": "url"
            }
        }
    ]
    
    print("Testing /api/validate endpoint...\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print("-" * 50)
        
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{base_url}/api/validate",
                json=test_case['data'],
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            print(f"Status Code: {response.status_code}")
            print(f"Processing Time: {processing_time:.2f} seconds")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Success: {result.get('status', 'Unknown')}")
                print(f"Verdict: {result.get('verdict', 'Unknown')}")
                print(f"Confidence: {result.get('confidence', 'Unknown')}")
                print(f"MHFN Prediction: {result.get('mhfn_prediction', 'Unknown')}")
                
                proofs = result.get('proofs', [])
                print(f"Proofs Found: {len(proofs)}")
                
                for j, proof in enumerate(proofs, 1):
                    print(f"  Proof {j}:")
                    print(f"    Source: {proof.get('source', 'Unknown')}")
                    print(f"    Status: {proof.get('status', 'Unknown')}")
                    print(f"    URL: {proof.get('url', 'Unknown')}")
                    print(f"    Summary: {proof.get('summary', 'Unknown')[:100]}...")
                
            else:
                print(f"Error: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
            
        print("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
    test_validate_endpoint()