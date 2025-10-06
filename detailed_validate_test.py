#!/usr/bin/env python3
import requests
import json
import time

def detailed_test_validate_endpoint():
    """Detailed test of the /api/validate endpoint to see full response structure"""
    base_url = "http://localhost:5001"
    
    test_data = {
        "text": "Breaking: Scientists discover cure for all diseases in secret lab",
        "type": "text"
    }
    
    print("Testing /api/validate endpoint with detailed response...\n")
    print(f"Request data: {json.dumps(test_data, indent=2)}")
    print("-" * 60)
    
    try:
        start_time = time.time()
        
        response = requests.post(
            f"{base_url}/api/validate",
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"Status Code: {response.status_code}")
        print(f"Processing Time: {processing_time:.2f} seconds")
        print(f"Response Headers: {dict(response.headers)}")
        print("-" * 60)
        
        if response.status_code == 200:
            try:
                result = response.json()
                print("Full Response JSON:")
                print(json.dumps(result, indent=2, default=str))
                
                # Extract key information
                print("\n" + "=" * 60)
                print("KEY INFORMATION EXTRACTED:")
                print("=" * 60)
                
                print(f"Status: {result.get('status', 'Unknown')}")
                print(f"Input Type: {result.get('input_type', 'Unknown')}")
                
                # MHFN Analysis
                mhfn = result.get('mhfn_analysis', {})
                print(f"\nMHFN Analysis:")
                print(f"  Prediction: {mhfn.get('prediction', 'Unknown')}")
                print(f"  Confidence: {mhfn.get('confidence', 'Unknown')}")
                print(f"  Processing Time: {mhfn.get('processing_time', 'Unknown')}s")
                
                # Proof Analysis
                proof = result.get('proof_analysis', {})
                print(f"\nProof Analysis:")
                print(f"  Total Sources Checked: {proof.get('total_sources_checked', 0)}")
                print(f"  Verified Sources: {proof.get('verified_sources', 0)}")
                print(f"  Error Sources: {proof.get('error_sources', 0)}")
                print(f"  Processing Time: {proof.get('processing_time', 'Unknown')}s")
                
                proofs = proof.get('proofs', [])
                if proofs:
                    print(f"\n  Proofs Details:")
                    for i, p in enumerate(proofs, 1):
                        print(f"    Proof {i}:")
                        print(f"      Source: {p.get('source', 'Unknown')}")
                        print(f"      Status: {p.get('status', 'Unknown')}")
                        print(f"      URL: {p.get('url', 'Unknown')}")
                        print(f"      Summary: {p.get('summary', 'Unknown')[:100]}...")
                
                # Final Verdict
                verdict = result.get('final_verdict', {})
                print(f"\nFinal Verdict:")
                print(f"  Prediction: {verdict.get('prediction', 'Unknown')}")
                print(f"  Confidence: {verdict.get('confidence', 'Unknown')}")
                print(f"  Reasoning: {verdict.get('reasoning', 'Unknown')}")
                
                # Performance Metrics
                perf = result.get('performance_metrics', {})
                print(f"\nPerformance Metrics:")
                print(f"  Total Processing Time: {perf.get('total_processing_time', 'Unknown')}s")
                print(f"  MHFN Time: {perf.get('mhfn_time', 'Unknown')}s")
                print(f"  Proof Fetching Time: {perf.get('proof_fetching_time', 'Unknown')}s")
                print(f"  Parallel Speedup: {perf.get('parallel_speedup', 'Unknown')}")
                
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Raw response: {response.text}")
        else:
            print(f"Error Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    detailed_test_validate_endpoint()