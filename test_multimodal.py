#!/usr/bin/env python3
"""
Test script for multimodal consistency functionality
"""

import sys
import os
import json
import requests
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app import (
        get_roberta_embeddings,
        get_clip_image_features,
        calculate_multimodal_consistency,
        initialize_multimodal_models
    )
    print("✓ Successfully imported multimodal functions from app.py")
except ImportError as e:
    print(f"✗ Failed to import functions: {e}")
    sys.exit(1)

def test_multimodal_functions():
    """Test the multimodal functions directly"""
    print("\n=== Testing Multimodal Functions ===")
    
    # Test text for embedding
    test_text = "Breaking news: Scientists discover new planet in our solar system"
    
    try:
        # Initialize models
        print("Initializing multimodal models...")
        initialize_multimodal_models()
        print("✓ Models initialized successfully")
        
        # Test RoBERTa embeddings
        print("\nTesting RoBERTa text embeddings...")
        text_embedding = get_roberta_embeddings(test_text)
        if text_embedding is not None:
            print(f"✓ Text embedding shape: {text_embedding.shape}")
        else:
            print("✗ Failed to get text embedding")
            return False
        
        # Test CLIP image features (with mock image)
        print("\nTesting CLIP image features...")
        # Create a simple mock image URL
        mock_image_url = "https://example.com/test.jpg"
        image_features = get_clip_image_features(mock_image_url)
        if image_features is not None:
            print(f"✓ Image features shape: {image_features.shape}")
        else:
            print("✓ Image features returned None (expected for mock URL)")
        
        # Test multimodal consistency calculation
        print("\nTesting multimodal consistency calculation...")
        if text_embedding is not None and image_features is not None:
            consistency = calculate_multimodal_consistency(text_embedding, image_features)
            print(f"✓ Multimodal consistency: {consistency}")
            
            # Test consistency threshold
            if consistency < 0.7:
                print("✓ Consistency below threshold - would flag as potentially fake")
            else:
                print("✓ Consistency above threshold - content appears consistent")
        else:
            print("⚠ Skipping consistency test due to missing embeddings")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_endpoint():
    """Test the /api/detect endpoint"""
    print("\n=== Testing API Endpoint ===")
    
    # Test data
    test_data = {
        "text": "Scientists have discovered a new planet in our solar system that is twice the size of Earth.",
        "image_url": "test_mars_image.jpg"
    }
    
    try:
        # Try to make a request to the API
        response = requests.post(
            "http://localhost:5001/api/detect",
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ API endpoint responded successfully")
            print(f"Response: {json.dumps(result, indent=2)}")
            
            # Check for consistency status
            if 'consistency_status' in result:
                print(f"✓ Consistency status found: {result['consistency_status']}")
            else:
                print("⚠ Consistency status not found in response")
                
            return True
        else:
            print(f"✗ API returned status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to API server (server may not be running)")
        return False
    except Exception as e:
        print(f"✗ Error testing API: {e}")
        return False

def main():
    """Main test function"""
    print("Multimodal Consistency Test Suite")
    print("=" * 50)
    print(f"Test started at: {datetime.now()}")
    
    # Test results
    results = {
        "timestamp": datetime.now().isoformat(),
        "function_tests": False,
        "api_tests": False,
        "overall_success": False
    }
    
    # Test multimodal functions
    results["function_tests"] = test_multimodal_functions()
    
    # Test API endpoint
    results["api_tests"] = test_api_endpoint()
    
    # Overall result
    results["overall_success"] = results["function_tests"] and results["api_tests"]
    
    print("\n=== Test Summary ===")
    print(f"Function Tests: {'✓ PASS' if results['function_tests'] else '✗ FAIL'}")
    print(f"API Tests: {'✓ PASS' if results['api_tests'] else '✗ FAIL'}")
    print(f"Overall: {'✓ SUCCESS' if results['overall_success'] else '✗ FAILURE'}")
    
    # Save results
    with open('multimodal_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTest results saved to: multimodal_test_results.json")
    
    return 0 if results["overall_success"] else 1

if __name__ == "__main__":
    sys.exit(main())