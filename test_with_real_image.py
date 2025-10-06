#!/usr/bin/env python3
"""
Test multimodal functionality with a real image URL
"""

import requests
import json
from datetime import datetime

def test_multimodal_with_real_image():
    """Test the multimodal functionality with a real image"""
    
    # Use a real image URL that should be accessible
    test_data = {
        "text": "Breaking: NASA announces discovery of water on Mars surface",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/02/OSIRIS_Mars_true_color.jpg/256px-OSIRIS_Mars_true_color.jpg"
    }
    
    print("Testing multimodal consistency with real image...")
    print(f"Text: {test_data['text']}")
    print(f"Image URL: {test_data['image_url']}")
    
    try:
        response = requests.post(
            "http://localhost:5001/api/detect",
            json=test_data,
            timeout=60  # Longer timeout for image processing
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✓ API Response:")
            print(json.dumps(result, indent=2))
            
            # Check for consistency status
            if 'consistency_status' in result:
                print(f"\n✓ Multimodal consistency status: {result['consistency_status']}")
                if 'consistency_score' in result:
                    print(f"✓ Consistency score: {result['consistency_score']}")
                return True
            else:
                print("\n⚠ Consistency status not found in response")
                print("This might indicate the multimodal processing didn't complete")
                return False
        else:
            print(f"\n✗ API returned status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False

def test_text_only():
    """Test with text only for comparison"""
    
    test_data = {
        "text": "Breaking: NASA announces discovery of water on Mars surface"
    }
    
    print("\nTesting text-only detection...")
    
    try:
        response = requests.post(
            "http://localhost:5001/api/detect",
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✓ Text-only Response:")
            print(json.dumps(result, indent=2))
            return True
        else:
            print(f"\n✗ API returned status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False

if __name__ == "__main__":
    print("Multimodal API Test with Real Image")
    print("=" * 50)
    print(f"Test started at: {datetime.now()}")
    
    # Test multimodal
    multimodal_success = test_multimodal_with_real_image()
    
    # Test text-only for comparison
    text_success = test_text_only()
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Multimodal Test: {'✓ PASS' if multimodal_success else '✗ FAIL'}")
    print(f"Text-only Test: {'✓ PASS' if text_success else '✗ FAIL'}")
    
    if multimodal_success and text_success:
        print("\n🎉 All tests passed! Multimodal functionality is working.")
    else:
        print("\n⚠ Some tests failed. Check the output above for details.")