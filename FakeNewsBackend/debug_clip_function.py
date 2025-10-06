#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test the get_clip_image_features function directly
print("Testing get_clip_image_features function...")

try:
    from app import get_clip_image_features, initialize_multimodal_models
    
    # Initialize models first
    print("Initializing multimodal models...")
    initialize_multimodal_models()
    
    # Test with local file path
    print(f"\nTesting get_clip_image_features with 'test_mars_image.jpg'...")
    result = get_clip_image_features("test_mars_image.jpg")
    print(f"Result: {type(result) if result is not None else 'None'}")
    
except Exception as e:
    print(f"Error in get_clip_image_features: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("If this function fails with 'No scheme supplied', then the bug is here")
print("If it works, then the bug is elsewhere in the API endpoint")
print("="*60)