#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the function that's causing the issue
from app import load_image_from_path_or_url, get_clip_image_features

print("Testing image loading with local file path...")

# Test with the same input that's causing the error
test_image_path = "test_mars_image.jpg"

print(f"Testing with: {test_image_path}")

try:
    result = load_image_from_path_or_url(test_image_path)
    print(f"load_image_from_path_or_url result: {result}")
except Exception as e:
    print(f"Error in load_image_from_path_or_url: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50 + "\n")

try:
    result = get_clip_image_features(test_image_path)
    print(f"get_clip_image_features result: {result}")
except Exception as e:
    print(f"Error in get_clip_image_features: {e}")
    import traceback
    traceback.print_exc()