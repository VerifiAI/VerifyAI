#!/usr/bin/env python3

import requests
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test the exact scenario causing the error
print("Testing image URL processing fix...")

# This is what's causing the error - requests.get with local file path
test_image_path = "test_mars_image.jpg"

print(f"\n1. Testing direct requests.get with local file path: {test_image_path}")
try:
    response = requests.get(test_image_path)
    print(f"Success: {response}")
except Exception as e:
    print(f"Error (expected): {e}")
    print(f"Error type: {type(e).__name__}")

# Now test the proper way using our load_image_from_path_or_url function
print(f"\n2. Testing proper image loading function...")
try:
    from app import load_image_from_path_or_url
    result = load_image_from_path_or_url(test_image_path)
    print(f"load_image_from_path_or_url result: {type(result)}")
    if result is not None:
        print("✅ Image loaded successfully")
    else:
        print("❌ Image loading failed")
except Exception as e:
    print(f"Error in load_image_from_path_or_url: {e}")

# Test with a URL
print(f"\n3. Testing with actual URL...")
test_url = "https://httpbin.org/image/png"
try:
    from app import load_image_from_path_or_url
    result = load_image_from_path_or_url(test_url)
    print(f"URL load result: {type(result)}")
    if result is not None:
        print("✅ URL image loaded successfully")
    else:
        print("❌ URL image loading failed")
except Exception as e:
    print(f"Error loading from URL: {e}")

print("\n" + "="*50)
print("DIAGNOSIS: The error occurs when requests.get is called directly")
print("with a local file path instead of using load_image_from_path_or_url")
print("="*50)