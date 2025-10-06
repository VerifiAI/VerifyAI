#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Testing CLIP processor directly...")

try:
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image
    import torch
    
    print("Initializing CLIP models...")
    clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    clip_model.eval()
    
    print("Loading test image...")
    test_image_path = "test_mars_image.jpg"
    
    if os.path.exists(test_image_path):
        image = Image.open(test_image_path).convert('RGB')
        print(f"Image loaded successfully: {image.size}")
        
        print("Processing image with CLIP processor...")
        inputs = clip_processor(images=image, return_tensors='pt')
        print(f"CLIP processor inputs: {inputs.keys()}")
        
        print("Getting image features...")
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
            print(f"Image features shape: {image_features.shape}")
            
        print("SUCCESS: CLIP processing completed without errors")
    else:
        print(f"Test image not found: {test_image_path}")
        
except Exception as e:
    print(f"Error in CLIP processing: {e}")
    import traceback
    traceback.print_exc()