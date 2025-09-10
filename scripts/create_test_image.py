#!/usr/bin/env python3
"""
Create a simple test image for multimodal testing
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_test_image():
    """Create a simple test image with text"""
    
    # Create a new image with white background
    width, height = 400, 300
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Add some text to the image
    try:
        # Try to use a default font
        font = ImageFont.load_default()
    except:
        font = None
    
    # Draw some shapes and text
    draw.rectangle([50, 50, 350, 250], outline='blue', width=3)
    draw.text((60, 70), "NASA Mars Discovery", fill='black', font=font)
    draw.text((60, 100), "Water found on Mars surface", fill='red', font=font)
    draw.text((60, 130), "Scientific breakthrough", fill='blue', font=font)
    
    # Add some geometric shapes to make it more interesting
    draw.ellipse([200, 160, 280, 220], outline='green', width=2)
    draw.polygon([(100, 180), (120, 160), (140, 180), (120, 200)], outline='purple', width=2)
    
    # Save the image
    image_path = 'test_mars_image.jpg'
    image.save(image_path, 'JPEG')
    
    print(f"✓ Test image created: {image_path}")
    print(f"✓ Image size: {width}x{height}")
    
    return os.path.abspath(image_path)

if __name__ == "__main__":
    image_path = create_test_image()
    print(f"Full path: {image_path}")