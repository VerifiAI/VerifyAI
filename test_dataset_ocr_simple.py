#!/usr/bin/env python3
"""
Simple OCR Dataset Testing
Provides analysis and testing instructions for OCR with dataset images
"""

import os
import urllib.request
import urllib.error
from PIL import Image

def check_server_status():
    """Check if the local server is running"""
    try:
        response = urllib.request.urlopen('http://localhost:8080')
        print("✅ Local server is running on port 8080")
        return True
    except urllib.error.URLError:
        print("❌ Local server is not running on port 8080")
        print("Please start the server with: python3 -m http.server 8080")
        return False

def analyze_dataset_images():
    """Analyze dataset images for OCR potential"""
    print("\n=== Dataset Image Analysis ===")
    
    images_dir = "images"
    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        return []
    
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("❌ No image files found")
        return []
    
    print(f"Found {len(image_files)} images in dataset:")
    
    analysis_results = []
    
    for img in image_files:
        img_path = os.path.join(images_dir, img)
        try:
            # Get file info
            file_size = os.path.getsize(img_path)
            
            # Open image to get dimensions
            with Image.open(img_path) as pil_img:
                width, height = pil_img.size
                mode = pil_img.mode
            
            # Analyze likelihood of containing text
            text_keywords = ['quote', 'meme', 'infographic', 'chart', 'conspiracy']
            likely_has_text = any(keyword in img.lower() for keyword in text_keywords)
            
            result = {
                'filename': img,
                'size_bytes': file_size,
                'dimensions': f"{width}x{height}",
                'mode': mode,
                'likely_has_text': likely_has_text,
                'ocr_priority': 'High' if likely_has_text else 'Medium'
            }
            
            analysis_results.append(result)
            
            status_icon = "📝" if likely_has_text else "🖼️"
            print(f"  {status_icon} {img}")
            print(f"      Size: {file_size:,} bytes | Dimensions: {width}x{height} | Mode: {mode}")
            print(f"      OCR Priority: {result['ocr_priority']} | Text Likely: {'Yes' if likely_has_text else 'Maybe'}")
            
        except Exception as e:
            print(f"  ❌ {img} - Error analyzing: {e}")
    
    return analysis_results

def check_ocr_integration():
    """Check OCR module integration"""
    print("\n=== OCR Integration Check ===")
    
    ocr_module_path = "FakeNewsBackend/ocr-module.js"
    
    if not os.path.exists(ocr_module_path):
        print(f"❌ OCR module not found: {ocr_module_path}")
        return False
    
    try:
        with open(ocr_module_path, 'r') as f:
            content = f.read()
        
        # Check for required functions
        required_functions = [
            'initializeOCRModule',
            'processImageWithOCR',
            'executeOCRPipeline',
            'validateImageFile'
        ]
        
        missing_functions = []
        for func in required_functions:
            if func not in content:
                missing_functions.append(func)
        
        if missing_functions:
            print(f"❌ Missing functions: {', '.join(missing_functions)}")
            return False
        
        print("✅ All required OCR functions found")
        
        # Check Tesseract.js integration
        if 'tesseract' in content.lower():
            print("✅ Tesseract.js integration detected")
        else:
            print("⚠️  Tesseract.js integration not clearly detected")
        
        # Check file size
        file_size = os.path.getsize(ocr_module_path)
        print(f"✅ OCR module size: {file_size:,} bytes")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking OCR module: {e}")
        return False

def check_test_page():
    """Check if test page exists"""
    print("\n=== Test Page Check ===")
    
    test_page_path = "FakeNewsBackend/test_dataset_ocr.html"
    
    if not os.path.exists(test_page_path):
        print(f"❌ Test page not found: {test_page_path}")
        return False
    
    try:
        with open(test_page_path, 'r') as f:
            content = f.read()
        
        # Check for required elements
        required_elements = [
            'conspiracy_infographic.jpg',
            'fake_celebrity_quote.jpg',
            'fake_health_meme.jpg',
            'legitimate_chart.jpg',
            'test_mars_image.jpg',
            'testOCR(',
            'ocr-module.js'
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"⚠️  Missing elements in test page: {', '.join(missing_elements)}")
        else:
            print("✅ Test page contains all required elements")
        
        file_size = os.path.getsize(test_page_path)
        print(f"✅ Test page size: {file_size:,} bytes")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking test page: {e}")
        return False

def provide_testing_instructions(analysis_results):
    """Provide comprehensive testing instructions"""
    print("\n" + "="*60)
    print("🧪 OCR DATASET TESTING INSTRUCTIONS")
    print("="*60)
    
    print("\n📋 SETUP:")
    print("1. Ensure local server is running: python3 -m http.server 8080")
    print("2. Open browser and navigate to: http://localhost:8080/test_dataset_ocr.html")
    
    print("\n🎯 TESTING PROCEDURE:")
    print("1. You'll see 5 dataset images with 'Test OCR' buttons")
    print("2. Click each 'Test OCR' button to extract text")
    print("3. Wait 10-30 seconds for OCR processing per image")
    print("4. Check results displayed below each image")
    
    print("\n📊 EXPECTED RESULTS BY IMAGE:")
    
    high_priority = [r for r in analysis_results if r['ocr_priority'] == 'High']
    medium_priority = [r for r in analysis_results if r['ocr_priority'] == 'Medium']
    
    if high_priority:
        print("\n🔥 HIGH PRIORITY (Likely to contain text):")
        for result in high_priority:
            print(f"   📝 {result['filename']} ({result['dimensions']})")
            if 'quote' in result['filename'].lower():
                print("      Expected: Quote text, attribution")
            elif 'meme' in result['filename'].lower():
                print("      Expected: Meme text, captions")
            elif 'infographic' in result['filename'].lower():
                print("      Expected: Headlines, statistics, labels")
            elif 'chart' in result['filename'].lower():
                print("      Expected: Chart labels, data values, titles")
    
    if medium_priority:
        print("\n📷 MEDIUM PRIORITY (May contain text):")
        for result in medium_priority:
            print(f"   🖼️  {result['filename']} ({result['dimensions']})")
            print("      Expected: Embedded text, watermarks, or no text")
    
    print("\n✅ SUCCESS CRITERIA:")
    print("- OCR process completes without JavaScript errors")
    print("- Text extraction works (even if no text found)")
    print("- Results display properly in the interface")
    print("- Processing time is reasonable (< 60 seconds per image)")
    
    print("\n🐛 TROUBLESHOOTING:")
    print("- If 'OCR function not available': Refresh page")
    print("- If processing hangs: Check browser console for errors")
    print("- If no results: Verify Tesseract.js CDN access")
    print("- If images don't load: Check image file paths")
    
    print("\n📈 PERFORMANCE NOTES:")
    print("- First OCR may take longer (Tesseract.js initialization)")
    print("- Larger images take more processing time")
    print("- Some images may legitimately contain no readable text")
    
    print("\n" + "="*60)

def main():
    """Main test runner"""
    print("🔍 OCR Dataset Testing Setup and Analysis")
    print("="*50)
    
    # Check server
    server_ok = check_server_status()
    
    # Analyze images
    analysis_results = analyze_dataset_images()
    
    # Check OCR integration
    ocr_ok = check_ocr_integration()
    
    # Check test page
    test_page_ok = check_test_page()
    
    # Provide instructions
    if analysis_results:
        provide_testing_instructions(analysis_results)
    
    # Summary
    print("\n📊 SETUP SUMMARY:")
    print(f"✅ Server Running: {'Yes' if server_ok else 'No'}")
    print(f"✅ Images Found: {len(analysis_results)}")
    print(f"✅ OCR Module: {'Ready' if ocr_ok else 'Issues'}")
    print(f"✅ Test Page: {'Ready' if test_page_ok else 'Issues'}")
    
    high_priority_count = len([r for r in analysis_results if r['ocr_priority'] == 'High'])
    print(f"📝 High Priority Images: {high_priority_count}")
    
    if server_ok and analysis_results and ocr_ok and test_page_ok:
        print("\n🎉 Everything is ready for OCR testing!")
        print("👉 Open http://localhost:8080/test_dataset_ocr.html to start testing")
        return True
    else:
        print("\n⚠️  Some issues detected. Please resolve them before testing.")
        return False

if __name__ == "__main__":
    main()