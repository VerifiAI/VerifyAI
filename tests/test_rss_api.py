#!/usr/bin/env python3
import requests
import json

def test_rss_fact_checker():
    url = "http://localhost:5001/api/rss-fact-check"
    
    # Test data
    test_data = {
        "text": "Breaking: Scientists discover new planet in our solar system"
    }
    
    try:
        print(f"Testing RSS Fact Checker API at {url}")
        print(f"Request data: {json.dumps(test_data, indent=2)}")
        print("-" * 50)
        
        response = requests.post(url, json=test_data, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print("-" * 50)
        
        if response.status_code == 200:
            result = response.json()
            print("SUCCESS! API Response:")
            print(json.dumps(result, indent=2))
        else:
            print(f"ERROR! Status: {response.status_code}")
            print(f"Response text: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Raw response: {response.text}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    test_rss_fact_checker()