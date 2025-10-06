#!/usr/bin/env python3
"""
Serper API Integration Test
This script demonstrates how to use the SERPER_API_KEY from the .env file
to perform web searches and display results in the terminal.
"""

import os
import requests
import json
from dotenv import load_dotenv
from typing import Dict, List, Optional

# Load environment variables
load_dotenv()

class SerperSearchClient:
    """Client for interacting with Serper API"""
    
    def __init__(self):
        self.api_key = os.getenv('SERPER_API_KEY')
        self.base_url = "https://google.serper.dev/search"
        
        if not self.api_key:
            raise ValueError("SERPER_API_KEY not found in environment variables")
    
    def search(self, query: str, num_results: int = 5) -> Optional[Dict]:
        """Perform a web search using Serper API"""
        
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        
        payload = {
            'q': query,
            'num': num_results
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error: HTTP {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None
    
    def display_results(self, results: Dict, query: str):
        """Display search results in a formatted way"""
        
        print("\n" + "="*80)
        print(f"SERPER API SEARCH RESULTS FOR: '{query}'")
        print("="*80)
        
        if 'organic' in results:
            organic_results = results['organic']
            
            for i, result in enumerate(organic_results, 1):
                print(f"\n[{i}] {result.get('title', 'No Title')}")
                print(f"    URL: {result.get('link', 'No URL')}")
                print(f"    Snippet: {result.get('snippet', 'No snippet available')}")
                
                if 'date' in result:
                    print(f"    Date: {result['date']}")
        
        # Display knowledge graph if available
        if 'knowledgeGraph' in results:
            kg = results['knowledgeGraph']
            print(f"\n📊 KNOWLEDGE GRAPH:")
            print(f"    Title: {kg.get('title', 'N/A')}")
            print(f"    Type: {kg.get('type', 'N/A')}")
            if 'description' in kg:
                print(f"    Description: {kg['description']}")
        
        # Display related searches
        if 'relatedSearches' in results:
            print(f"\n🔍 RELATED SEARCHES:")
            for related in results['relatedSearches'][:3]:
                print(f"    - {related.get('query', 'N/A')}")
        
        print("\n" + "="*80)

def main():
    """Main function to demonstrate Serper API integration"""
    
    print("🚀 Starting Serper API Integration Test...")
    
    try:
        # Initialize the client
        client = SerperSearchClient()
        print(f"✅ Successfully loaded SERPER_API_KEY: {client.api_key[:10]}...")
        
        # Test queries
        test_queries = [
            "fake news detection machine learning",
            "fact checking algorithms 2024",
            "misinformation detection AI"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Searching for: '{query}'")
            results = client.search(query, num_results=3)
            
            if results:
                client.display_results(results, query)
                print(f"✅ Search completed successfully!")
            else:
                print(f"❌ Search failed for query: '{query}'")
            
            # Add a small delay between requests
            import time
            time.sleep(1)
        
        print("\n🎉 Serper API integration test completed!")
        
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

if __name__ == "__main__":
    main()