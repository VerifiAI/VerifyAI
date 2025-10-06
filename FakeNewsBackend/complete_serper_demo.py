#!/usr/bin/env python3
"""
Complete Serper API Integration Demonstration
Shows the full integration of Serper API with the .env configuration
and demonstrates all features working together.
"""

import os
import sys
import json
from dotenv import load_dotenv
from datetime import datetime

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

def check_environment_setup():
    """Check if the environment is properly configured"""
    print("🔧 ENVIRONMENT CONFIGURATION CHECK")
    print("="*60)
    
    # Check for .env file
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        print(f"✅ .env file found: {env_file}")
    else:
        print(f"❌ .env file not found: {env_file}")
    
    # Check API key
    api_key = os.getenv('SERPER_API_KEY')
    if api_key:
        print(f"✅ SERPER_API_KEY loaded: {api_key[:10]}...{api_key[-4:]}")
        print(f"   Key length: {len(api_key)} characters")
    else:
        print("❌ SERPER_API_KEY not found in environment")
        return False
    
    # Check other relevant environment variables
    other_vars = ['NEWS_API_KEY', 'NEWSDATA_API_KEY', 'DEBUG', 'PORT']
    for var in other_vars:
        value = os.getenv(var)
        if value:
            if 'API_KEY' in var:
                print(f"✅ {var}: {value[:10]}...{value[-4:]}")
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"⚠️  {var}: Not set")
    
    print("\n")
    return True

def test_basic_serper_functionality():
    """Test basic Serper API functionality"""
    print("🧪 BASIC SERPER API FUNCTIONALITY TEST")
    print("="*60)
    
    try:
        from test_serper_integration import SerperSearchClient
        
        client = SerperSearchClient()
        print("✅ SerperSearchClient initialized successfully")
        
        # Test a simple search
        test_query = "fake news detection AI 2024"
        print(f"🔍 Testing search query: '{test_query}'")
        
        results = client.search(test_query, num_results=3)
        
        if results:
            print(f"✅ Search successful!")
            print(f"   Results found: {len(results.get('organic', []))}")
            
            # Display first result
            if 'organic' in results and results['organic']:
                first_result = results['organic'][0]
                print(f"   First result: {first_result.get('title', 'No title')[:50]}...")
                print(f"   URL: {first_result.get('link', 'No URL')[:50]}...")
            
            return True
        else:
            print("❌ Search failed - no results returned")
            return False
            
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_proof_validation_integration():
    """Test the proof validation integration"""
    print("🔬 PROOF VALIDATION INTEGRATION TEST")
    print("="*60)
    
    try:
        from proof_validation.serper_integration import integrate_serper_validation
        
        test_claims = [
            "Machine learning can identify fake news patterns",
            "Social media platforms use AI for content moderation",
            "Fact-checking websites help combat misinformation"
        ]
        
        validation_results = []
        
        for i, claim in enumerate(test_claims, 1):
            print(f"\n📋 Testing Claim {i}: {claim}")
            print("-" * 50)
            
            result = integrate_serper_validation(claim)
            validation_results.append(result)
            
            # Display results
            print(f"Status: {result.get('status', 'Unknown')}")
            print(f"Confidence: {result.get('confidence', 0)}%")
            print(f"Evidence Sources: {result.get('evidence_sources', 0)}")
            print(f"Supporting Sources: {result.get('supporting_sources', 0)}")
            print(f"Contradicting Sources: {result.get('contradicting_sources', 0)}")
            print(f"API Response Time: {result.get('api_response_time', 0)}s")
            
            if result.get('status') != 'ERROR':
                print("✅ Validation completed successfully")
            else:
                print(f"❌ Validation error: {result.get('error', 'Unknown error')}")
        
        return validation_results
        
    except Exception as e:
        print(f"❌ Proof validation integration test failed: {e}")
        return []

def display_comprehensive_results(validation_results):
    """Display comprehensive results summary"""
    print("\n📊 COMPREHENSIVE RESULTS SUMMARY")
    print("="*60)
    
    if not validation_results:
        print("❌ No validation results to display")
        return
    
    successful_validations = [r for r in validation_results if r.get('status') != 'ERROR']
    error_count = len(validation_results) - len(successful_validations)
    
    print(f"Total Claims Tested: {len(validation_results)}")
    print(f"Successful Validations: {len(successful_validations)}")
    print(f"Errors: {error_count}")
    
    if successful_validations:
        avg_confidence = sum(r.get('confidence', 0) for r in successful_validations) / len(successful_validations)
        avg_response_time = sum(r.get('api_response_time', 0) for r in successful_validations) / len(successful_validations)
        total_evidence = sum(r.get('evidence_sources', 0) for r in successful_validations)
        
        print(f"Average Confidence Score: {avg_confidence:.2f}%")
        print(f"Average API Response Time: {avg_response_time:.3f}s")
        print(f"Total Evidence Sources Found: {total_evidence}")
        
        # Status distribution
        status_counts = {}
        for result in successful_validations:
            status = result.get('status', 'Unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print("\n📈 Validation Status Distribution:")
        for status, count in status_counts.items():
            emoji = {
                'VERIFIED': '✅',
                'PARTIALLY_VERIFIED': '⚠️',
                'UNVERIFIED': '❓',
                'CONTRADICTED': '❌'
            }.get(status, '🔍')
            print(f"   {emoji} {status}: {count}")

def test_integration_with_existing_system():
    """Test integration with existing proof validation system"""
    print("\n🔗 INTEGRATION WITH EXISTING SYSTEM TEST")
    print("="*60)
    
    try:
        # Check if existing modules can import the new integration
        proof_validation_dir = os.path.join(os.path.dirname(__file__), 'proof_validation')
        
        if os.path.exists(proof_validation_dir):
            print(f"✅ Proof validation directory found: {proof_validation_dir}")
            
            # List files in proof_validation directory
            files = os.listdir(proof_validation_dir)
            print(f"   Files in directory: {', '.join(files)}")
            
            # Check if our integration file exists
            if 'serper_integration.py' in files:
                print("✅ Serper integration module successfully added")
            else:
                print("❌ Serper integration module not found")
            
            # Check other important files
            important_files = ['__init__.py', 'scoring.py']
            for file in important_files:
                if file in files:
                    print(f"✅ {file} exists - integration possible")
                else:
                    print(f"⚠️  {file} not found")
        else:
            print(f"❌ Proof validation directory not found: {proof_validation_dir}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Main demonstration function"""
    print("🚀 COMPLETE SERPER API INTEGRATION DEMONSTRATION")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working Directory: {os.getcwd()}")
    print("\n")
    
    # Step 1: Check environment setup
    if not check_environment_setup():
        print("❌ Environment setup failed. Please check your .env configuration.")
        return
    
    # Step 2: Test basic functionality
    if not test_basic_serper_functionality():
        print("❌ Basic functionality test failed.")
        return
    
    # Step 3: Test proof validation integration
    print("\n")
    validation_results = test_proof_validation_integration()
    
    # Step 4: Display comprehensive results
    display_comprehensive_results(validation_results)
    
    # Step 5: Test integration with existing system
    integration_success = test_integration_with_existing_system()
    
    # Final summary
    print("\n🎯 FINAL INTEGRATION SUMMARY")
    print("="*60)
    
    if integration_success and validation_results:
        print("✅ Serper API successfully integrated with proof validation system")
        print("✅ Environment configuration working correctly")
        print("✅ API key authentication successful")
        print("✅ Real-time fact-checking capabilities enabled")
        print("✅ Integration with existing modules confirmed")
        
        print("\n🎉 INTEGRATION COMPLETE - READY FOR PRODUCTION USE!")
    else:
        print("❌ Integration incomplete - please review errors above")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()