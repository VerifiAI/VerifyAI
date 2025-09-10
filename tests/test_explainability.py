#!/usr/bin/env python3
"""
Comprehensive Test Suite for Explainability Features
Tests SHAP, LIME, Grad-CAM, and BERTopic integration
"""

import pytest
import json
import sys
import os
from unittest.mock import patch, MagicMock

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app import app, logger
except ImportError as e:
    print(f"Error importing app: {e}")
    sys.exit(1)

class TestExplainabilityFeatures:
    """Test suite for explainability endpoint and features"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    @pytest.fixture
    def sample_text_data(self):
        """Sample text data for testing"""
        return {
            "text": "Breaking news: Scientists discover new planet with potential for life. This groundbreaking research could change our understanding of the universe.",
            "image_url": "https://example.com/sample_image.jpg"
        }
    
    @pytest.fixture
    def sample_fake_data(self):
        """Sample fake news data for testing"""
        return {
            "text": "SHOCKING: Aliens have landed and are secretly controlling the government! This is definitely true and not fake at all.",
            "image_url": "https://example.com/fake_image.jpg"
        }
    
    def test_explain_endpoint_exists(self, client):
        """Test that the /api/explain endpoint exists"""
        response = client.post('/api/explain', 
                             json={"text": "test", "image_url": ""},
                             content_type='application/json')
        
        # Should not return 404
        assert response.status_code != 404
        print("✓ /api/explain endpoint exists")
    
    def test_explain_endpoint_with_valid_data(self, client, sample_text_data):
        """Test explain endpoint with valid input data"""
        response = client.post('/api/explain',
                             json=sample_text_data,
                             content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Check response structure
        assert 'status' in data
        assert 'explanation' in data
        
        explanation = data['explanation']
        
        # Check for expected explanation components
        expected_components = ['shap_values', 'lime_explanation', 'topic_clusters']
        for component in expected_components:
            assert component in explanation, f"Missing {component} in explanation"
        
        print("✓ Explain endpoint returns valid structure")
    
    def test_shap_values_structure(self, client, sample_text_data):
        """Test SHAP values structure and content"""
        response = client.post('/api/explain',
                             json=sample_text_data,
                             content_type='application/json')
        
        data = json.loads(response.data)
        shap_values = data['explanation']['shap_values']
        
        # Check SHAP structure
        assert isinstance(shap_values, list), "SHAP values should be a list"
        
        if shap_values:  # If SHAP values are available
            for shap_item in shap_values:
                assert 'token' in shap_item, "SHAP item should have 'token'"
                assert 'importance' in shap_item, "SHAP item should have 'importance'"
                assert isinstance(shap_item['importance'], (int, float)), "Importance should be numeric"
        
        print("✓ SHAP values have correct structure")
    
    def test_lime_explanation_structure(self, client, sample_text_data):
        """Test LIME explanation structure and content"""
        response = client.post('/api/explain',
                             json=sample_text_data,
                             content_type='application/json')
        
        data = json.loads(response.data)
        lime_explanation = data['explanation']['lime_explanation']
        
        # Check LIME structure
        assert isinstance(lime_explanation, list), "LIME explanation should be a list"
        
        if lime_explanation:  # If LIME explanation is available
            for lime_item in lime_explanation:
                assert 'feature' in lime_item, "LIME item should have 'feature'"
                assert 'weight' in lime_item, "LIME item should have 'weight'"
                assert isinstance(lime_item['weight'], (int, float)), "Weight should be numeric"
        
        print("✓ LIME explanation has correct structure")
    
    def test_topic_clusters_structure(self, client, sample_text_data):
        """Test BERTopic clusters structure and content"""
        response = client.post('/api/explain',
                             json=sample_text_data,
                             content_type='application/json')
        
        data = json.loads(response.data)
        topic_clusters = data['explanation']['topic_clusters']
        
        # Check topic clusters structure
        assert isinstance(topic_clusters, list), "Topic clusters should be a list"
        
        if topic_clusters:  # If topic clusters are available
            for topic in topic_clusters:
                assert 'topic_id' in topic, "Topic should have 'topic_id'"
                assert 'probability' in topic, "Topic should have 'probability'"
                assert 'keywords' in topic, "Topic should have 'keywords'"
                
                # Check probability is valid
                assert 0 <= topic['probability'] <= 1, "Probability should be between 0 and 1"
                
                # Check keywords structure
                assert isinstance(topic['keywords'], list), "Keywords should be a list"
        
        print("✓ Topic clusters have correct structure")
    
    def test_grad_cam_structure(self, client, sample_text_data):
        """Test Grad-CAM structure when image is provided"""
        response = client.post('/api/explain',
                             json=sample_text_data,
                             content_type='application/json')
        
        data = json.loads(response.data)
        explanation = data['explanation']
        
        # Check if Grad-CAM is present (might be mock data)
        if 'grad_cam' in explanation:
            grad_cam = explanation['grad_cam']
            
            assert 'heatmap_available' in grad_cam, "Grad-CAM should indicate heatmap availability"
            assert 'explanation' in grad_cam, "Grad-CAM should have explanation text"
            
            if grad_cam['heatmap_available']:
                assert 'attention_regions' in grad_cam, "Should have attention regions when heatmap available"
                
                for region in grad_cam['attention_regions']:
                    assert 'region' in region, "Region should have name"
                    assert 'importance' in region, "Region should have importance score"
                    assert 0 <= region['importance'] <= 1, "Importance should be between 0 and 1"
        
        print("✓ Grad-CAM structure is valid")
    
    def test_explain_endpoint_error_handling(self, client):
        """Test error handling for invalid requests"""
        # Test with missing data
        response = client.post('/api/explain',
                             json={},
                             content_type='application/json')
        
        # Should handle gracefully (either 400 or return limited explanation)
        assert response.status_code in [200, 400]
        
        # Test with invalid JSON
        response = client.post('/api/explain',
                             data="invalid json",
                             content_type='application/json')
        
        assert response.status_code in [200, 400]
        
        print("✓ Error handling works correctly")
    
    def test_explain_performance(self, client, sample_text_data):
        """Test that explanation generation completes within reasonable time"""
        import time
        
        start_time = time.time()
        response = client.post('/api/explain',
                             json=sample_text_data,
                             content_type='application/json')
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Should complete within 10 seconds (generous for mock data)
        assert response_time < 10.0, f"Response took too long: {response_time:.2f}s"
        assert response.status_code == 200
        
        print(f"✓ Explanation generated in {response_time:.2f}s")
    
    def test_explain_with_different_text_lengths(self, client):
        """Test explanation with various text lengths"""
        test_cases = [
            {"text": "Short.", "image_url": ""},
            {"text": "Medium length text that should provide reasonable analysis for testing purposes.", "image_url": ""},
            {"text": "Very long text that contains multiple sentences and should provide comprehensive analysis. " * 10, "image_url": ""}
        ]
        
        for i, test_case in enumerate(test_cases):
            response = client.post('/api/explain',
                                 json=test_case,
                                 content_type='application/json')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'explanation' in data
            
            print(f"✓ Test case {i+1} handled correctly")
    
    def test_explainability_integration(self, client, sample_fake_data):
        """Test full integration with fake news detection"""
        # First, test detection
        detect_response = client.post('/api/detect',
                                    json=sample_fake_data,
                                    content_type='application/json')
        
        # Then test explanation
        explain_response = client.post('/api/explain',
                                     json=sample_fake_data,
                                     content_type='application/json')
        
        assert explain_response.status_code == 200
        
        explain_data = json.loads(explain_response.data)
        
        # Check that explanation provides insights
        explanation = explain_data['explanation']
        
        # At least one explanation method should provide data
        has_explanation = (
            (explanation.get('shap_values') and len(explanation['shap_values']) > 0) or
            (explanation.get('lime_explanation') and len(explanation['lime_explanation']) > 0) or
            (explanation.get('topic_clusters') and len(explanation['topic_clusters']) > 0)
        )
        
        assert has_explanation, "Should provide at least one type of explanation"
        
        print("✓ Full integration test passed")

def run_explainability_tests():
    """Run all explainability tests"""
    print("\n🧪 STARTING EXPLAINABILITY TESTS")
    print("=" * 50)
    
    # Create test instance
    test_instance = TestExplainabilityFeatures()
    
    # Create mock client
    app.config['TESTING'] = True
    client = app.test_client()
    
    # Sample data
    sample_text = {
        "text": "Scientists discover new exoplanet with potential for life",
        "image_url": "https://example.com/planet.jpg"
    }
    
    sample_fake = {
        "text": "BREAKING: Aliens control government! Shocking truth revealed!",
        "image_url": "https://example.com/alien.jpg"
    }
    
    tests_passed = 0
    total_tests = 0
    
    # Run individual tests
    test_methods = [
        ('Endpoint Exists', lambda: test_instance.test_explain_endpoint_exists(client)),
        ('Valid Data', lambda: test_instance.test_explain_endpoint_with_valid_data(client, sample_text)),
        ('SHAP Structure', lambda: test_instance.test_shap_values_structure(client, sample_text)),
        ('LIME Structure', lambda: test_instance.test_lime_explanation_structure(client, sample_text)),
        ('Topic Structure', lambda: test_instance.test_topic_clusters_structure(client, sample_text)),
        ('Grad-CAM Structure', lambda: test_instance.test_grad_cam_structure(client, sample_text)),
        ('Error Handling', lambda: test_instance.test_explain_endpoint_error_handling(client)),
        ('Performance', lambda: test_instance.test_explain_performance(client, sample_text)),
        ('Text Lengths', lambda: test_instance.test_explain_with_different_text_lengths(client)),
        ('Integration', lambda: test_instance.test_explainability_integration(client, sample_fake))
    ]
    
    for test_name, test_func in test_methods:
        total_tests += 1
        try:
            test_func()
            tests_passed += 1
            print(f"✅ {test_name} - PASSED")
        except Exception as e:
            print(f"❌ {test_name} - FAILED: {str(e)}")
    
    print("\n" + "=" * 50)
    print(f"🏁 EXPLAINABILITY TESTS COMPLETED")
    print(f"📊 Results: {tests_passed}/{total_tests} tests passed")
    print(f"📈 Success Rate: {(tests_passed/total_tests)*100:.1f}%")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = run_explainability_tests()
    sys.exit(0 if success else 1)