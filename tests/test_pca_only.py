#!/usr/bin/env python3
"""
Simple PCA Test Script for Debugging
Tests only the PCA fitting functionality
"""

import sys
import logging
from data_loader import FakeNewsDataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_pca_only():
    """Test only PCA fitting functionality"""
    print("Testing PCA fitting only...")
    
    try:
        # Initialize data loader
        data_loader = FakeNewsDataLoader()
        
        # Check if PCA model exists
        print(f"Has pca_model attribute: {hasattr(data_loader, 'pca_model')}")
        print(f"PCA model value: {getattr(data_loader, 'pca_model', 'NOT_FOUND')}")
        
        # Check if fit method exists
        print(f"Has fit_pca_on_training_data method: {hasattr(data_loader, 'fit_pca_on_training_data')}")
        
        # Create sample texts
        sample_texts = [
            "This is a sample news article about technology.",
            "Another article about politics and current events.",
            "Science news about recent discoveries.",
            "Sports news about the latest games.",
            "Entertainment news about movies and shows."
        ]
        
        print(f"Testing with {len(sample_texts)} sample texts")
        
        # Try to fit PCA
        if hasattr(data_loader, 'fit_pca_on_training_data'):
            print("Calling fit_pca_on_training_data...")
            data_loader.fit_pca_on_training_data(sample_texts)
            print("PCA fitting call completed")
            
            # Check PCA model after fitting
            print(f"PCA model after fitting: {data_loader.pca_model}")
            print(f"PCA model type: {type(data_loader.pca_model)}")
            
            if data_loader.pca_model is not None:
                pca = data_loader.pca_model
                print(f"PCA n_components: {getattr(pca, 'n_components_', 'NOT_FOUND')}")
                print(f"Has explained_variance_ratio_: {hasattr(pca, 'explained_variance_ratio_')}")
                
                if hasattr(pca, 'explained_variance_ratio_'):
                    print(f"Explained variance ratio shape: {pca.explained_variance_ratio_.shape}")
                    print(f"Total variance preserved: {pca.explained_variance_ratio_.sum():.3f}")
                    print("✓ PCA fitting successful!")
                    return True
                else:
                    print("✗ PCA model missing explained_variance_ratio_")
                    return False
            else:
                print("✗ PCA model is None after fitting")
                return False
        else:
            print("✗ fit_pca_on_training_data method not found")
            return False
            
    except Exception as e:
        print(f"✗ Error during PCA testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pca_only()
    print(f"\nPCA Test Result: {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)