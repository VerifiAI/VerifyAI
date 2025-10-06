#!/usr/bin/env python3
"""
Dataset Analysis Script
Analyzes the Fakeddit dataset statistics for train/test/validation splits
"""

import pandas as pd
import os
import json
from pathlib import Path

def analyze_dataset():
    """Analyze dataset statistics"""
    data_dir = Path('data/processed')
    files = {
        'train': 'fakeddit_processed_train.parquet',
        'validation': 'fakeddit_processed_val.parquet', 
        'test': 'fakeddit_processed_test.parquet'
    }
    
    total_stats = {
        'dataset_name': 'Fakeddit',
        'dataset_type': 'Multimodal Fake News Detection',
        'splits': {},
        'total_samples': 0,
        'total_fake': 0,
        'total_real': 0
    }
    
    for split_name, filename in files.items():
        filepath = data_dir / filename
        if filepath.exists():
            print(f"\n=== {split_name.upper()} SPLIT ({filename}) ===")
            df = pd.read_parquet(filepath)
            
            # Basic statistics
            total_samples = len(df)
            columns = list(df.columns)
            
            print(f"Total samples: {total_samples:,}")
            print(f"Columns ({len(columns)}): {columns}")
            print(f"Shape: {df.shape}")
            
            # Label distribution
            label_dist = {}
            if 'label' in df.columns:
                label_counts = df['label'].value_counts().to_dict()
                label_dist = label_counts
                print(f"Label distribution: {label_counts}")
                
                # Calculate percentages
                for label, count in label_counts.items():
                    pct = (count / total_samples) * 100
                    label_name = 'Fake' if label == 1 else 'Real'
                    print(f"  {label_name} (label={label}): {count:,} ({pct:.1f}%)")
            
            # Text analysis
            text_stats = {}
            if 'clean_title' in df.columns:
                text_col = df['clean_title'].dropna()
                text_stats = {
                    'total_text_samples': len(text_col),
                    'avg_text_length': text_col.str.len().mean(),
                    'max_text_length': text_col.str.len().max(),
                    'min_text_length': text_col.str.len().min()
                }
                print(f"Text samples: {len(text_col):,}")
                print(f"Avg text length: {text_stats['avg_text_length']:.1f} chars")
            
            # Image analysis
            image_stats = {}
            if 'image_url' in df.columns:
                image_col = df['image_url'].dropna()
                image_stats = {
                    'total_image_samples': len(image_col),
                    'image_availability': len(image_col) / total_samples * 100
                }
                print(f"Image samples: {len(image_col):,} ({image_stats['image_availability']:.1f}%)")
            
            # Metadata analysis
            metadata_cols = [col for col in columns if col not in ['clean_title', 'image_url', 'label']]
            print(f"Metadata columns ({len(metadata_cols)}): {metadata_cols[:5]}{'...' if len(metadata_cols) > 5 else ''}")
            
            # Store split statistics
            total_stats['splits'][split_name] = {
                'total_samples': total_samples,
                'label_distribution': label_dist,
                'text_stats': text_stats,
                'image_stats': image_stats,
                'metadata_columns': len(metadata_cols),
                'all_columns': columns
            }
            
            # Add to totals
            total_stats['total_samples'] += total_samples
            if 'label' in df.columns:
                total_stats['total_fake'] += label_dist.get(1, 0)
                total_stats['total_real'] += label_dist.get(0, 0)
        else:
            print(f"\n❌ File not found: {filepath}")
    
    # Print overall statistics
    print("\n" + "="*60)
    print("OVERALL DATASET STATISTICS")
    print("="*60)
    print(f"Total samples across all splits: {total_stats['total_samples']:,}")
    print(f"Total fake news samples: {total_stats['total_fake']:,}")
    print(f"Total real news samples: {total_stats['total_real']:,}")
    
    if total_stats['total_samples'] > 0:
        fake_pct = (total_stats['total_fake'] / total_stats['total_samples']) * 100
        real_pct = (total_stats['total_real'] / total_stats['total_samples']) * 100
        print(f"Fake news percentage: {fake_pct:.1f}%")
        print(f"Real news percentage: {real_pct:.1f}%")
    
    # Split distribution
    print("\nSplit Distribution:")
    for split_name, split_data in total_stats['splits'].items():
        pct = (split_data['total_samples'] / total_stats['total_samples']) * 100
        print(f"  {split_name.capitalize()}: {split_data['total_samples']:,} ({pct:.1f}%)")
    
    # Save detailed statistics
    with open('dataset_statistics.json', 'w') as f:
        json.dump(total_stats, f, indent=2)
    
    print(f"\n📊 Detailed statistics saved to: dataset_statistics.json")
    return total_stats

if __name__ == '__main__':
    stats = analyze_dataset()