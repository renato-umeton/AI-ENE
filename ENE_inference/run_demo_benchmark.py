#!/usr/bin/env python3
"""
Demo script to generate sample benchmarking outputs.

This script creates mock data and runs the benchmarking code to demonstrate
the functionality and generate example output files.
"""

import os
import sys
import numpy as np
import SimpleITK as sitk
from pathlib import Path

# Add ENE_inference to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from benchmark_huggingface import HuggingFaceBenchmark


def create_mock_data(output_dir="./test_data"):
    """Create mock medical images and segmentations for testing."""
    print("Creating mock test data...")
    
    image_dir = os.path.join(output_dir, "images")
    seg_dir = os.path.join(output_dir, "segmentations")
    
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)
    
    # Create 3 sample cases
    for i in range(3):
        case_id = f"case_{i+1:03d}"
        
        # Create a simple 3D image (64x64x32)
        image_array = np.random.rand(32, 64, 64) * 100
        image = sitk.GetImageFromArray(image_array)
        image.SetSpacing([1.0, 1.0, 2.0])
        
        # Save image
        image_path = os.path.join(image_dir, f"{case_id}_0000.nii.gz")
        sitk.WriteImage(image, image_path)
        
        # Create a simple segmentation with some random blobs
        seg_array = np.zeros((32, 64, 64), dtype=np.uint8)
        seg_array[10:20, 20:40, 20:40] = 1  # Add a blob
        seg = sitk.GetImageFromArray(seg_array)
        seg.SetSpacing([1.0, 1.0, 2.0])
        
        # Save segmentation
        seg_path = os.path.join(seg_dir, f"{case_id}_seg.nii")
        sitk.WriteImage(seg, seg_path)
    
    print(f"✓ Created 3 mock test cases in {output_dir}")
    return image_dir, seg_dir


def create_mock_model():
    """Create a mock model file."""
    print("Creating mock AI-ENE model...")
    
    model_dir = os.path.join(SCRIPT_DIR, "ene_model")
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "demo_model.h5")
    
    # Create a simple mock model structure using h5py if available,
    # otherwise create a dummy file
    try:
        import h5py
        import tensorflow as tf
        from tensorflow import keras
        
        # Create a minimal model
        model = keras.Sequential([
            keras.layers.Input(shape=(64, 64, 32, 1)),
            keras.layers.Conv3D(16, (3, 3, 3), activation='relu'),
            keras.layers.GlobalAveragePooling3D(),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.save(model_path)
        print(f"✓ Created mock Keras model at {model_path}")
    except ImportError:
        # Create a dummy H5 file
        try:
            import h5py
            with h5py.File(model_path, 'w') as f:
                f.create_dataset('dummy', data=np.array([1, 2, 3]))
            print(f"✓ Created mock H5 file at {model_path}")
        except ImportError:
            # Just create an empty file
            with open(model_path, 'w') as f:
                f.write("mock model")
            print(f"✓ Created mock model file at {model_path}")
    
    return model_path


def run_demo_benchmark():
    """Run a demonstration benchmark with mock data."""
    print("\n" + "="*80)
    print("RUNNING DEMO BENCHMARK")
    print("="*80 + "\n")
    
    # Create mock data
    image_dir, seg_dir = create_mock_data()
    model_path = create_mock_model()
    
    # Create output directory
    output_dir = "./demo_benchmark_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nInitializing benchmark...")
    
    # Use a minimal set of models to avoid network issues
    # Since we can't download real models, we'll just demonstrate the framework
    benchmark = HuggingFaceBenchmark(
        output_dir=output_dir,
        hf_models=[]  # Empty list to skip HuggingFace model downloads
    )
    
    print("✓ Benchmark initialized")
    
    # Test metric computation functions
    print("\nTesting metric computation functions...")
    
    # Test Dice score
    pred = np.ones((10, 10), dtype=np.uint8)
    gt = np.ones((10, 10), dtype=np.uint8)
    dice = benchmark.compute_dice_score(pred, gt)
    print(f"✓ Dice score (perfect overlap): {dice:.4f}")
    
    # Test IoU
    iou = benchmark.compute_iou(pred, gt)
    print(f"✓ IoU score (perfect overlap): {iou:.4f}")
    
    # Test sensitivity/specificity
    sens, spec = benchmark.compute_sensitivity_specificity(pred, gt)
    print(f"✓ Sensitivity: {sens:.4f}, Specificity: {spec:.4f}")
    
    # Generate mock results manually since we can't run full inference
    print("\nGenerating mock benchmark results...")
    
    import pandas as pd
    from datetime import datetime
    
    results = []
    case_ids = ["case_001", "case_002", "case_003"]
    
    for case_id in case_ids:
        # AI-ENE results
        results.append({
            'case_id': case_id,
            'model': 'AI-ENE',
            'inference_time': np.random.uniform(1.5, 3.0),
            'success': True,
            'image_shape': '(32, 64, 64)',
            'seg_shape': '(32, 64, 64)',
        })
    
    df_results = pd.DataFrame(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"benchmark_results_{timestamp}.csv")
    df_results.to_csv(output_path, index=False)
    print(f"✓ Results saved to {output_path}")
    
    # Generate summary report
    print("\nGenerating summary report...")
    summary = benchmark.generate_summary_report(df_results)
    print("✓ Summary report generated")
    
    # Generate HTML leaderboard
    print("\nGenerating HTML leaderboard...")
    html_path = benchmark.generate_html_leaderboard(df_results, summary)
    print(f"✓ HTML leaderboard generated: {html_path}")
    
    print("\n" + "="*80)
    print("DEMO BENCHMARK COMPLETE")
    print("="*80)
    print(f"\nOutput files saved to: {output_dir}")
    print("\nGenerated files:")
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        size = os.path.getsize(file_path)
        print(f"  - {file} ({size} bytes)")
    print("\n")


if __name__ == "__main__":
    run_demo_benchmark()
