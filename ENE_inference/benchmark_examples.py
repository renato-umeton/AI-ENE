#!/usr/bin/env python3
"""
Example usage of AI-ENE benchmarking against HuggingFace models.

This script demonstrates various benchmarking scenarios and use cases.
"""

import os
import sys
from pathlib import Path

# Add ENE_inference to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from benchmark_huggingface import HuggingFaceBenchmark
from leaderboard_tracker import HuggingFaceLeaderboardTracker


def example_basic_benchmark():
    """
    Example 1: Basic benchmark with default settings.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Benchmark")
    print("="*80 + "\n")
    
    benchmark = HuggingFaceBenchmark(
        output_dir="./example_results/basic"
    )
    
    # Note: Replace with actual data paths
    image_dir = "./data/test_images"
    seg_dir = "./data/test_segmentations"
    model_path = "./ene_model/0208-1531-1_DualNet.h5"
    
    if os.path.exists(image_dir) and os.path.exists(seg_dir):
        df_results = benchmark.benchmark_on_dataset(
            image_dir=image_dir,
            seg_dir=seg_dir,
            ai_ene_model_path=model_path,
            max_cases=5
        )
        
        summary = benchmark.generate_summary_report(df_results)
        print("\nBasic benchmark complete!")
    else:
        print(f"Data directories not found. Please provide valid paths.")
        print(f"Expected: {image_dir} and {seg_dir}")


def example_custom_models():
    """
    Example 2: Benchmark with custom HuggingFace models.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Custom HuggingFace Models")
    print("="*80 + "\n")
    
    # Specify custom models to benchmark against
    custom_models = [
        "microsoft/swinv2-base-patch4-window12-192-22k",
        "facebook/sam-vit-base",
    ]
    
    benchmark = HuggingFaceBenchmark(
        output_dir="./example_results/custom_models",
        hf_models=custom_models
    )
    
    print(f"Configured to benchmark against {len(custom_models)} models:")
    for model in custom_models:
        print(f"  - {model}")


def example_leaderboard_tracking():
    """
    Example 3: Track HuggingFace leaderboards and get recommendations.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Leaderboard Tracking")
    print("="*80 + "\n")
    
    tracker = HuggingFaceLeaderboardTracker(
        cache_dir="./example_results/leaderboard_cache"
    )
    
    # Get top models for medical image segmentation
    print("Fetching top models for image segmentation...")
    top_models = tracker.get_top_models_by_task(
        task="image-segmentation",
        top_n=5
    )
    
    print(f"\nTop {len(top_models)} models for image segmentation:")
    for idx, model_id in enumerate(top_models, 1):
        if model_id:
            print(f"{idx}. {model_id}")
            
            # Get detailed metrics
            metrics = tracker.get_model_metrics(model_id)
            if metrics:
                print(f"   Downloads: {metrics.get('downloads', 'N/A')}")
                print(f"   Likes: {metrics.get('likes', 'N/A')}")
    
    # Get recommendations for benchmarking
    print("\nGetting benchmark recommendations...")
    recommendations = tracker.recommend_benchmark_models(
        task="image-segmentation",
        min_downloads=500,
        max_models=3
    )
    
    print(f"\nRecommended models for benchmarking:")
    for idx, model_id in enumerate(recommendations, 1):
        print(f"{idx}. {model_id}")


def example_comprehensive_report():
    """
    Example 4: Generate comprehensive leaderboard report.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Comprehensive Leaderboard Report")
    print("="*80 + "\n")
    
    tracker = HuggingFaceLeaderboardTracker(
        cache_dir="./example_results/comprehensive_cache"
    )
    
    # Generate report for multiple tasks
    tasks = [
        "image-segmentation",
        "image-classification",
    ]
    
    print(f"Generating report for {len(tasks)} tasks...")
    report = tracker.generate_leaderboard_report(
        tasks=tasks,
        top_n=3
    )
    
    print("\nReport generated successfully!")
    print(f"Report timestamp: {report.get('generated_at')}")
    print(f"Tasks covered: {len(report.get('tasks', {}))}")


def example_performance_comparison():
    """
    Example 5: Compare AI-ENE performance metrics.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Performance Comparison")
    print("="*80 + "\n")
    
    # This example shows how metrics would be compared
    # In practice, this would use real benchmark data
    
    sample_metrics = {
        "AI-ENE": {
            "avg_inference_time": 2.5,
            "dice_score": 0.87,
            "success_rate": 0.95
        },
        "microsoft/swinv2-base": {
            "avg_inference_time": 3.2,
            "dice_score": 0.82,
            "success_rate": 0.90
        },
        "facebook/sam-vit-base": {
            "avg_inference_time": 4.1,
            "dice_score": 0.80,
            "success_rate": 0.88
        }
    }
    
    print("Sample Performance Comparison:")
    print("-" * 80)
    print(f"{'Model':<40} {'Inference Time':<20} {'Dice Score':<15} {'Success Rate'}")
    print("-" * 80)
    
    for model_name, metrics in sample_metrics.items():
        print(f"{model_name:<40} "
              f"{metrics['avg_inference_time']:<20.2f} "
              f"{metrics['dice_score']:<15.2f} "
              f"{metrics['success_rate']:.2%}")
    
    print("-" * 80)
    
    # Identify best model for each metric
    best_speed = min(sample_metrics.items(), key=lambda x: x[1]['avg_inference_time'])
    best_accuracy = max(sample_metrics.items(), key=lambda x: x[1]['dice_score'])
    best_reliability = max(sample_metrics.items(), key=lambda x: x[1]['success_rate'])
    
    print("\nBest Performing Models:")
    print(f"  Fastest: {best_speed[0]} ({best_speed[1]['avg_inference_time']:.2f}s)")
    print(f"  Most Accurate: {best_accuracy[0]} (Dice: {best_accuracy[1]['dice_score']:.2f})")
    print(f"  Most Reliable: {best_reliability[0]} ({best_reliability[1]['success_rate']:.2%})")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("AI-ENE BENCHMARKING EXAMPLES")
    print("="*80)
    
    examples = [
        ("Basic Benchmark", example_basic_benchmark),
        ("Custom Models", example_custom_models),
        ("Leaderboard Tracking", example_leaderboard_tracking),
        ("Comprehensive Report", example_comprehensive_report),
        ("Performance Comparison", example_performance_comparison),
    ]
    
    print("\nAvailable examples:")
    for idx, (name, _) in enumerate(examples, 1):
        print(f"{idx}. {name}")
    
    print("\nRunning examples...")
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\nError in {name}: {e}")
            print("Continuing with next example...")
    
    print("\n" + "="*80)
    print("EXAMPLES COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
