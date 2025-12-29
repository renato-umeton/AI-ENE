#!/usr/bin/env python3
"""
Run AI-ENE benchmarking with HuggingFace models.

This script provides a convenient interface to run benchmarks comparing
AI-ENE performance against HuggingFace leaderboard models.
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

# Add ENE_inference to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from benchmark_huggingface import HuggingFaceBenchmark


def load_config(config_path: str) -> dict:
    """Load benchmark configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main entry point for benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Run AI-ENE benchmarking against HuggingFace models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run benchmark with default config
  python run_benchmark.py --image-dir ./data/images --seg-dir ./data/segs
  
  # Run with custom config
  python run_benchmark.py --config my_config.yaml --image-dir ./data/images --seg-dir ./data/segs
  
  # Run with specific HuggingFace models
  python run_benchmark.py --image-dir ./data/images --seg-dir ./data/segs \\\\
    --hf-models microsoft/swinv2-base-patch4-window12-192-22k facebook/sam-vit-base
  
  # Limit to 5 test cases
  python run_benchmark.py --image-dir ./data/images --seg-dir ./data/segs --max-cases 5
        """
    )
    
    parser.add_argument(
        "--config",
        default=os.path.join(SCRIPT_DIR, "benchmark_config.yaml"),
        help="Path to benchmark configuration YAML file (default: benchmark_config.yaml)"
    )
    parser.add_argument(
        "--image-dir",
        help="Directory containing test images (overrides config)"
    )
    parser.add_argument(
        "--seg-dir",
        help="Directory containing segmentations (overrides config)"
    )
    parser.add_argument(
        "--model-path",
        help="Path to AI-ENE model (overrides config)"
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save benchmark results (overrides config)"
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        help="Maximum number of cases to process (overrides config)"
    )
    parser.add_argument(
        "--hf-models",
        nargs="+",
        help="List of HuggingFace model IDs to benchmark (overrides config)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        print(f"Loading configuration from {args.config}")
        config = load_config(args.config)
    else:
        print(f"Warning: Config file {args.config} not found, using defaults")
        config = {}
    
    # Override config with command line arguments
    image_dir = args.image_dir or config.get('dataset', {}).get('image_dir')
    seg_dir = args.seg_dir or config.get('dataset', {}).get('seg_dir')
    model_path = args.model_path or config.get('models', {}).get('ai_ene_model', './ene_model/0208-1531-1_DualNet.h5')
    output_dir = args.output_dir or config.get('models', {}).get('output_dir', './benchmark_results')
    max_cases = args.max_cases or config.get('benchmark', {}).get('max_cases')
    
    # Get HuggingFace models list
    if args.hf_models:
        hf_models = args.hf_models
    else:
        # Extract from config
        hf_models_config = config.get('huggingface_models', [])
        hf_models = [m['model_id'] for m in hf_models_config if 'model_id' in m]
    
    # Validate required arguments
    if not image_dir or not seg_dir:
        print("Error: --image-dir and --seg-dir are required", file=sys.stderr)
        print("Provide them via command line or in the config file", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(image_dir):
        print(f"Error: Image directory not found: {image_dir}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(seg_dir):
        print(f"Error: Segmentation directory not found: {seg_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Print configuration summary
    print("\n" + "="*80)
    print("AI-ENE BENCHMARK CONFIGURATION")
    print("="*80)
    print(f"Image Directory:     {image_dir}")
    print(f"Segmentation Dir:    {seg_dir}")
    print(f"AI-ENE Model:        {model_path}")
    print(f"Output Directory:    {output_dir}")
    print(f"Max Cases:           {max_cases if max_cases else 'All'}")
    print(f"HuggingFace Models:  {len(hf_models)}")
    for model in hf_models:
        print(f"  - {model}")
    print("="*80 + "\n")
    
    # Create benchmark instance
    benchmark = HuggingFaceBenchmark(
        output_dir=output_dir,
        hf_models=hf_models
    )
    
    try:
        # Run benchmark
        print("Starting benchmark...")
        df_results = benchmark.benchmark_on_dataset(
            image_dir=image_dir,
            seg_dir=seg_dir,
            ai_ene_model_path=model_path,
            max_cases=max_cases
        )
        
        # Generate summary
        print("\nGenerating summary report...")
        summary = benchmark.generate_summary_report(df_results)
        
        # Generate HTML leaderboard
        print("\nGenerating HTML leaderboard...")
        html_path = benchmark.generate_html_leaderboard(df_results, summary)
        
        print("\n" + "="*80)
        print("BENCHMARKING COMPLETE!")
        print("="*80)
        print(f"Results saved to: {output_dir}")
        print(f"HTML leaderboard: {html_path}")
        print("="*80 + "\n")
        
        return 0
    
    except Exception as e:
        print(f"\nError during benchmarking: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
