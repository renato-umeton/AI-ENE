#!/usr/bin/env python3
"""
Integration test to verify benchmarking functionality without requiring full dependencies.

This test validates the core logic and structure of the benchmarking system.
"""

import os
import sys
import json
from pathlib import Path

# Add ENE_inference to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        import benchmark_huggingface
        print("✓ benchmark_huggingface imported successfully")
    except Exception as e:
        print(f"✗ Failed to import benchmark_huggingface: {e}")
        return False
    
    try:
        import leaderboard_tracker
        print("✓ leaderboard_tracker imported successfully")
    except Exception as e:
        print(f"✗ Failed to import leaderboard_tracker: {e}")
        return False
    
    try:
        import run_benchmark
        print("✓ run_benchmark imported successfully")
    except Exception as e:
        print(f"✗ Failed to import run_benchmark: {e}")
        return False
    
    try:
        import benchmark_examples
        print("✓ benchmark_examples imported successfully")
    except Exception as e:
        print(f"✗ Failed to import benchmark_examples: {e}")
        return False
    
    return True


def test_config_file():
    """Test that configuration file exists and is valid."""
    print("\nTesting configuration file...")
    
    config_path = os.path.join(SCRIPT_DIR, "benchmark_config.yaml")
    
    if not os.path.exists(config_path):
        print(f"✗ Config file not found: {config_path}")
        return False
    
    print(f"✓ Config file exists: {config_path}")
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check for expected sections
        required_sections = ['huggingface_models', 'benchmark', 'dataset', 'models']
        for section in required_sections:
            if section in config:
                print(f"✓ Config section '{section}' found")
            else:
                print(f"✗ Config section '{section}' missing")
                return False
        
        return True
    except ImportError:
        print("⚠ PyYAML not installed, skipping config validation")
        return True
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return False


def test_file_structure():
    """Test that all expected files exist."""
    print("\nTesting file structure...")
    
    expected_files = [
        "benchmark_huggingface.py",
        "leaderboard_tracker.py",
        "run_benchmark.py",
        "benchmark_config.yaml",
        "benchmark_examples.py",
        "test_benchmark.py",
        "BENCHMARK_README.md",
    ]
    
    all_exist = True
    for filename in expected_files:
        filepath = os.path.join(SCRIPT_DIR, filename)
        if os.path.exists(filepath):
            print(f"✓ {filename} exists")
        else:
            print(f"✗ {filename} missing")
            all_exist = False
    
    return all_exist


def test_class_structure():
    """Test that expected classes are defined."""
    print("\nTesting class structure...")
    
    try:
        from benchmark_huggingface import HuggingFaceBenchmark
        print("✓ HuggingFaceBenchmark class found")
        
        # Check for expected methods
        expected_methods = [
            'compute_dice_score',
            'compute_iou',
            'compute_sensitivity_specificity',
            'benchmark_on_dataset',
            'generate_summary_report',
        ]
        
        for method in expected_methods:
            if hasattr(HuggingFaceBenchmark, method):
                print(f"  ✓ Method '{method}' exists")
            else:
                print(f"  ✗ Method '{method}' missing")
                return False
        
    except Exception as e:
        print(f"✗ Failed to import HuggingFaceBenchmark: {e}")
        return False
    
    try:
        from leaderboard_tracker import HuggingFaceLeaderboardTracker
        print("✓ HuggingFaceLeaderboardTracker class found")
        
        expected_methods = [
            'fetch_model_info',
            'search_medical_models',
            'get_top_models_by_task',
            'generate_leaderboard_report',
            'recommend_benchmark_models',
        ]
        
        for method in expected_methods:
            if hasattr(HuggingFaceLeaderboardTracker, method):
                print(f"  ✓ Method '{method}' exists")
            else:
                print(f"  ✗ Method '{method}' missing")
                return False
        
    except Exception as e:
        print(f"✗ Failed to import HuggingFaceLeaderboardTracker: {e}")
        return False
    
    return True


def test_readme():
    """Test that README exists and contains expected sections."""
    print("\nTesting README...")
    
    readme_path = os.path.join(SCRIPT_DIR, "BENCHMARK_README.md")
    
    if not os.path.exists(readme_path):
        print(f"✗ README not found: {readme_path}")
        return False
    
    print(f"✓ README exists: {readme_path}")
    
    with open(readme_path, 'r') as f:
        content = f.read()
    
    expected_sections = [
        "Quick Start",
        "Configuration",
        "Examples",
        "Testing",
        "Benchmark Metrics",
    ]
    
    for section in expected_sections:
        if section in content:
            print(f"✓ README section '{section}' found")
        else:
            print(f"✗ README section '{section}' missing")
            return False
    
    return True


def main():
    """Run all integration tests."""
    print("="*80)
    print("BENCHMARKING INTEGRATION TESTS")
    print("="*80)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Configuration File", test_config_file),
        ("Class Structure", test_class_structure),
        ("README Documentation", test_readme),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ Error in {test_name}: {e}")
            results[test_name] = False
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All integration tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
