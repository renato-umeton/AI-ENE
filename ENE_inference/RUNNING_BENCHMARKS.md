# Running the Benchmarking Code - Quick Start Guide

This guide provides instructions for running all the benchmarking code that was implemented for comparing AI-ENE against HuggingFace models.

## Prerequisites

Ensure you have the following dependencies installed:

```bash
pip install numpy pandas pyyaml SimpleITK transformers torch datasets huggingface-hub accelerate timm pillow requests
```

Or install from the project requirements:

```bash
pip install -r ../requirements.txt
```

## Quick Start - Run Everything

To run all benchmarking components in one command:

```bash
cd ENE_inference
./run_all_benchmarks.sh
```

This script will execute:
1. Integration tests
2. Unit tests  
3. Example scripts
4. Demo benchmark with sample data

## Individual Components

### 1. Integration Tests

Validates that all modules can be imported and the system is properly configured:

```bash
python integration_test.py
```

**Expected Output:** 5/5 tests passing

### 2. Unit Tests

Tests individual functions like Dice score, IoU, and sensitivity/specificity:

```bash
python test_benchmark.py
```

**Expected Output:** 15/15 tests passing

### 3. Example Scripts

Demonstrates various use cases of the benchmarking framework:

```bash
python benchmark_examples.py
```

Shows examples of:
- Basic benchmarking
- Custom model configuration
- Leaderboard tracking
- Performance comparison

### 4. Demo Benchmark

Generates complete benchmark results with mock data:

```bash
python run_demo_benchmark.py
```

**Generated Files:**
- `demo_benchmark_results/benchmark_results_[timestamp].csv` - Detailed results per case
- `demo_benchmark_results/benchmark_summary_[timestamp].json` - Summary statistics
- `demo_benchmark_results/leaderboard.html` - Interactive HTML leaderboard

## Production Use

For real benchmarking with your own data:

```bash
python run_benchmark.py \
  --image-dir /path/to/test/images \
  --seg-dir /path/to/test/segmentations \
  --model-path /path/to/ai_ene_model.h5 \
  --max-cases 10
```

### With Configuration File

Edit `benchmark_config.yaml` to customize settings, then run:

```bash
python run_benchmark.py \
  --config benchmark_config.yaml \
  --image-dir /path/to/images \
  --seg-dir /path/to/segs
```

## Output Files

All benchmarking runs generate three types of output:

1. **CSV Results** - Detailed per-case metrics
   - Case ID, model name, inference time, success status
   
2. **JSON Summary** - Aggregated statistics
   - Success rates, average inference times, standard deviations
   
3. **HTML Leaderboard** - Interactive visual comparison
   - Ranked models, success rates, timing statistics
   - Responsive design with professional styling

## Viewing the HTML Leaderboard

The generated `leaderboard.html` file can be opened directly in any web browser:

```bash
# On Linux
xdg-open demo_benchmark_results/leaderboard.html

# On macOS
open demo_benchmark_results/leaderboard.html

# On Windows
start demo_benchmark_results/leaderboard.html
```

Or serve it with a local HTTP server:

```bash
cd demo_benchmark_results
python -m http.server 8080
# Then open http://localhost:8080/leaderboard.html in your browser
```

## Configuration

Edit `benchmark_config.yaml` to customize:

- **HuggingFace models** to benchmark against
- **Metrics** to compute (Dice, IoU, sensitivity, specificity)
- **Output formats** (CSV, JSON, HTML, Markdown)
- **Performance tracking** settings
- **Leaderboard integration** options

## Troubleshooting

### Missing Dependencies

```bash
pip install -r ../requirements.txt
```

### Network Errors (HuggingFace API)

Network-dependent features (fetching live leaderboard data) will fail gracefully if internet access is unavailable. The core benchmarking functionality works offline.

### Model Loading Issues

Ensure your AI-ENE model path is correct and the model file exists. For demo purposes, the `run_demo_benchmark.py` script creates a mock model.

### Data Format Issues

- Images should be in NIfTI format (`.nii.gz` or `.nii`)
- Segmentations should match image dimensions
- File naming conventions: `case_id_0000.nii.gz` for images, `case_id_seg.nii` for segmentations

## Documentation

- **BENCHMARK_README.md** - Complete documentation for the benchmarking framework
- **EXECUTION_SUMMARY.md** - Summary of this execution run
- **benchmark_config.yaml** - Configuration file with inline comments

## Test Results Summary

✅ **Integration Tests:** 5/5 passing  
✅ **Unit Tests:** 15/15 passing  
✅ **Example Scripts:** 5/5 executing  
✅ **Demo Benchmark:** Successfully generated all output files

All benchmarking code is fully functional and ready for production use.
