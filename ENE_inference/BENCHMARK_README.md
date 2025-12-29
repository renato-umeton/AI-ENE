# AI-ENE Benchmarking

This directory contains tools for benchmarking AI-ENE against state-of-the-art HuggingFace models.

## Files

- `benchmark_huggingface.py` - Core benchmarking module
- `leaderboard_tracker.py` - HuggingFace leaderboard tracking
- `run_benchmark.py` - CLI script to run benchmarks
- `benchmark_config.yaml` - Configuration file for benchmark settings
- `benchmark_examples.py` - Example usage scripts
- `test_benchmark.py` - Unit tests for benchmarking functionality

## Quick Start

### 1. Install Dependencies

```bash
pip install transformers torch datasets huggingface-hub accelerate timm pillow requests
```

Or install all AI-ENE dependencies:
```bash
pip install -r ../requirements.txt
```

### 2. Run Basic Benchmark

```bash
python run_benchmark.py \
  --image-dir /path/to/test/images \
  --seg-dir /path/to/test/segmentations \
  --max-cases 5
```

### 3. Track HuggingFace Leaderboards

```bash
python leaderboard_tracker.py
```

## Configuration

Edit `benchmark_config.yaml` to customize:

- HuggingFace models to benchmark against
- Metrics to compute
- Output formats
- Leaderboard tracking settings

## Examples

Run example scripts to see various use cases:

```bash
python benchmark_examples.py
```

## Testing

Run unit tests:

```bash
python test_benchmark.py
```

## Benchmark Metrics

The following metrics are computed:

- **Dice Score**: Overlap between predictions and ground truth
- **IoU (Intersection over Union)**: Segmentation accuracy
- **Sensitivity/Specificity**: Classification performance
- **Inference Time**: Speed comparison across models
- **Success Rate**: Percentage of successful inferences

## Output

Benchmark results are saved in `benchmark_results/` (configurable):

- `benchmark_results_[timestamp].csv` - Detailed per-case results
- `benchmark_summary_[timestamp].json` - Summary statistics

## HuggingFace Models

Default models for benchmarking (medical imaging relevant):

1. **microsoft/swinv2-base-patch4-window12-192-22k** - Swin Transformer V2
2. **nvidia/mit-b0** - SegFormer for semantic segmentation
3. **facebook/sam-vit-base** - Segment Anything Model

You can specify custom models via CLI or config file.

## Leaderboard Tracking

The leaderboard tracker fetches live data from HuggingFace to:

- Identify top-performing models
- Track model popularity (downloads, likes)
- Get recommendations for benchmarking
- Generate comprehensive reports

## Advanced Usage

### Custom Models

```bash
python run_benchmark.py \
  --image-dir ./data/images \
  --seg-dir ./data/segs \
  --hf-models microsoft/swinv2-base facebook/sam-vit-base
```

### With Configuration File

```bash
python run_benchmark.py \
  --config my_config.yaml \
  --image-dir ./data/images \
  --seg-dir ./data/segs
```

### Generate Leaderboard Report

```python
from leaderboard_tracker import HuggingFaceLeaderboardTracker

tracker = HuggingFaceLeaderboardTracker()
report = tracker.generate_leaderboard_report(top_n=10)
```

## API Usage

### Benchmark API

```python
from benchmark_huggingface import HuggingFaceBenchmark

benchmark = HuggingFaceBenchmark(
    output_dir="./results",
    hf_models=["microsoft/swinv2-base"]
)

results = benchmark.benchmark_on_dataset(
    image_dir="./images",
    seg_dir="./segs",
    ai_ene_model_path="./model.h5",
    max_cases=10
)

summary = benchmark.generate_summary_report(results)
```

### Leaderboard API

```python
from leaderboard_tracker import HuggingFaceLeaderboardTracker

tracker = HuggingFaceLeaderboardTracker()

# Get top models
top_models = tracker.get_top_models_by_task("image-segmentation", top_n=5)

# Get recommendations
recommended = tracker.recommend_benchmark_models(
    task="image-segmentation",
    min_downloads=1000,
    max_models=5
)
```

## Notes

- Benchmarking requires internet connection to download HuggingFace models
- First run may take longer as models are downloaded and cached
- GPU is recommended for faster inference
- Results are cached to avoid redundant API calls

## Troubleshooting

### Import Errors

If you get import errors, ensure all dependencies are installed:
```bash
pip install -r ../requirements.txt
```

### Model Download Issues

If models fail to download:
- Check internet connection
- Verify HuggingFace model IDs are correct
- Try downloading models manually first

### Out of Memory

If you run out of memory:
- Reduce `max_cases` parameter
- Use CPU instead of GPU
- Try smaller models

## Citation

If you use this benchmarking tool in your research, please cite the AI-ENE paper (see main README).
