#!/bin/bash
# Complete Benchmarking Workflow Script
# This script demonstrates running all benchmarking code components

set -e  # Exit on error

echo "=================================================================="
echo "AI-ENE Benchmarking - Complete Workflow"
echo "=================================================================="
echo ""

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Step 1: Integration Tests
echo -e "${BLUE}Step 1: Running Integration Tests...${NC}"
python integration_test.py
echo -e "${GREEN}✓ Integration tests completed${NC}"
echo ""

# Step 2: Unit Tests
echo -e "${BLUE}Step 2: Running Unit Tests...${NC}"
python test_benchmark.py
echo -e "${GREEN}✓ Unit tests completed${NC}"
echo ""

# Step 3: Example Scripts
echo -e "${BLUE}Step 3: Running Example Scripts...${NC}"
python benchmark_examples.py
echo -e "${GREEN}✓ Example scripts completed${NC}"
echo ""

# Step 4: Demo Benchmark
echo -e "${BLUE}Step 4: Running Demo Benchmark...${NC}"
python run_demo_benchmark.py
echo -e "${GREEN}✓ Demo benchmark completed${NC}"
echo ""

# Summary
echo "=================================================================="
echo "WORKFLOW COMPLETE"
echo "=================================================================="
echo ""
echo "Generated Output Files:"
echo "  1. Integration test results (stdout)"
echo "  2. Unit test results (stdout)"
echo "  3. Example execution logs (stdout)"
echo "  4. Demo benchmark results:"
echo "     - demo_benchmark_results/benchmark_results_*.csv"
echo "     - demo_benchmark_results/benchmark_summary_*.json"
echo "     - demo_benchmark_results/leaderboard.html"
echo "     - example_results/comprehensive_cache/leaderboard_report_*.json"
echo ""
echo "Documentation:"
echo "  - BENCHMARK_README.md (Usage guide)"
echo "  - EXECUTION_SUMMARY.md (This execution summary)"
echo "  - benchmark_config.yaml (Configuration)"
echo ""
echo -e "${GREEN}All benchmarking code executed successfully!${NC}"
echo ""
