#!/usr/bin/env python3
"""
Tests for HuggingFace benchmarking functionality.

This test suite validates the benchmarking and leaderboard tracking components.
"""

import os
import sys
import tempfile
import unittest
import numpy as np
from pathlib import Path

# Add ENE_inference to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from benchmark_huggingface import HuggingFaceBenchmark
from leaderboard_tracker import HuggingFaceLeaderboardTracker


class TestBenchmarkMetrics(unittest.TestCase):
    """Test metric computation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.benchmark = HuggingFaceBenchmark(output_dir=self.temp_dir)
    
    def test_dice_score_perfect_overlap(self):
        """Test Dice score with perfect overlap."""
        pred = np.ones((10, 10), dtype=np.uint8)
        gt = np.ones((10, 10), dtype=np.uint8)
        
        dice = self.benchmark.compute_dice_score(pred, gt)
        self.assertAlmostEqual(dice, 1.0, places=5)
    
    def test_dice_score_no_overlap(self):
        """Test Dice score with no overlap."""
        pred = np.zeros((10, 10), dtype=np.uint8)
        gt = np.ones((10, 10), dtype=np.uint8)
        
        dice = self.benchmark.compute_dice_score(pred, gt)
        self.assertLess(dice, 0.1)
    
    def test_dice_score_partial_overlap(self):
        """Test Dice score with partial overlap."""
        pred = np.zeros((10, 10), dtype=np.uint8)
        pred[:5, :] = 1  # Half filled
        
        gt = np.zeros((10, 10), dtype=np.uint8)
        gt[:5, :] = 1  # Same half filled
        
        dice = self.benchmark.compute_dice_score(pred, gt)
        self.assertAlmostEqual(dice, 1.0, places=5)
    
    def test_iou_perfect_overlap(self):
        """Test IoU with perfect overlap."""
        pred = np.ones((10, 10), dtype=np.uint8)
        gt = np.ones((10, 10), dtype=np.uint8)
        
        iou = self.benchmark.compute_iou(pred, gt)
        self.assertAlmostEqual(iou, 1.0, places=5)
    
    def test_iou_no_overlap(self):
        """Test IoU with no overlap."""
        pred = np.zeros((10, 10), dtype=np.uint8)
        gt = np.ones((10, 10), dtype=np.uint8)
        
        iou = self.benchmark.compute_iou(pred, gt)
        self.assertLess(iou, 0.1)
    
    def test_sensitivity_specificity(self):
        """Test sensitivity and specificity computation."""
        # Test case 1: Perfect positive prediction (all true positives)
        # When both pred and gt are all ones, we have:
        # TP = 100 (all predicted positives match ground truth)
        # TN = 0 (no true negatives)
        # FP = 0 (no false positives)
        # FN = 0 (no false negatives)
        # Sensitivity = TP / (TP + FN) = 100 / 100 = 1.0
        pred = np.ones((10, 10), dtype=np.uint8)
        gt = np.ones((10, 10), dtype=np.uint8)
        
        sens, spec = self.benchmark.compute_sensitivity_specificity(pred, gt)
        self.assertAlmostEqual(sens, 1.0, places=5)
        
        # Test case 2: Perfect negative prediction (all true negatives)
        # When both pred and gt are all zeros, we have:
        # TP = 0 (no true positives)
        # TN = 100 (all predicted negatives match ground truth)
        # FP = 0 (no false positives)
        # FN = 0 (no false negatives)
        # Specificity = TN / (TN + FP) = 100 / 100 = 1.0
        pred = np.zeros((10, 10), dtype=np.uint8)
        gt = np.zeros((10, 10), dtype=np.uint8)
        
        sens, spec = self.benchmark.compute_sensitivity_specificity(pred, gt)
        self.assertAlmostEqual(spec, 1.0, places=5)


class TestLeaderboardTracker(unittest.TestCase):
    """Test leaderboard tracking functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = HuggingFaceLeaderboardTracker(cache_dir=self.temp_dir)
    
    def test_initialization(self):
        """Test tracker initialization."""
        self.assertTrue(os.path.exists(self.temp_dir))
        self.assertEqual(self.tracker.cache_dir, self.temp_dir)
    
    def test_cache_directory_creation(self):
        """Test that cache directory is created."""
        self.assertTrue(os.path.isdir(self.tracker.cache_dir))
    
    def test_medical_tasks_defined(self):
        """Test that medical tasks are defined."""
        self.assertIsInstance(self.tracker.medical_tasks, list)
        self.assertGreater(len(self.tracker.medical_tasks), 0)
        self.assertIn("image-segmentation", self.tracker.medical_tasks)


class TestBenchmarkInitialization(unittest.TestCase):
    """Test benchmark initialization and configuration."""
    
    def test_default_initialization(self):
        """Test benchmark with default settings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark = HuggingFaceBenchmark(output_dir=temp_dir)
            self.assertTrue(os.path.exists(temp_dir))
            self.assertIsInstance(benchmark.hf_models, list)
            self.assertGreater(len(benchmark.hf_models), 0)
    
    def test_custom_models(self):
        """Test benchmark with custom model list."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_models = ["model1", "model2"]
            benchmark = HuggingFaceBenchmark(
                output_dir=temp_dir,
                hf_models=custom_models
            )
            self.assertEqual(benchmark.hf_models, custom_models)
    
    def test_results_initialization(self):
        """Test that results list is initialized."""
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark = HuggingFaceBenchmark(output_dir=temp_dir)
            self.assertIsInstance(benchmark.results, list)
            self.assertEqual(len(benchmark.results), 0)


class TestMetricValidation(unittest.TestCase):
    """Test metric validation and edge cases."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.benchmark = HuggingFaceBenchmark(output_dir=self.temp_dir)
    
    def test_dice_score_with_zeros(self):
        """Test Dice score when both masks are empty."""
        pred = np.zeros((10, 10), dtype=np.uint8)
        gt = np.zeros((10, 10), dtype=np.uint8)
        
        # Should handle gracefully with smoothing factor
        dice = self.benchmark.compute_dice_score(pred, gt)
        self.assertIsInstance(dice, float)
        self.assertTrue(0 <= dice <= 1)
    
    def test_iou_with_zeros(self):
        """Test IoU when both masks are empty."""
        pred = np.zeros((10, 10), dtype=np.uint8)
        gt = np.zeros((10, 10), dtype=np.uint8)
        
        # Should handle gracefully with smoothing factor
        iou = self.benchmark.compute_iou(pred, gt)
        self.assertIsInstance(iou, float)
        self.assertTrue(0 <= iou <= 1)
    
    def test_metrics_with_3d_arrays(self):
        """Test metrics with 3D arrays."""
        pred = np.random.randint(0, 2, (10, 10, 10), dtype=np.uint8)
        gt = np.random.randint(0, 2, (10, 10, 10), dtype=np.uint8)
        
        dice = self.benchmark.compute_dice_score(pred, gt)
        self.assertIsInstance(dice, float)
        self.assertTrue(0 <= dice <= 1)
        
        iou = self.benchmark.compute_iou(pred, gt)
        self.assertIsInstance(iou, float)
        self.assertTrue(0 <= iou <= 1)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestBenchmarkMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestLeaderboardTracker))
    suite.addTests(loader.loadTestsFromTestCase(TestBenchmarkInitialization))
    suite.addTests(loader.loadTestsFromTestCase(TestMetricValidation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
