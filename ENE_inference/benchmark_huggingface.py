#!/usr/bin/env python3
"""
Benchmark AI-ENE against HuggingFace medical imaging models.

This module provides functionality to compare the AI-ENE model's performance
against state-of-the-art models from HuggingFace's medical imaging leaderboards.
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import SimpleITK as sitk

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


class HuggingFaceBenchmark:
    """
    Benchmark AI-ENE against HuggingFace models for medical image segmentation.
    
    This class provides methods to:
    - Load and run HuggingFace models
    - Compare segmentation/classification performance
    - Generate benchmark reports
    """
    
    def __init__(
        self,
        output_dir: str = "./benchmark_results",
        hf_models: Optional[List[str]] = None
    ):
        """
        Initialize the benchmark.
        
        Args:
            output_dir: Directory to save benchmark results
            hf_models: List of HuggingFace model IDs to benchmark against
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Default HuggingFace models for benchmarking
        # NOTE: These are general-purpose vision models, not specifically trained
        # for medical imaging. They serve as baseline comparisons. For production
        # benchmarking, consider using models fine-tuned on medical imaging datasets
        # or configure custom models via the benchmark_config.yaml file.
        self.hf_models = hf_models or [
            "microsoft/swinv2-base-patch4-window12-192-22k",  # General purpose vision transformer
            "nvidia/mit-b0",  # SegFormer base model for semantic segmentation
            "facebook/sam-vit-base",  # Segment Anything Model - universal segmentation
        ]
        
        self.results = []
        
    def load_hf_model(self, model_id: str) -> Optional[Any]:
        """
        Load a HuggingFace model for inference.
        
        Args:
            model_id: HuggingFace model identifier
            
        Returns:
            Loaded model or None if loading fails
        """
        try:
            from transformers import AutoImageProcessor, AutoModel
            
            logging.info(f"Loading HuggingFace model: {model_id}")
            processor = AutoImageProcessor.from_pretrained(model_id)
            model = AutoModel.from_pretrained(model_id)
            
            logging.info(f"Successfully loaded {model_id}")
            return {"processor": processor, "model": model, "id": model_id}
        except Exception as e:
            logging.error(f"Failed to load HuggingFace model {model_id}: {e}")
            return None
    
    def compute_dice_score(
        self,
        pred: np.ndarray,
        ground_truth: np.ndarray,
        smooth: float = 1e-6
    ) -> float:
        """
        Compute Dice coefficient between prediction and ground truth.
        
        Args:
            pred: Binary prediction mask
            ground_truth: Binary ground truth mask
            smooth: Smoothing factor to avoid division by zero
            
        Returns:
            Dice coefficient (0 to 1)
        """
        pred_flat = pred.flatten()
        gt_flat = ground_truth.flatten()
        
        intersection = np.sum(pred_flat * gt_flat)
        union = np.sum(pred_flat) + np.sum(gt_flat)
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return float(dice)
    
    def compute_iou(
        self,
        pred: np.ndarray,
        ground_truth: np.ndarray,
        smooth: float = 1e-6
    ) -> float:
        """
        Compute Intersection over Union (IoU).
        
        Args:
            pred: Binary prediction mask
            ground_truth: Binary ground truth mask
            smooth: Smoothing factor
            
        Returns:
            IoU score (0 to 1)
        """
        pred_flat = pred.flatten()
        gt_flat = ground_truth.flatten()
        
        intersection = np.sum(pred_flat * gt_flat)
        union = np.sum(pred_flat) + np.sum(gt_flat) - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        return float(iou)
    
    def compute_sensitivity_specificity(
        self,
        pred: np.ndarray,
        ground_truth: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute sensitivity (recall) and specificity.
        
        Args:
            pred: Binary prediction mask
            ground_truth: Binary ground truth mask
            
        Returns:
            Tuple of (sensitivity, specificity)
        """
        pred_flat = pred.flatten().astype(bool)
        gt_flat = ground_truth.flatten().astype(bool)
        
        tp = np.sum(pred_flat & gt_flat)
        tn = np.sum(~pred_flat & ~gt_flat)
        fp = np.sum(pred_flat & ~gt_flat)
        fn = np.sum(~pred_flat & gt_flat)
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return float(sensitivity), float(specificity)
    
    def run_ai_ene_inference(
        self,
        image_path: str,
        seg_path: str,
        model_path: str
    ) -> Dict[str, Any]:
        """
        Run AI-ENE inference on a single case.
        
        NOTE: This is a simplified demonstration. For full AI-ENE inference,
        the complete pipeline from ene_classification.py should be used,
        including node splitting, cropping, preprocessing, and prediction.
        
        This method primarily tracks inference timing and validates data loading.
        For production benchmarking, integrate with the full ENE classification
        pipeline or implement complete preprocessing and inference logic.
        
        Args:
            image_path: Path to input image
            seg_path: Path to segmentation
            model_path: Path to AI-ENE model
            
        Returns:
            Dictionary with predictions and timing
        """
        try:
            from tensorflow.keras.models import load_model
            import tensorflow as tf
            
            # Suppress TF warnings
            tf.get_logger().setLevel('ERROR')
            
            start_time = time.time()
            
            # Load model
            model = load_model(model_path)
            
            # Load image and segmentation
            image = sitk.ReadImage(image_path)
            seg = sitk.ReadImage(seg_path)
            
            image_array = sitk.GetArrayFromImage(image)
            seg_array = sitk.GetArrayFromImage(seg)
            
            # NOTE: Full AI-ENE inference would include:
            # 1. Node splitting via watershed
            # 2. Per-node cropping and preprocessing
            # 3. Normalization and dilation
            # 4. Dual-input model prediction (main + small crops)
            # 5. ENE classification and volume calculation
            #
            # This simplified version measures loading/model time only
            # For accurate benchmarking, use the complete ene_classification.py pipeline
            
            inference_time = time.time() - start_time
            
            return {
                "model": "AI-ENE",
                "inference_time": inference_time,
                "success": True,
                "image_shape": image_array.shape,
                "seg_shape": seg_array.shape,
                "note": "Simplified inference - use full pipeline for production benchmarking"
            }
        except Exception as e:
            logging.error(f"AI-ENE inference failed: {e}")
            return {
                "model": "AI-ENE",
                "success": False,
                "error": str(e)
            }
    
    def run_hf_model_inference(
        self,
        model_info: Dict[str, Any],
        image_array: np.ndarray
    ) -> Dict[str, Any]:
        """
        Run HuggingFace model inference.
        
        Args:
            model_info: Model information including processor and model
            image_array: Input image as numpy array
            
        Returns:
            Dictionary with predictions and timing
        """
        try:
            import torch
            
            start_time = time.time()
            
            processor = model_info["processor"]
            model = model_info["model"]
            model_id = model_info["id"]
            
            # Convert to format expected by processor (3D -> 2D slice for demo)
            if image_array.ndim == 3:
                # Use middle slice for 2D models
                slice_2d = image_array[image_array.shape[0] // 2, :, :]
            else:
                slice_2d = image_array
            
            # Normalize to 0-255 range for vision models
            slice_2d = ((slice_2d - slice_2d.min()) / 
                       (slice_2d.max() - slice_2d.min() + 1e-8) * 255)
            slice_2d = slice_2d.astype(np.uint8)
            
            # Convert grayscale to RGB (most vision models expect 3 channels)
            if slice_2d.ndim == 2:
                slice_rgb = np.stack([slice_2d] * 3, axis=-1)
            else:
                slice_rgb = slice_2d
            
            # Process and run inference
            inputs = processor(images=slice_rgb, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            inference_time = time.time() - start_time
            
            return {
                "model": model_id,
                "inference_time": inference_time,
                "success": True,
                "output_shape": str(outputs.last_hidden_state.shape) if hasattr(outputs, 'last_hidden_state') else "N/A"
            }
        except Exception as e:
            logging.error(f"HuggingFace model {model_info['id']} inference failed: {e}")
            return {
                "model": model_info["id"],
                "success": False,
                "error": str(e)
            }
    
    def benchmark_on_dataset(
        self,
        image_dir: str,
        seg_dir: str,
        ai_ene_model_path: str,
        max_cases: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Run benchmark on a dataset.
        
        Args:
            image_dir: Directory containing images
            seg_dir: Directory containing segmentations
            ai_ene_model_path: Path to AI-ENE model
            max_cases: Maximum number of cases to process
            
        Returns:
            DataFrame with benchmark results
        """
        logging.info("Starting benchmark...")
        
        # Find all cases
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii.gz')])
        if max_cases:
            image_files = image_files[:max_cases]
        
        logging.info(f"Found {len(image_files)} cases to process")
        
        # Load HuggingFace models
        hf_model_objects = []
        for model_id in self.hf_models:
            model_obj = self.load_hf_model(model_id)
            if model_obj:
                hf_model_objects.append(model_obj)
        
        results = []
        
        for idx, image_file in enumerate(image_files):
            case_id = image_file.replace('_0000.nii.gz', '').replace('.nii.gz', '')
            logging.info(f"Processing case {idx+1}/{len(image_files)}: {case_id}")
            
            image_path = os.path.join(image_dir, image_file)
            
            # Find corresponding segmentation
            seg_file = None
            for potential_seg in os.listdir(seg_dir):
                if case_id in potential_seg and potential_seg.endswith('.nii'):
                    seg_file = potential_seg
                    break
            
            if not seg_file:
                logging.warning(f"No segmentation found for {case_id}, skipping")
                continue
            
            seg_path = os.path.join(seg_dir, seg_file)
            
            # Run AI-ENE inference
            ai_ene_result = self.run_ai_ene_inference(
                image_path, seg_path, ai_ene_model_path
            )
            ai_ene_result['case_id'] = case_id
            results.append(ai_ene_result)
            
            # Run HuggingFace model inference
            try:
                image = sitk.ReadImage(image_path)
                image_array = sitk.GetArrayFromImage(image)
                
                for hf_model in hf_model_objects:
                    hf_result = self.run_hf_model_inference(hf_model, image_array)
                    hf_result['case_id'] = case_id
                    results.append(hf_result)
            except Exception as e:
                logging.error(f"Failed to process case {case_id}: {e}")
        
        # Create DataFrame
        df_results = pd.DataFrame(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"benchmark_results_{timestamp}.csv")
        df_results.to_csv(output_path, index=False)
        logging.info(f"Results saved to {output_path}")
        
        return df_results
    
    def generate_summary_report(self, df_results: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics from benchmark results.
        
        Args:
            df_results: DataFrame with benchmark results
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {}
        
        # Group by model
        for model_name in df_results['model'].unique():
            model_data = df_results[df_results['model'] == model_name]
            
            successful = model_data[model_data['success'] == True]
            
            summary[model_name] = {
                'total_cases': len(model_data),
                'successful_cases': len(successful),
                'success_rate': len(successful) / len(model_data) if len(model_data) > 0 else 0,
                'avg_inference_time': successful['inference_time'].mean() if len(successful) > 0 else None,
                'std_inference_time': successful['inference_time'].std() if len(successful) > 0 else None,
                'min_inference_time': successful['inference_time'].min() if len(successful) > 0 else None,
                'max_inference_time': successful['inference_time'].max() if len(successful) > 0 else None,
            }
        
        # Save summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(self.output_dir, f"benchmark_summary_{timestamp}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logging.info(f"Summary saved to {summary_path}")
        
        # Print summary
        logging.info("\n" + "="*80)
        logging.info("BENCHMARK SUMMARY")
        logging.info("="*80)
        for model_name, stats in summary.items():
            logging.info(f"\nModel: {model_name}")
            logging.info(f"  Success Rate: {stats['success_rate']:.2%}")
            if stats['avg_inference_time'] is not None:
                logging.info(f"  Avg Inference Time: {stats['avg_inference_time']:.3f}s")
                logging.info(f"  Std Inference Time: {stats['std_inference_time']:.3f}s")
        logging.info("="*80)
        
        return summary


def main():
    """Main entry point for benchmarking."""
    parser = argparse.ArgumentParser(
        description="Benchmark AI-ENE against HuggingFace models"
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing test images"
    )
    parser.add_argument(
        "--seg-dir",
        required=True,
        help="Directory containing segmentations"
    )
    parser.add_argument(
        "--model-path",
        default="./ene_model/0208-1531-1_DualNet.h5",
        help="Path to AI-ENE model"
    )
    parser.add_argument(
        "--output-dir",
        default="./benchmark_results",
        help="Directory to save benchmark results"
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Maximum number of cases to process"
    )
    parser.add_argument(
        "--hf-models",
        nargs="+",
        default=None,
        help="List of HuggingFace model IDs to benchmark"
    )
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = HuggingFaceBenchmark(
        output_dir=args.output_dir,
        hf_models=args.hf_models
    )
    
    # Run benchmark
    df_results = benchmark.benchmark_on_dataset(
        image_dir=args.image_dir,
        seg_dir=args.seg_dir,
        ai_ene_model_path=args.model_path,
        max_cases=args.max_cases
    )
    
    # Generate summary
    summary = benchmark.generate_summary_report(df_results)
    
    logging.info("\nBenchmarking complete!")


if __name__ == "__main__":
    main()
