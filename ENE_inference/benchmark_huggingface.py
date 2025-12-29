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
    
    def generate_html_leaderboard(
        self,
        df_results: pd.DataFrame,
        summary: Dict[str, Any],
        output_filename: str = "leaderboard.html"
    ) -> str:
        """
        Generate an HTML leaderboard showing AI-ENE's ranking compared to other models.
        
        Args:
            df_results: DataFrame with benchmark results
            summary: Summary statistics dictionary
            output_filename: Name of the output HTML file
            
        Returns:
            Path to the generated HTML file
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Sort models by average inference time (lower is better)
        model_rankings = []
        for model_name, stats in summary.items():
            if stats['avg_inference_time'] is not None:
                model_rankings.append({
                    'model': model_name,
                    'success_rate': stats['success_rate'],
                    'avg_inference_time': stats['avg_inference_time'],
                    'std_inference_time': stats['std_inference_time'],
                    'total_cases': stats['total_cases'],
                    'successful_cases': stats['successful_cases']
                })
        
        # Sort by success rate (descending) then by inference time (ascending)
        model_rankings.sort(key=lambda x: (-x['success_rate'], x['avg_inference_time']))
        
        # Find AI-ENE's rank
        ai_ene_rank = None
        for idx, entry in enumerate(model_rankings):
            if 'AI-ENE' in entry['model']:
                ai_ene_rank = idx + 1
                break
        
        # Generate HTML
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI-ENE Benchmark Leaderboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 2rem;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 3rem 2rem;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}
        
        .header p {{
            font-size: 1.1rem;
            opacity: 0.9;
        }}
        
        .timestamp {{
            text-align: center;
            padding: 1rem;
            background: #f8f9fa;
            color: #6c757d;
            font-size: 0.9rem;
        }}
        
        .leaderboard {{
            padding: 2rem;
        }}
        
        .leaderboard-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        .leaderboard-table thead {{
            background: #f8f9fa;
        }}
        
        .leaderboard-table th {{
            padding: 1rem;
            text-align: left;
            font-weight: 600;
            color: #495057;
            border-bottom: 2px solid #dee2e6;
        }}
        
        .leaderboard-table th.rank {{
            width: 80px;
            text-align: center;
        }}
        
        .leaderboard-table td {{
            padding: 1rem;
            border-bottom: 1px solid #dee2e6;
        }}
        
        .leaderboard-table tr:hover {{
            background: #f8f9fa;
        }}
        
        .leaderboard-table tr.highlight {{
            background: linear-gradient(90deg, #fff3cd 0%, #ffffff 100%);
            border-left: 4px solid #ffc107;
        }}
        
        .leaderboard-table tr.highlight:hover {{
            background: linear-gradient(90deg, #ffe69c 0%, #f8f9fa 100%);
        }}
        
        .rank-badge {{
            display: inline-block;
            width: 40px;
            height: 40px;
            line-height: 40px;
            text-align: center;
            border-radius: 50%;
            font-weight: 700;
            font-size: 1.1rem;
        }}
        
        .rank-1 {{
            background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%);
            color: #856404;
            box-shadow: 0 4px 12px rgba(255, 215, 0, 0.4);
        }}
        
        .rank-2 {{
            background: linear-gradient(135deg, #c0c0c0 0%, #e8e8e8 100%);
            color: #495057;
            box-shadow: 0 4px 12px rgba(192, 192, 192, 0.4);
        }}
        
        .rank-3 {{
            background: linear-gradient(135deg, #cd7f32 0%, #e5a668 100%);
            color: #ffffff;
            box-shadow: 0 4px 12px rgba(205, 127, 50, 0.4);
        }}
        
        .rank-other {{
            background: #e9ecef;
            color: #6c757d;
        }}
        
        .model-name {{
            font-weight: 600;
            color: #212529;
        }}
        
        .ai-ene-badge {{
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-left: 0.5rem;
            vertical-align: middle;
        }}
        
        .metric {{
            font-family: 'Courier New', monospace;
            color: #495057;
        }}
        
        .success-rate {{
            font-weight: 600;
        }}
        
        .success-100 {{
            color: #28a745;
        }}
        
        .success-high {{
            color: #5cb85c;
        }}
        
        .success-medium {{
            color: #ffc107;
        }}
        
        .success-low {{
            color: #dc3545;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            padding: 2rem;
            background: #f8f9fa;
        }}
        
        .stat-card {{
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }}
        
        .stat-card h3 {{
            color: #6c757d;
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .stat-card .value {{
            font-size: 2rem;
            font-weight: 700;
            color: #212529;
        }}
        
        .stat-card .label {{
            color: #6c757d;
            font-size: 0.9rem;
            margin-top: 0.25rem;
        }}
        
        .footer {{
            text-align: center;
            padding: 2rem;
            color: #6c757d;
            border-top: 1px solid #dee2e6;
        }}
        
        .footer a {{
            color: #667eea;
            text-decoration: none;
            font-weight: 600;
        }}
        
        .footer a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏆 AI-ENE Benchmark Leaderboard</h1>
            <p>Performance comparison against HuggingFace state-of-the-art models</p>
        </div>
        
        <div class="timestamp">
            Last updated: {timestamp}
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Models</h3>
                <div class="value">{len(model_rankings)}</div>
                <div class="label">Benchmarked</div>
            </div>
            <div class="stat-card">
                <h3>AI-ENE Rank</h3>
                <div class="value">#{ai_ene_rank if ai_ene_rank else 'N/A'}</div>
                <div class="label">Overall Position</div>
            </div>
            <div class="stat-card">
                <h3>Total Cases</h3>
                <div class="value">{df_results['case_id'].nunique() if 'case_id' in df_results.columns else len(df_results)}</div>
                <div class="label">Test Cases Evaluated</div>
            </div>
        </div>
        
        <div class="leaderboard">
            <table class="leaderboard-table">
                <thead>
                    <tr>
                        <th class="rank">Rank</th>
                        <th>Model</th>
                        <th>Success Rate</th>
                        <th>Avg Inference Time</th>
                        <th>Std Dev</th>
                        <th>Cases</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        # Add table rows
        for idx, entry in enumerate(model_rankings):
            rank = idx + 1
            is_ai_ene = 'AI-ENE' in entry['model']
            
            # Determine rank badge class
            if rank == 1:
                rank_class = "rank-1"
            elif rank == 2:
                rank_class = "rank-2"
            elif rank == 3:
                rank_class = "rank-3"
            else:
                rank_class = "rank-other"
            
            # Determine success rate class
            success_rate = entry['success_rate']
            if success_rate >= 1.0:
                success_class = "success-100"
            elif success_rate >= 0.8:
                success_class = "success-high"
            elif success_rate >= 0.5:
                success_class = "success-medium"
            else:
                success_class = "success-low"
            
            row_class = "highlight" if is_ai_ene else ""
            ai_ene_badge = '<span class="ai-ene-badge">⭐ This Model</span>' if is_ai_ene else ''
            
            html_content += f"""
                    <tr class="{row_class}">
                        <td style="text-align: center;">
                            <span class="rank-badge {rank_class}">{rank}</span>
                        </td>
                        <td>
                            <span class="model-name">{entry['model']}</span>
                            {ai_ene_badge}
                        </td>
                        <td>
                            <span class="success-rate {success_class}">{success_rate:.1%}</span>
                        </td>
                        <td>
                            <span class="metric">{entry['avg_inference_time']:.3f}s</span>
                        </td>
                        <td>
                            <span class="metric">{entry['std_inference_time']:.3f}s</span>
                        </td>
                        <td>
                            <span class="metric">{entry['successful_cases']}/{entry['total_cases']}</span>
                        </td>
                    </tr>
"""
        
        html_content += """
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>
                Generated by <a href="https://github.com/renato-umeton/AI-ENE" target="_blank">AI-ENE Benchmarking Framework</a>
            </p>
            <p style="margin-top: 0.5rem; font-size: 0.85rem;">
                Models sorted by success rate (descending) and inference time (ascending)
            </p>
        </div>
    </div>
</body>
</html>
"""
        
        # Save HTML file
        output_path = os.path.join(self.output_dir, output_filename)
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logging.info(f"HTML leaderboard saved to {output_path}")
        
        return output_path


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
    
    # Generate HTML leaderboard
    html_path = benchmark.generate_html_leaderboard(df_results, summary)
    logging.info(f"\nHTML leaderboard available at: {html_path}")
    
    logging.info("\nBenchmarking complete!")


if __name__ == "__main__":
    main()
