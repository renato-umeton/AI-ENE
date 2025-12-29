#!/usr/bin/env python3
"""
Fetch and track HuggingFace leaderboard data for medical imaging models.

This module provides utilities to:
- Fetch latest model rankings from HuggingFace leaderboards
- Track performance trends over time
- Identify top-performing models for benchmarking
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

import requests


class HuggingFaceLeaderboardTracker:
    """Track and fetch data from HuggingFace leaderboards."""
    
    def __init__(
        self,
        cache_dir: str = "./leaderboard_cache",
        update_interval_hours: int = 24
    ):
        """
        Initialize leaderboard tracker.
        
        Args:
            cache_dir: Directory to cache leaderboard data
            update_interval_hours: How often to refresh leaderboard data
        """
        self.cache_dir = cache_dir
        self.update_interval_hours = update_interval_hours
        os.makedirs(cache_dir, exist_ok=True)
        
        # HuggingFace API endpoints
        self.api_base = "https://huggingface.co/api"
        
        # Medical imaging related tasks
        self.medical_tasks = [
            "image-segmentation",
            "image-classification",
            "object-detection",
        ]
        
    def fetch_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch detailed information about a HuggingFace model.
        
        Args:
            model_id: HuggingFace model identifier
            
        Returns:
            Dictionary with model information or None if fetch fails
        """
        try:
            url = f"{self.api_base}/models/{model_id}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            model_info = response.json()
            
            # Cache the result
            cache_path = os.path.join(
                self.cache_dir,
                f"{model_id.replace('/', '_')}_info.json"
            )
            with open(cache_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            return model_info
        except Exception as e:
            logging.error(f"Failed to fetch info for {model_id}: {e}")
            return None
    
    def search_medical_models(
        self,
        task: Optional[str] = None,
        limit: int = 10,
        sort: str = "downloads"
    ) -> List[Dict[str, Any]]:
        """
        Search for medical imaging models on HuggingFace.
        
        Args:
            task: Specific task to filter (e.g., 'image-segmentation')
            limit: Maximum number of models to return
            sort: Sort criterion ('downloads', 'likes', 'created')
            
        Returns:
            List of model information dictionaries
        """
        try:
            # Build search query
            url = f"{self.api_base}/models"
            params = {
                "limit": limit,
                "sort": sort,
                "direction": -1,  # Descending
            }
            
            # Add task filter if specified
            if task:
                params["filter"] = task
            
            # Add medical/biomedical search terms
            params["search"] = "medical OR biomedical OR radiology OR clinical"
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            models = response.json()
            
            # Cache results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cache_path = os.path.join(
                self.cache_dir,
                f"medical_models_search_{timestamp}.json"
            )
            with open(cache_path, 'w') as f:
                json.dump(models, f, indent=2)
            
            logging.info(f"Found {len(models)} medical imaging models")
            return models
        
        except Exception as e:
            logging.error(f"Failed to search medical models: {e}")
            return []
    
    def get_top_models_by_task(
        self,
        task: str,
        top_n: int = 5
    ) -> List[str]:
        """
        Get top N model IDs for a specific task.
        
        Args:
            task: Task type (e.g., 'image-segmentation')
            top_n: Number of top models to return
            
        Returns:
            List of model IDs
        """
        models = self.search_medical_models(task=task, limit=top_n)
        return [m.get('modelId', m.get('id', '')) for m in models if m.get('modelId') or m.get('id')]
    
    def get_model_metrics(self, model_id: str) -> Dict[str, Any]:
        """
        Get available metrics for a model.
        
        Args:
            model_id: HuggingFace model identifier
            
        Returns:
            Dictionary with model metrics
        """
        model_info = self.fetch_model_info(model_id)
        
        if not model_info:
            return {}
        
        metrics = {
            "model_id": model_id,
            "downloads": model_info.get("downloads", 0),
            "likes": model_info.get("likes", 0),
            "created_at": model_info.get("createdAt"),
            "last_modified": model_info.get("lastModified"),
            "tags": model_info.get("tags", []),
            "pipeline_tag": model_info.get("pipeline_tag"),
            "library_name": model_info.get("library_name"),
        }
        
        return metrics
    
    def generate_leaderboard_report(
        self,
        tasks: Optional[List[str]] = None,
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive leaderboard report.
        
        Args:
            tasks: List of tasks to include (None = all medical tasks)
            top_n: Number of top models per task
            
        Returns:
            Dictionary with leaderboard report
        """
        tasks = tasks or self.medical_tasks
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "tasks": {},
        }
        
        for task in tasks:
            logging.info(f"Fetching top models for task: {task}")
            model_ids = self.get_top_models_by_task(task, top_n)
            
            task_models = []
            for model_id in model_ids:
                if model_id:
                    metrics = self.get_model_metrics(model_id)
                    if metrics:
                        task_models.append(metrics)
            
            report["tasks"][task] = task_models
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(
            self.cache_dir,
            f"leaderboard_report_{timestamp}.json"
        )
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logging.info(f"Leaderboard report saved to {report_path}")
        return report
    
    def recommend_benchmark_models(
        self,
        task: str = "image-segmentation",
        min_downloads: int = 1000,
        max_models: int = 5
    ) -> List[str]:
        """
        Recommend models for benchmarking based on popularity and relevance.
        
        Args:
            task: Task type to focus on
            min_downloads: Minimum number of downloads required
            max_models: Maximum number of models to recommend
            
        Returns:
            List of recommended model IDs
        """
        models = self.search_medical_models(task=task, limit=max_models * 2)
        
        recommended = []
        for model in models:
            model_id = model.get('modelId', model.get('id'))
            downloads = model.get('downloads', 0)
            
            if model_id and downloads >= min_downloads:
                recommended.append(model_id)
            
            if len(recommended) >= max_models:
                break
        
        logging.info(f"Recommended {len(recommended)} models for benchmarking")
        return recommended


def main():
    """Demo usage of leaderboard tracker."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    
    tracker = HuggingFaceLeaderboardTracker()
    
    # Generate leaderboard report
    print("Generating leaderboard report...")
    report = tracker.generate_leaderboard_report(top_n=5)
    
    print("\n" + "="*80)
    print("HUGGINGFACE MEDICAL IMAGING LEADERBOARD")
    print("="*80)
    
    for task, models in report["tasks"].items():
        print(f"\n{task.upper()}:")
        print("-" * 80)
        for idx, model in enumerate(models, 1):
            print(f"{idx}. {model['model_id']}")
            print(f"   Downloads: {model['downloads']:,}")
            print(f"   Likes: {model['likes']}")
            print(f"   Library: {model.get('library_name', 'N/A')}")
    
    print("\n" + "="*80)
    
    # Get recommendations
    print("\nRecommended models for benchmarking:")
    recommendations = tracker.recommend_benchmark_models(
        task="image-segmentation",
        min_downloads=1000,
        max_models=5
    )
    for idx, model_id in enumerate(recommendations, 1):
        print(f"{idx}. {model_id}")


if __name__ == "__main__":
    main()
