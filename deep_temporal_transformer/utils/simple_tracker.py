"""
Simple Experiment Tracker
Lightweight experiment logging to JSON for thesis documentation.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import hashlib


class SimpleExperimentTracker:
    """Simple JSON-based experiment tracker for thesis documentation."""
    
    def __init__(self, experiments_dir: str = "experiments"):
        """
        Initialize experiment tracker.
        
        Args:
            experiments_dir: Directory to store experiment logs
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiments_file = self.experiments_dir / "experiments_log.json"
        self.experiments = self._load_experiments()
        
        self.current_experiment = None
    
    def _load_experiments(self) -> Dict[str, Any]:
        """Load existing experiments from JSON file."""
        if self.experiments_file.exists():
            with open(self.experiments_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_experiments(self):
        """Save experiments to JSON file."""
        with open(self.experiments_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)
    
    def start_experiment(
        self,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        description: str = ""
    ) -> str:
        """
        Start tracking a new experiment.
        
        Args:
            name: Experiment name (auto-generated if None)
            config: Model configuration
            description: Experiment description
            
        Returns:
            experiment_id
        """
        # Generate experiment ID
        if name is None:
            exp_num = len(self.experiments) + 1
            name = f"exp_{exp_num:03d}"
        
        exp_id = name
        
        # Create experiment record
        self.current_experiment = {
            'id': exp_id,
            'name': name,
            'description': description,
            'start_time': datetime.now().isoformat(),
            'config': config or {},
            'metrics': {},
            'status': 'running'
        }
        
        return exp_id
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters for current experiment."""
        if self.current_experiment is None:
            raise RuntimeError("No active experiment. Call start_experiment() first.")
        
        self.current_experiment['config'].update(params)
    
    def log_metric(self, metric_name: str, value: float, step: Optional[int] = None):
        """
        Log a metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            step: Training step/epoch (optional)
        """
        if self.current_experiment is None:
            raise RuntimeError("No active experiment.")
        
        if metric_name not in self.current_experiment['metrics']:
            self.current_experiment['metrics'][metric_name] = []
        
        log_entry = {'value': value}
        if step is not None:
            log_entry['step'] = step
        
        self.current_experiment['metrics'][metric_name].append(log_entry)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics at once."""
        for metric_name, value in metrics.items():
            self.log_metric(metric_name, value, step)
    
    def end_experiment(self, final_results: Optional[Dict[str, Any]] = None):
        """
        End the current experiment.
        
        Args:
            final_results: Final evaluation results
        """
        if self.current_experiment is None:
            raise RuntimeError("No active experiment.")
        
        self.current_experiment['end_time'] = datetime.now().isoformat()
        self.current_experiment['status'] = 'completed'
        
        if final_results:
            self.current_experiment['final_results'] = final_results
        
        # Save to experiments log
        exp_id = self.current_experiment['id']
        self.experiments[exp_id] = self.current_experiment
        self._save_experiments()
        
        # Also save individual experiment file
        exp_file = self.experiments_dir / f"{exp_id}.json"
        with open(exp_file, 'w') as f:
            json.dump(self.current_experiment, f, indent=2)
        
        print(f"✅ Experiment {exp_id} saved to {exp_file}")
        
        self.current_experiment = None
    
    def get_experiment(self, exp_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an experiment by ID."""
        return self.experiments.get(exp_id)
    
    def list_experiments(self) -> list:
        """List all experiments."""
        return list(self.experiments.keys())
    
    def get_best_experiment(self, metric: str = 'f1') -> Optional[Dict[str, Any]]:
        """
        Get the experiment with the best performance on a metric.
        
        Args:
            metric: Metric to compare (e.g., 'f1', 'auc')
            
        Returns:
            Best experiment dict or None
        """
        best_exp = None
        best_value = -float('inf')
        
        for exp in self.experiments.values():
            if exp.get('status') != 'completed':
                continue
            
            final_results = exp.get('final_results', {})
            value = final_results.get(metric)
            
            if value is not None and value > best_value:
                best_value = value
                best_exp = exp
        
        return best_exp
    
    def compare_experiments(
        self,
        exp_ids: list,
        metrics: list = ['f1', 'auc', 'precision', 'recall']
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple experiments.
        
        Args:
            exp_ids: List of experiment IDs to compare
            metrics: Metrics to compare
            
        Returns:
            Dict mapping exp_id to metrics dict
        """
        comparison = {}
        
        for exp_id in exp_ids:
            exp = self.get_experiment(exp_id)
            if exp and exp.get('status') == 'completed':
                final_results = exp.get('final_results', {})
                comparison[exp_id] = {
                    m: final_results.get(m, None) for m in metrics
                }
        
        return comparison


# Global tracker instance for easy access
_global_tracker = None

def get_tracker(experiments_dir: str = "experiments") -> SimpleExperimentTracker:
    """Get or create global experiment tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = SimpleExperimentTracker(experiments_dir)
    return _global_tracker
