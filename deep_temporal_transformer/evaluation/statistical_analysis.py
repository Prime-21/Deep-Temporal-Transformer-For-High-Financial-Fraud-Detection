"""
Statistical Analysis Tools for Thesis
Provides confidence intervals, hypothesis tests, and effect sizes for academic rigor.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import logging

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Statistical analysis tools for model evaluation."""
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize statistical analyzer.
        
        Args:
            confidence_level: Confidence level for intervals (default 0.95 for 95%)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def bootstrap_confidence_interval(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_fn,
        n_bootstrap: int = 1000,
        random_state: int = 42
    ) -> Tuple[float, float, float]:
        """
        Calculate confidence interval using bootstrap resampling.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels or probabilities
            metric_fn: Metric function (e.g., f1_score)
            n_bootstrap: Number of bootstrap samples
            random_state: Random seed
            
        Returns:
            (mean, lower_bound, upper_bound)
        """
        np.random.seed(random_state)
        n_samples = len(y_true)
        scores = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            # Calculate metric
            try:
                score = metric_fn(y_true_boot, y_pred_boot)
                scores.append(score)
            except:
                continue
        
        scores = np.array(scores)
        mean_score = np.mean(scores)
        lower = np.percentile(scores, (self.alpha / 2) * 100)
        upper = np.percentile(scores, (1 - self.alpha / 2) * 100)
        
        return mean_score, lower, upper
    
    def compare_models(
        self,
        y_true: np.ndarray,
        y_pred_a: np.ndarray,
        y_pred_b: np.ndarray,
        model_a_name: str = "Model A",
        model_b_name: str = "Model B"
    ) -> Dict[str, any]:
        """
        Compare two models using paired t-test and McNemar's test.
        
        Args:
            y_true: True labels
            y_pred_a: Predictions from model A
            y_pred_b: Predictions from model B
            model_a_name: Name of model A
            model_b_name: Name of model B
            
        Returns:
            Dictionary with comparison results
        """
        # Calculate metrics for both models
        f1_a = f1_score(y_true, y_pred_a)
        f1_b = f1_score(y_true, y_pred_b)
        
        # McNemar's test (for binary classifications)
        # Contingency table: both correct, A correct B wrong, A wrong B correct, both wrong
        both_correct = np.sum((y_pred_a == y_true) & (y_pred_b == y_true))
        a_correct_b_wrong = np.sum((y_pred_a == y_true) & (y_pred_b != y_true))
        a_wrong_b_correct = np.sum((y_pred_a != y_true) & (y_pred_b == y_true))
        both_wrong = np.sum((y_pred_a != y_true) & (y_pred_b != y_true))
        
        # McNemar's test statistic
        if (a_correct_b_wrong + a_wrong_b_correct) > 0:
            mcnemar_stat = ((abs(a_correct_b_wrong - a_wrong_b_correct) - 1) ** 2) / \
                          (a_correct_b_wrong + a_wrong_b_correct)
            mcnemar_p = 1 - stats.chi2.cdf(mcnemar_stat, 1)
        else:
            mcnemar_stat = 0
            mcnemar_p = 1.0
        
        # Effect size (Cohen's h for proportions)
        p_a = f1_a
        p_b = f1_b
        cohens_h = 2 * (np.arcsin(np.sqrt(p_a)) - np.arcsin(np.sqrt(p_b)))
        
        results = {
            f'{model_a_name}_f1': f1_a,
            f'{model_b_name}_f1': f1_b,
            'f1_difference': f1_a - f1_b,
            'mcnemar_statistic': mcnemar_stat,
            'mcnemar_p_value': mcnemar_p,
            'statistically_significant': mcnemar_p < 0.05,
            'cohens_h': cohens_h,
            'effect_size': self._interpret_cohens_h(cohens_h),
            'winner': model_a_name if f1_a > f1_b else model_b_name
        }
        
        return results
    
    def _interpret_cohens_h(self, h: float) -> str:
        """Interpret Cohen's h effect size."""
        abs_h = abs(h)
        if abs_h < 0.2:
            return "negligible"
        elif abs_h < 0.5:
            return "small"
        elif abs_h < 0.8:
            return "medium"
        else:
            return "large"
    
    def cross_validation_stats(
        self,
        cv_scores: List[float]
    ) -> Dict[str, float]:
        """
        Calculate statistics from cross-validation scores.
        
        Args:
            cv_scores: List of scores from each fold
            
        Returns:
            Dictionary with mean, std, and confidence interval
        """
        cv_scores = np.array(cv_scores)
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        # Confidence interval using t-distribution
        n = len(cv_scores)
        t_stat = stats.t.ppf(1 - self.alpha / 2, n - 1)
        margin = t_stat * (std_score / np.sqrt(n))
        
        return {
            'mean': mean_score,
            'std': std_score,
            'ci_lower': mean_score - margin,
            'ci_upper': mean_score + margin,
            'n_folds': n
        }
    
    def comprehensive_evaluation(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        n_bootstrap: int = 1000
    ) -> Dict[str, any]:
        """
        Perform comprehensive evaluation with confidence intervals.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional, for AUC)
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Dictionary with all metrics and their confidence intervals
        """
        results = {}
        
        # F1 Score with CI
        f1_mean, f1_lower, f1_upper = self.bootstrap_confidence_interval(
            y_true, y_pred, f1_score, n_bootstrap
        )
        results['f1'] = {
            'mean': f1_mean,
            'ci_lower': f1_lower,
            'ci_upper': f1_upper,
            'ci_string': f"{f1_mean:.4f} [{f1_lower:.4f}, {f1_upper:.4f}]"
        }
        
        # Precision with CI
        prec_mean, prec_lower, prec_upper = self.bootstrap_confidence_interval(
            y_true, y_pred, precision_score, n_bootstrap
        )
        results['precision'] = {
            'mean': prec_mean,
            'ci_lower': prec_lower,
            'ci_upper': prec_upper,
            'ci_string': f"{prec_mean:.4f} [{prec_lower:.4f}, {prec_upper:.4f}]"
        }
        
        # Recall with CI
        rec_mean, rec_lower, rec_upper = self.bootstrap_confidence_interval(
            y_true, y_pred, recall_score, n_bootstrap
        )
        results['recall'] = {
            'mean': rec_mean,
            'ci_lower': rec_lower,
            'ci_upper': rec_upper,
            'ci_string': f"{rec_mean:.4f} [{rec_lower:.4f}, {rec_upper:.4f}]"
        }
        
        # AUC with CI (if probabilities provided)
        if y_prob is not None:
            auc_mean, auc_lower, auc_upper = self.bootstrap_confidence_interval(
                y_true, y_prob, roc_auc_score, n_bootstrap
            )
            results['auc'] = {
                'mean': auc_mean,
                'ci_lower': auc_lower,
                'ci_upper': auc_upper,
                'ci_string': f"{auc_mean:.4f} [{auc_lower:.4f}, {auc_upper:.4f}]"
            }
        
        return results


def format_result_with_ci(metric_dict: Dict[str, any], latex: bool = False) -> str:
    """
    Format a metric result with confidence interval.
    
    Args:
        metric_dict: Dictionary with 'mean', 'ci_lower', 'ci_upper'
        latex: If True, format for LaTeX
        
    Returns:
        Formatted string
    """
    if latex:
        return f"{metric_dict['mean']:.3f} $\\pm$ [{metric_dict['ci_lower']:.3f}, {metric_dict['ci_upper']:.3f}]"
    else:
        return metric_dict['ci_string']
