"""
Thesis Results Export Tools
Generate LaTeX tables, CSV exports, and formatted results for academic thesis.
"""

import json
import csv
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np
from datetime import datetime


class ThesisResultsExporter:
    """Export results in thesis-ready formats (LaTeX, CSV, JSON)."""
    
    def __init__(self, output_dir: str = "thesis_results"):
        """Initialize exporter with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_comparison_table_latex(
        self,
        results: Dict[str, Dict[str, Any]],
        metrics: List[str] = ['f1', 'auc', 'precision', 'recall'],
        caption: str = "Model Performance Comparison",
        label: str = "tab:model_comparison"
    ) -> str:
        """
        Generate LaTeX table comparing multiple models.
        
        Args:
            results: Dict mapping model names to their results
            metrics: List of metrics to include
            caption: Table caption
            label: LaTeX label for referencing
            
        Returns:
            LaTeX table string
        """
        models = list(results.keys())
        
        # Start table
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += f"\\caption{{{caption}}}\n"
        latex += f"\\label{{{label}}}\n"
        
        # Column specification
        num_cols = len(metrics) + 1  # models + metrics
        latex += "\\begin{tabular}{l" + "c" * len(metrics) + "}\n"
        latex += "\\toprule\n"
        
        # Header row
        header = "Model & " + " & ".join([m.upper() for m in metrics]) + " \\\\\n"
        latex += header
        latex += "\\midrule\n"
        
        # Data rows
        for model in models:
            row = model.replace("_", " ")  # Clean model name
            for metric in metrics:
                value = results[model].get(metric, {})
                
                # Check if we have confidence intervals
                if isinstance(value, dict) and 'mean' in value:
                    # Format with CI
                    mean = value['mean']
                    ci_lower = value['ci_lower']
                    ci_upper = value['ci_upper']
                    row += f" & {mean:.4f}"
                    row += f" \\tiny{{$\\pm$[{ci_lower:.4f},{ci_upper:.4f}]}}}"
                else:
                    # Simple value
                    row += f" & {value:.4f}"
            
            row += " \\\\\n"
            latex += row
        
        # Close table
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    def save_latex_table(
        self,
        latex_table: str,
        filename: str = "model_comparison.tex"
    ) -> Path:
        """Save LaTeX table to file."""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            f.write(latex_table)
        return filepath
    
    def export_results_csv(
        self,
        results: Dict[str, Dict[str, Any]],
        filename: str = "results.csv"
    ) -> Path:
        """
        Export results to CSV format.
        
        Args:
            results: Dict mapping model names to their results
            filename: Output filename
            
        Returns:
            Path to saved CSV file
        """
        filepath = self.output_dir / filename
        
        # Flatten results
        rows = []
        for model_name, model_results in results.items():
            row = {'model': model_name}
            
            for metric, value in model_results.items():
                if isinstance(value, dict):
                    # Expand nested dicts
                    for k, v in value.items():
                        row[f"{metric}_{k}"] = v
                else:
                    row[metric] = value
            
            rows.append(row)
        
        # Write CSV
        if rows:
            fieldnames = rows[0].keys()
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        
        return filepath
    
    def export_results_json(
        self,
        results: Dict[str, Any],
        filename: str = "results.json"
    ) -> Path:
        """Export results to JSON format."""
        filepath = self.output_dir / filename
        
        # Convert numpy arrays to lists
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        results_clean = convert_numpy(results)
        
        with open(filepath, 'w') as f:
            json.dump(results_clean, f, indent=2)
        
        return filepath
    
    def generate_ablation_table_latex(
        self,
        ablation_results: Dict[str, float],
        baseline_f1: float,
        caption: str = "Ablation Study Results",
        label: str = "tab:ablation"
    ) -> str:
        """
        Generate LaTeX table for ablation study.
        
        Args:
            ablation_results: Dict mapping component -> F1 score without it
            baseline_f1: Full model F1 score
            caption: Table caption
            label: LaTeX label
            
        Returns:
            LaTeX table string
        """
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += f"\\caption{{{caption}}}\n"
        latex += f"\\label{{{label}}}\n"
        latex += "\\begin{tabular}{lcc}\n"
        latex += "\\toprule\n"
        latex += "Component Removed & F1 Score & $\\Delta$ F1 \\\\\n"
        latex += "\\midrule\n"
        
        # Full model
        latex += f"\\textbf{{Full Model}} & {baseline_f1:.4f} & -- \\\\\n"
        latex += "\\midrule\n"
        
        # Ablations
        for component, f1_score in ablation_results.items():
            delta = baseline_f1 - f1_score
            latex += f"{component.replace('_', ' ').title()} & {f1_score:.4f} & "
            latex += f"{delta:+.4f} \\\\\n"
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    def create_thesis_summary(
        self,
        transformer_results: Dict[str, Any],
        baseline_results: Dict[str, Dict[str, Any]],
        additional_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a comprehensive summary for thesis.
        
        Args:
            transformer_results: Results from Deep Temporal Transformer
            baseline_results: Results from baseline models
            additional_info: Extra information (training time, params, etc.)
            
        Returns:
            Markdown formatted summary
        """
        summary = "# Deep Temporal Transformer - Thesis Results Summary\n\n"
        summary += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        summary += "## Model Performance\n\n"
        summary += "### Deep Temporal Transformer (Proposed Method)\n"
        summary += f"- **F1 Score**: {transformer_results.get('f1', 'N/A')}\n"
        summary += f"- **AUC Score**: {transformer_results.get('auc', 'N/A')}\n"
        summary += f"- **Precision**: {transformer_results.get('precision', 'N/A')}\n"
        summary += f"- **Recall**: {transformer_results.get('recall', 'N/A')}\n\n"
        
        summary += "### Baseline Models\n"
        for model_name, results in baseline_results.items():
            summary += f"\n**{model_name}**:\n"
            summary += f"- F1: {results.get('f1', 'N/A')}\n"
            summary += f"- AUC: {results.get('auc', 'N/A')}\n"
        
        if additional_info:
            summary += "\n## Additional Information\n"
            for key, value in additional_info.items():
                summary += f"- **{key.replace('_', ' ').title()}**: {value}\n"
        
        summary += "\n---\n"
        summary += "\\# Results exported for thesis submission\n"
        
        return summary
    
    def save_all_formats(
        self,
        results: Dict[str, Dict[str, Any]],
        prefix: str = "experiment"
    ) -> Dict[str, Path]:
        """
        Save results in all formats (LaTeX, CSV, JSON).
        
        Args:
            results: Results dictionary
            prefix: Filename prefix
            
        Returns:
            Dict mapping format to saved filepath
        """
        saved_files = {}
        
        # LaTeX table
        latex_table = self.generate_comparison_table_latex(results)
        saved_files['latex'] = self.save_latex_table(
            latex_table, 
            f"{prefix}_table.tex"
        )
        
        # CSV
        saved_files['csv'] = self.export_results_csv(
            results,
            f"{prefix}_results.csv"
        )
        
        # JSON
        saved_files['json'] = self.export_results_json(
            results,
            f"{prefix}_results.json"
        )
        
        return saved_files


def quick_latex_table(
    model_results: Dict[str, float],
    model_name: str = "Deep Temporal Transformer"
) -> str:
    """
    Quick helper to generate single-model LaTeX row.
    
    Args:
        model_results: Dict with f1, auc, precision, recall
        model_name: Name of the model
        
    Returns:
        LaTeX table row
    """
    row = model_name
    for metric in ['f1', 'auc', 'precision', 'recall']:
        value = model_results.get(metric, 0)
        row += f" & {value:.4f}"
    row += " \\\\\n"
    return row
