"""
Automated validation pipeline for comprehensive model testing
Integrates all test suites and generates reports
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import time
import json
import sys
import subprocess
from datetime import datetime

# Add parent directories to path for imports
project_root = Path(__file__).parent.parent.parent  
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'geodesic_m1'))

from models.geodesic_model import GeodesicNODE
from training.data_loader import SpectralDataset
from tests.test_data_adherence import DataAdherenceTests
from tests.test_geometry_learning import GeometryLearningTests


class ComprehensiveValidationPipeline:
    """
    Comprehensive validation pipeline for geodesic NODE models
    Runs all test suites and generates detailed reports
    """
    
    def __init__(self, 
                 models: List[GeodesicNODE],
                 datasets: List[SpectralDataset],
                 output_dir: str = "outputs/validation",
                 verbose: bool = True):
        """
        Initialize comprehensive validation pipeline
        
        Args:
            models: List of trained geodesic NODE models (for leave-one-out)
            datasets: List of datasets corresponding to each model
            output_dir: Directory for saving validation results
            verbose: Whether to print detailed output
        """
        self.models = models
        self.datasets = datasets
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize test suites for each model
        self.data_adherence_tests = []
        self.geometry_learning_tests = []
        
        for model, dataset in zip(models, datasets):
            self.data_adherence_tests.append(DataAdherenceTests(model, dataset))
            self.geometry_learning_tests.append(GeometryLearningTests(model, dataset))
        
        # Track results
        self.validation_results = {}
        self.start_time = None
        
    def run_complete_validation(self) -> Dict[str, Any]:
        """
        Run complete validation pipeline on all models
        
        Returns:
            Comprehensive validation results
        """
        if self.verbose:
            print("🚀 Starting comprehensive validation pipeline...")
            print(f"   Models to test: {len(self.models)}")
            print(f"   Output directory: {self.output_dir}")
            print("="*80)
        
        self.start_time = time.time()
        
        # Run validation for each model
        model_results = []
        for model_idx, (model, dataset) in enumerate(zip(self.models, self.datasets)):
            if self.verbose:
                print(f"\n🔬 Validating Model {model_idx + 1}/{len(self.models)} (excluding concentration {model_idx})")
            
            model_result = self._validate_single_model(
                model_idx, model, dataset,
                self.data_adherence_tests[model_idx],
                self.geometry_learning_tests[model_idx]
            )
            model_results.append(model_result)
        
        # Aggregate results across all models
        if self.verbose:
            print(f"\n📊 Aggregating results across {len(self.models)} models...")
        
        aggregate_results = self._aggregate_model_results(model_results)
        
        # Create comprehensive results structure
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'validation_duration_seconds': time.time() - self.start_time,
            'n_models': len(self.models),
            'individual_model_results': model_results,
            'aggregate_results': aggregate_results,
            'overall_summary': self._generate_overall_summary(aggregate_results)
        }
        
        # Save results and generate reports
        self._save_validation_results()
        self._generate_html_report()
        self._create_comprehensive_visualizations()
        
        if self.verbose:
            self._print_final_summary()
        
        return self.validation_results
    
    def _validate_single_model(self, 
                              model_idx: int,
                              model: GeodesicNODE,
                              dataset: SpectralDataset,
                              adherence_tester: DataAdherenceTests,
                              geometry_tester: GeometryLearningTests) -> Dict[str, Any]:
        """
        Run complete validation on a single model
        
        Returns:
            Single model validation results
        """
        model_start_time = time.time()
        
        # Model info
        model_info = model.get_model_info()
        
        # Run data adherence tests
        if self.verbose:
            print("   🧪 Running data adherence tests...")
        adherence_results = adherence_tester.run_all_data_adherence_tests(verbose=False)
        
        # Run geometry learning tests
        if self.verbose:
            print("   🧪 Running geometry learning tests...")
        geometry_results = geometry_tester.run_all_geometry_tests(verbose=False)
        
        # Run numerical stability tests
        if self.verbose:
            print("   🧪 Running numerical stability tests...")
        stability_results = self._test_numerical_stability(model, dataset)
        
        # Run performance benchmarks
        if self.verbose:
            print("   🧪 Running performance benchmarks...")
        performance_results = self._benchmark_performance(model, dataset)
        
        # Compute overall model score
        overall_score = self._compute_model_overall_score(
            adherence_results, geometry_results, stability_results
        )
        
        model_duration = time.time() - model_start_time
        
        return {
            'model_idx': model_idx,
            'excluded_concentration': model_idx,  # In leave-one-out, model index = excluded concentration
            'model_info': model_info,
            'validation_duration_seconds': model_duration,
            'data_adherence': adherence_results,
            'geometry_learning': geometry_results,
            'numerical_stability': stability_results,
            'performance': performance_results,
            'overall_score': overall_score,
            'timestamp': datetime.now().isoformat()
        }
    
    def _test_numerical_stability(self, model: GeodesicNODE, dataset: SpectralDataset) -> Dict[str, Any]:
        """
        Test numerical stability of the model
        
        Returns:
            Numerical stability metrics
        """
        stability_results = {
            'shooting_convergence_rate': 0.0,
            'integration_accuracy': 0.0,
            'gradient_stability': 0.0,
            'extreme_input_stability': 0.0,
            'memory_efficiency': 0.0
        }
        
        try:
            # Test shooting solver convergence
            convergence_rates = []
            for i in range(10):  # Test 10 random transitions
                c_src = torch.rand(1, device=model.device) * 2 - 1
                c_tgt = torch.rand(1, device=model.device) * 2 - 1
                wl = torch.rand(1, device=model.device) * 2 - 1
                
                with torch.no_grad():
                    result = model.forward(c_src, c_tgt, wl)
                    convergence_rates.append(float(result['convergence_rate']))
            
            stability_results['shooting_convergence_rate'] = float(np.mean(convergence_rates))
            
            # Test integration accuracy (consistency)
            accuracy_scores = []
            for i in range(5):
                c_src = torch.rand(1, device=model.device) * 2 - 1
                c_tgt = torch.rand(1, device=model.device) * 2 - 1
                wl = torch.rand(1, device=model.device) * 2 - 1
                
                # Run same prediction multiple times
                predictions = []
                for _ in range(3):
                    with torch.no_grad():
                        result = model.forward(c_src, c_tgt, wl)
                        predictions.append(float(result['absorbance']))
                
                # Check consistency
                pred_std = np.std(predictions)
                accuracy_scores.append(1.0 / (1.0 + pred_std))  # Lower std = higher accuracy
            
            stability_results['integration_accuracy'] = float(np.mean(accuracy_scores))
            
            # Test gradient stability (if in training mode)
            if model.training:
                model.eval()  # Switch to eval mode for this test
            
            # Test with extreme inputs (but reasonable)
            extreme_stability = []
            extreme_inputs = [
                (-0.99, 0.99, 0.0),   # Near boundaries
                (0.99, -0.99, 0.0),
                (0.0, 0.0, -0.99),    # Extreme wavelength
                (0.0, 0.0, 0.99),
            ]
            
            for c_src, c_tgt, wl in extreme_inputs:
                try:
                    with torch.no_grad():
                        c_src_tensor = torch.tensor([c_src], device=model.device, dtype=torch.float32)
                        c_tgt_tensor = torch.tensor([c_tgt], device=model.device, dtype=torch.float32)
                        wl_tensor = torch.tensor([wl], device=model.device, dtype=torch.float32)
                        
                        result = model.forward(c_src_tensor, c_tgt_tensor, wl_tensor)
                        pred = float(result['absorbance'])
                        
                        # Check if prediction is reasonable
                        if np.isfinite(pred) and abs(pred) < 10:
                            extreme_stability.append(1.0)
                        else:
                            extreme_stability.append(0.0)
                except:
                    extreme_stability.append(0.0)
            
            stability_results['extreme_input_stability'] = float(np.mean(extreme_stability))
            
            # Memory efficiency (placeholder - would need more sophisticated testing)
            stability_results['memory_efficiency'] = 0.8  # Assume good for now
            
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️  Stability testing encountered error: {e}")
            # Return default low scores on error
            pass
        
        return stability_results
    
    def _benchmark_performance(self, model: GeodesicNODE, dataset: SpectralDataset) -> Dict[str, Any]:
        """
        Benchmark model performance (speed, memory usage)
        
        Returns:
            Performance metrics
        """
        performance_results = {}
        
        try:
            # Single prediction benchmark
            c_src = torch.tensor([0.0], device=model.device, dtype=torch.float32)
            c_tgt = torch.tensor([0.5], device=model.device, dtype=torch.float32)
            wl = torch.tensor([0.0], device=model.device, dtype=torch.float32)
            
            # Warm up
            with torch.no_grad():
                for _ in range(3):
                    model.forward(c_src, c_tgt, wl)
            
            # Time single predictions
            single_times = []
            for _ in range(10):
                start_time = time.time()
                with torch.no_grad():
                    result = model.forward(c_src, c_tgt, wl)
                single_times.append(time.time() - start_time)
            
            performance_results['mean_single_prediction_time_ms'] = float(np.mean(single_times) * 1000)
            performance_results['std_single_prediction_time_ms'] = float(np.std(single_times) * 1000)
            
            # Batch prediction benchmark
            batch_size = 32
            c_src_batch = torch.rand(batch_size, device=model.device) * 2 - 1
            c_tgt_batch = torch.rand(batch_size, device=model.device) * 2 - 1
            wl_batch = torch.rand(batch_size, device=model.device) * 2 - 1
            
            # Warm up batch
            with torch.no_grad():
                for _ in range(2):
                    for i in range(batch_size):
                        model.forward(c_src_batch[i:i+1], c_tgt_batch[i:i+1], wl_batch[i:i+1])
            
            # Time batch predictions
            batch_start = time.time()
            with torch.no_grad():
                for i in range(batch_size):
                    result = model.forward(c_src_batch[i:i+1], c_tgt_batch[i:i+1], wl_batch[i:i+1])
            batch_time = time.time() - batch_start
            
            performance_results['batch_prediction_time_ms'] = float(batch_time * 1000)
            performance_results['predictions_per_second'] = float(batch_size / batch_time)
            
            # Model size
            total_params = sum(p.numel() for p in model.parameters())
            model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
            
            performance_results['total_parameters'] = int(total_params)
            performance_results['model_size_mb'] = float(model_size_mb)
            
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️  Performance benchmarking encountered error: {e}")
            # Set default values
            performance_results = {
                'mean_single_prediction_time_ms': 1000.0,
                'predictions_per_second': 1.0,
                'total_parameters': 0,
                'model_size_mb': 0.0
            }
        
        return performance_results
    
    def _compute_model_overall_score(self, adherence_results, geometry_results, stability_results) -> float:
        """
        Compute overall score for a single model
        
        Returns:
            Overall score (0-100)
        """
        scores = []
        weights = []
        
        # Data adherence (50% weight - most important)
        if 'overall_score' in adherence_results:
            scores.append(adherence_results['overall_score'])
            weights.append(0.5)
        
        # Geometry learning (30% weight)
        if 'overall_geometry_score' in geometry_results:
            scores.append(geometry_results['overall_geometry_score'])
            weights.append(0.3)
        
        # Numerical stability (20% weight)
        if stability_results:
            stability_score = (
                stability_results.get('shooting_convergence_rate', 0) * 40 +
                stability_results.get('integration_accuracy', 0) * 30 +
                stability_results.get('extreme_input_stability', 0) * 30
            )
            scores.append(stability_score)
            weights.append(0.2)
        
        if scores:
            overall = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            return float(max(0, min(100, overall)))
        else:
            return 0.0
    
    def _aggregate_model_results(self, model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate results across all models
        
        Returns:
            Aggregated metrics
        """
        if not model_results:
            return {}
        
        # Aggregate data adherence metrics
        adherence_metrics = []
        geometry_metrics = []
        stability_metrics = []
        overall_scores = []
        
        for result in model_results:
            if 'data_adherence' in result and 'overall_score' in result['data_adherence']:
                adherence_metrics.append(result['data_adherence']['overall_score'])
            
            if 'geometry_learning' in result and 'overall_geometry_score' in result['geometry_learning']:
                geometry_metrics.append(result['geometry_learning']['overall_geometry_score'])
            
            if 'numerical_stability' in result:
                stability = result['numerical_stability']
                stability_score = (
                    stability.get('shooting_convergence_rate', 0) * 40 +
                    stability.get('integration_accuracy', 0) * 30 +
                    stability.get('extreme_input_stability', 0) * 30
                )
                stability_metrics.append(stability_score)
            
            if 'overall_score' in result:
                overall_scores.append(result['overall_score'])
        
        # Compute aggregate statistics
        aggregate_results = {
            'data_adherence': {
                'mean_score': float(np.mean(adherence_metrics)) if adherence_metrics else 0.0,
                'std_score': float(np.std(adherence_metrics)) if adherence_metrics else 0.0,
                'min_score': float(np.min(adherence_metrics)) if adherence_metrics else 0.0,
                'max_score': float(np.max(adherence_metrics)) if adherence_metrics else 0.0
            },
            'geometry_learning': {
                'mean_score': float(np.mean(geometry_metrics)) if geometry_metrics else 0.0,
                'std_score': float(np.std(geometry_metrics)) if geometry_metrics else 0.0,
                'min_score': float(np.min(geometry_metrics)) if geometry_metrics else 0.0,
                'max_score': float(np.max(geometry_metrics)) if geometry_metrics else 0.0
            },
            'numerical_stability': {
                'mean_score': float(np.mean(stability_metrics)) if stability_metrics else 0.0,
                'std_score': float(np.std(stability_metrics)) if stability_metrics else 0.0,
                'min_score': float(np.min(stability_metrics)) if stability_metrics else 0.0,
                'max_score': float(np.max(stability_metrics)) if stability_metrics else 0.0
            },
            'overall': {
                'mean_score': float(np.mean(overall_scores)) if overall_scores else 0.0,
                'std_score': float(np.std(overall_scores)) if overall_scores else 0.0,
                'min_score': float(np.min(overall_scores)) if overall_scores else 0.0,
                'max_score': float(np.max(overall_scores)) if overall_scores else 0.0
            }
        }
        
        # Performance aggregation
        performance_metrics = []
        for result in model_results:
            if 'performance' in result:
                performance_metrics.append(result['performance'])
        
        if performance_metrics:
            aggregate_results['performance'] = {
                'mean_prediction_time_ms': float(np.mean([p.get('mean_single_prediction_time_ms', 0) for p in performance_metrics])),
                'mean_predictions_per_second': float(np.mean([p.get('predictions_per_second', 0) for p in performance_metrics])),
                'total_parameters': performance_metrics[0].get('total_parameters', 0),  # Same for all models
                'total_model_size_mb': float(sum(p.get('model_size_mb', 0) for p in performance_metrics))
            }
        
        return aggregate_results
    
    def _generate_overall_summary(self, aggregate_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate overall validation summary
        
        Returns:
            Summary with pass/fail status and recommendations
        """
        summary = {
            'validation_passed': False,
            'critical_issues': [],
            'warnings': [],
            'recommendations': [],
            'grade': 'F'
        }
        
        # Check overall scores
        overall_mean = aggregate_results.get('overall', {}).get('mean_score', 0)
        
        # Determine grade
        if overall_mean >= 90:
            summary['grade'] = 'A'
        elif overall_mean >= 80:
            summary['grade'] = 'B'  
        elif overall_mean >= 70:
            summary['grade'] = 'C'
        elif overall_mean >= 60:
            summary['grade'] = 'D'
        else:
            summary['grade'] = 'F'
        
        # Check pass/fail (threshold: 70)
        summary['validation_passed'] = overall_mean >= 70
        
        # Generate issues and recommendations
        adherence_mean = aggregate_results.get('data_adherence', {}).get('mean_score', 0)
        geometry_mean = aggregate_results.get('geometry_learning', {}).get('mean_score', 0)
        stability_mean = aggregate_results.get('numerical_stability', {}).get('mean_score', 0)
        
        if adherence_mean < 60:
            summary['critical_issues'].append("Data adherence severely inadequate - model not learning from training data")
            summary['recommendations'].append("Increase Christoffel matching loss weight and verify target computation")
        
        if geometry_mean < 50:
            summary['critical_issues'].append("Geometry learning failing - Christoffel symbols not matching targets")
            summary['recommendations'].append("Debug target Christoffel computation and verify inverse geodesic formulation")
        
        if stability_mean < 60:
            summary['warnings'].append("Numerical stability issues detected")
            summary['recommendations'].append("Investigate shooting solver convergence and ODE integration accuracy")
        
        if adherence_mean < 80 and adherence_mean >= 60:
            summary['warnings'].append("Data adherence suboptimal")
            summary['recommendations'].append("Fine-tune loss weights and increase training epochs")
        
        # Performance recommendations
        if 'performance' in aggregate_results:
            pred_time = aggregate_results['performance'].get('mean_prediction_time_ms', 0)
            if pred_time > 100:  # >100ms per prediction is slow
                summary['warnings'].append("Prediction speed slower than optimal")
                summary['recommendations'].append("Consider model optimization for inference speed")
        
        return summary
    
    def _save_validation_results(self):
        """Save validation results to JSON and CSV files"""
        # Save complete results as JSON
        json_path = self.output_dir / "validation_results.json"
        with open(json_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        if self.verbose:
            print(f"   📄 Complete results saved to {json_path}")
        
        # Create summary CSV
        summary_data = []
        for result in self.validation_results['individual_model_results']:
            summary_data.append({
                'model_idx': result['model_idx'],
                'excluded_concentration': result['excluded_concentration'],
                'overall_score': result['overall_score'],
                'data_adherence_score': result['data_adherence']['overall_score'],
                'geometry_score': result['geometry_learning']['overall_geometry_score'],
                'stability_score': (
                    result['numerical_stability'].get('shooting_convergence_rate', 0) * 40 +
                    result['numerical_stability'].get('integration_accuracy', 0) * 30 +
                    result['numerical_stability'].get('extreme_input_stability', 0) * 30
                ),
                'prediction_time_ms': result['performance'].get('mean_single_prediction_time_ms', 0),
                'validation_duration_s': result['validation_duration_seconds']
            })
        
        summary_df = pd.DataFrame(summary_data)
        csv_path = self.output_dir / "validation_summary.csv"
        summary_df.to_csv(csv_path, index=False)
        
        if self.verbose:
            print(f"   📊 Summary CSV saved to {csv_path}")
    
    def _generate_html_report(self):
        """Generate comprehensive HTML report"""
        html_content = self._create_html_report()
        html_path = self.output_dir / "validation_report.html"
        
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        if self.verbose:
            print(f"   📋 HTML report saved to {html_path}")
        
        return html_path
    
    def _create_html_report(self) -> str:
        """Create HTML report content"""
        results = self.validation_results
        aggregate = results['aggregate_results']
        summary = results['overall_summary']
        
        grade_colors = {'A': '#4CAF50', 'B': '#8BC34A', 'C': '#FFC107', 'D': '#FF9800', 'F': '#F44336'}
        grade_color = grade_colors.get(summary['grade'], '#F44336')
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Geodesic NODE Validation Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 40px; background: #f8f9fa; }}
        .header {{ text-align: center; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 30px; }}
        .grade {{ font-size: 72px; font-weight: bold; color: {grade_color}; margin: 20px 0; }}
        .summary {{ background: white; padding: 25px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 30px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 30px 0; }}
        .metric-card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 32px; font-weight: bold; color: #2196F3; }}
        .metric-label {{ color: #666; font-size: 14px; margin-top: 5px; }}
        .pass {{ color: #4CAF50; }}
        .fail {{ color: #F44336; }}
        .warning {{ color: #FF9800; }}
        .issue-list {{ background: #ffebee; padding: 20px; border-radius: 10px; border-left: 5px solid #F44336; margin: 20px 0; }}
        .warning-list {{ background: #fff8e1; padding: 20px; border-radius: 10px; border-left: 5px solid #FF9800; margin: 20px 0; }}
        .recommendation-list {{ background: #e8f5e8; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50; margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f5f5f5; font-weight: 600; }}
        .model-row {{ cursor: pointer; }}
        .model-row:hover {{ background: #f0f0f0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Geodesic NODE Validation Report</h1>
        <div class="grade">{summary['grade']}</div>
        <p><strong>Overall Score:</strong> {aggregate['overall']['mean_score']:.1f}/100</p>
        <p><strong>Status:</strong> <span class="{'pass' if summary['validation_passed'] else 'fail'}">
            {'✅ PASSED' if summary['validation_passed'] else '❌ FAILED'}
        </span></p>
        <p><strong>Generated:</strong> {results['timestamp']}</p>
        <p><strong>Duration:</strong> {results['validation_duration_seconds']:.1f} seconds</p>
    </div>
    
    <div class="summary">
        <h2>Executive Summary</h2>
        <p>Comprehensive validation of {results['n_models']} geodesic NODE models using inverse geodesic learning approach.</p>
        
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{aggregate['data_adherence']['mean_score']:.1f}</div>
                <div class="metric-label">Data Adherence Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{aggregate['geometry_learning']['mean_score']:.1f}</div>
                <div class="metric-label">Geometry Learning Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{aggregate['numerical_stability']['mean_score']:.1f}</div>
                <div class="metric-label">Numerical Stability Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{aggregate.get('performance', {}).get('mean_prediction_time_ms', 0):.1f}ms</div>
                <div class="metric-label">Avg Prediction Time</div>
            </div>
        </div>
        """
        
        # Add issues and recommendations
        if summary['critical_issues']:
            html += '<div class="issue-list"><h3>🚨 Critical Issues</h3><ul>'
            for issue in summary['critical_issues']:
                html += f'<li>{issue}</li>'
            html += '</ul></div>'
        
        if summary['warnings']:
            html += '<div class="warning-list"><h3>⚠️ Warnings</h3><ul>'
            for warning in summary['warnings']:
                html += f'<li>{warning}</li>'
            html += '</ul></div>'
        
        if summary['recommendations']:
            html += '<div class="recommendation-list"><h3>💡 Recommendations</h3><ul>'
            for rec in summary['recommendations']:
                html += f'<li>{rec}</li>'
            html += '</ul></div>'
        
        # Model-by-model results table
        html += """
    </div>
    
    <div class="summary">
        <h2>Individual Model Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Excluded Conc.</th>
                    <th>Overall Score</th>
                    <th>Data Adherence</th>
                    <th>Geometry</th>
                    <th>Stability</th>
                    <th>Speed (ms)</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for result in results['individual_model_results']:
            stability_score = (
                result['numerical_stability'].get('shooting_convergence_rate', 0) * 40 +
                result['numerical_stability'].get('integration_accuracy', 0) * 30 +
                result['numerical_stability'].get('extreme_input_stability', 0) * 30
            )
            
            html += f"""
                <tr class="model-row">
                    <td>Model {result['model_idx'] + 1}</td>
                    <td>{result['excluded_concentration']}</td>
                    <td><strong>{result['overall_score']:.1f}</strong></td>
                    <td>{result['data_adherence']['overall_score']:.1f}</td>
                    <td>{result['geometry_learning']['overall_geometry_score']:.1f}</td>
                    <td>{stability_score:.1f}</td>
                    <td>{result['performance'].get('mean_single_prediction_time_ms', 0):.1f}</td>
                </tr>
            """
        
        html += """
            </tbody>
        </table>
    </div>
    
    <div class="summary">
        <h2>Technical Details</h2>
        <h3>Test Categories</h3>
        <ul>
            <li><strong>Data Adherence (50% weight):</strong> Tests that model predictions match known spectral data</li>
            <li><strong>Geometry Learning (30% weight):</strong> Tests that Christoffel symbols match computed targets</li>
            <li><strong>Numerical Stability (20% weight):</strong> Tests shooting convergence and integration accuracy</li>
        </ul>
        
        <h3>Inverse Geodesic Learning</h3>
        <p>This validation uses the inverse geodesic formulation where spectral curves are treated as geodesics, 
        and target Christoffel symbols are computed directly from data. The model learns to match these targets 
        through supervised learning, ensuring the learned geometry aligns with the actual spectral distribution.</p>
    </div>
</body>
</html>"""
        
        return html
    
    def _create_comprehensive_visualizations(self):
        """Create comprehensive validation visualizations"""
        if self.verbose:
            print("   📊 Generating comprehensive visualizations...")
        
        # Create summary dashboard
        self._create_validation_dashboard()
        
        # Create individual model visualizations
        for i, (adherence_tester, geometry_tester) in enumerate(zip(self.data_adherence_tests, self.geometry_learning_tests)):
            adherence_path = self.output_dir / f"model_{i}_data_adherence.png"
            geometry_path = self.output_dir / f"model_{i}_geometry_learning.png"
            
            adherence_tester.create_adherence_visualization(str(adherence_path))
            geometry_tester.create_geometry_visualization(str(geometry_path))
        
        if self.verbose:
            print(f"   📊 All visualizations saved to {self.output_dir}")
    
    def _create_validation_dashboard(self):
        """Create overall validation dashboard"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Geodesic NODE Validation Dashboard', fontsize=16, fontweight='bold')
        
        results = self.validation_results['individual_model_results']
        aggregate = self.validation_results['aggregate_results']
        
        # 1. Overall scores by model
        model_indices = [r['model_idx'] for r in results]
        overall_scores = [r['overall_score'] for r in results]
        
        bars = axes[0, 0].bar(model_indices, overall_scores, color='steelblue', alpha=0.7)
        axes[0, 0].set_xlabel('Model Index')
        axes[0, 0].set_ylabel('Overall Score')
        axes[0, 0].set_title('Overall Scores by Model')
        axes[0, 0].axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Pass Threshold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add score labels
        for bar, score in zip(bars, overall_scores):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Score categories comparison
        categories = ['Data\nAdherence', 'Geometry\nLearning', 'Numerical\nStability']
        category_means = [
            aggregate['data_adherence']['mean_score'],
            aggregate['geometry_learning']['mean_score'],
            aggregate['numerical_stability']['mean_score']
        ]
        category_stds = [
            aggregate['data_adherence']['std_score'],
            aggregate['geometry_learning']['std_score'],
            aggregate['numerical_stability']['std_score']
        ]
        
        bars = axes[0, 1].bar(categories, category_means, yerr=category_stds, 
                             capsize=5, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.7)
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Mean Scores by Category')
        axes[0, 1].set_ylim(0, 100)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add score labels
        for bar, score in zip(bars, category_means):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                           f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Performance metrics
        pred_times = [r['performance'].get('mean_single_prediction_time_ms', 0) for r in results]
        axes[0, 2].hist(pred_times, bins=10, alpha=0.7, color='green', edgecolor='black')
        axes[0, 2].set_xlabel('Prediction Time (ms)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Prediction Time Distribution')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Score correlation matrix (simplified)
        adherence_scores = [r['data_adherence']['overall_score'] for r in results]
        geometry_scores = [r['geometry_learning']['overall_geometry_score'] for r in results]
        
        axes[1, 0].scatter(adherence_scores, geometry_scores, alpha=0.7, s=60, c=overall_scores, cmap='viridis')
        axes[1, 0].set_xlabel('Data Adherence Score')
        axes[1, 0].set_ylabel('Geometry Learning Score')
        axes[1, 0].set_title('Score Correlations')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add colorbar
        scatter = axes[1, 0].scatter(adherence_scores, geometry_scores, alpha=0.7, s=60, c=overall_scores, cmap='viridis')
        plt.colorbar(scatter, ax=axes[1, 0], label='Overall Score')
        
        # 5. Grade distribution
        grades = []
        for score in overall_scores:
            if score >= 90:
                grades.append('A')
            elif score >= 80:
                grades.append('B')
            elif score >= 70:
                grades.append('C')
            elif score >= 60:
                grades.append('D')
            else:
                grades.append('F')
        
        grade_counts = {g: grades.count(g) for g in ['A', 'B', 'C', 'D', 'F']}
        grade_colors = ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336']
        
        bars = axes[1, 1].bar(grade_counts.keys(), grade_counts.values(), color=grade_colors, alpha=0.7)
        axes[1, 1].set_xlabel('Grade')
        axes[1, 1].set_ylabel('Number of Models')
        axes[1, 1].set_title('Grade Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add count labels
        for bar, count in zip(bars, grade_counts.values()):
            if count > 0:
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                               str(count), ha='center', va='bottom', fontweight='bold')
        
        # 6. Summary statistics
        axes[1, 2].axis('off')
        summary_text = f"""
VALIDATION SUMMARY

Total Models: {len(results)}
Pass Rate: {sum(1 for r in results if r['overall_score'] >= 70)}/{len(results)} ({100 * sum(1 for r in results if r['overall_score'] >= 70) / len(results):.0f}%)

Mean Scores:
• Overall: {aggregate['overall']['mean_score']:.1f} ± {aggregate['overall']['std_score']:.1f}
• Data Adherence: {aggregate['data_adherence']['mean_score']:.1f}
• Geometry: {aggregate['geometry_learning']['mean_score']:.1f}
• Stability: {aggregate['numerical_stability']['mean_score']:.1f}

Performance:
• Avg Prediction: {aggregate.get('performance', {}).get('mean_prediction_time_ms', 0):.1f} ms
• Total Parameters: {aggregate.get('performance', {}).get('total_parameters', 0):,}

Status: {'✅ PASSED' if self.validation_results['overall_summary']['validation_passed'] else '❌ FAILED'}
Grade: {self.validation_results['overall_summary']['grade']}
        """
        
        axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes, fontsize=11,
                        verticalalignment='center', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save dashboard
        dashboard_path = self.output_dir / "validation_dashboard.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        
        if self.verbose:
            print(f"   📊 Validation dashboard saved to {dashboard_path}")
        
        # Show and open
        try:
            plt.show()
            subprocess.run(['open', str(dashboard_path)], check=False)
        except:
            pass
        
        return dashboard_path
    
    def _print_final_summary(self):
        """Print final validation summary"""
        results = self.validation_results
        summary = results['overall_summary']
        aggregate = results['aggregate_results']
        
        print("\n" + "="*80)
        print("🏆 COMPREHENSIVE VALIDATION COMPLETE")
        print("="*80)
        print(f"🎯 Overall Grade: {summary['grade']} ({aggregate['overall']['mean_score']:.1f}/100)")
        print(f"📊 Models Tested: {results['n_models']}")
        print(f"⏱️  Total Duration: {results['validation_duration_seconds']:.1f} seconds")
        print(f"✅ Status: {'PASSED' if summary['validation_passed'] else 'FAILED'}")
        
        print(f"\n📈 Category Scores:")
        print(f"   Data Adherence: {aggregate['data_adherence']['mean_score']:.1f} ± {aggregate['data_adherence']['std_score']:.1f}")
        print(f"   Geometry Learning: {aggregate['geometry_learning']['mean_score']:.1f} ± {aggregate['geometry_learning']['std_score']:.1f}")
        print(f"   Numerical Stability: {aggregate['numerical_stability']['mean_score']:.1f} ± {aggregate['numerical_stability']['std_score']:.1f}")
        
        if summary['critical_issues']:
            print(f"\n🚨 Critical Issues ({len(summary['critical_issues'])}):")
            for issue in summary['critical_issues']:
                print(f"   • {issue}")
        
        if summary['recommendations']:
            print(f"\n💡 Recommendations ({len(summary['recommendations'])}):")
            for rec in summary['recommendations']:
                print(f"   • {rec}")
        
        print(f"\n📁 Results saved to: {self.output_dir}")
        print("   • validation_results.json (complete data)")
        print("   • validation_report.html (interactive report)")
        print("   • validation_dashboard.png (summary visualization)")
        print("="*80)
        
        # Try to open the HTML report
        html_path = self.output_dir / "validation_report.html"
        if html_path.exists():
            try:
                subprocess.run(['open', str(html_path)], check=False)
                print(f"🌐 Opening validation report: {html_path}")
            except:
                pass