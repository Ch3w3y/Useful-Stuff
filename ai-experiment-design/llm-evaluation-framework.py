#!/usr/bin/env python3
"""
AI Experiment Design and LLM Evaluation Framework

Comprehensive framework for designing, executing, and evaluating AI/ML experiments
with focus on proper measurement of generative AI productivity gains and model performance.
"""

import json
import logging
import time
import random
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from enum import Enum
import uuid
import numpy as np
import pandas as pd
from pathlib import Path
import hashlib
import pickle

# Statistical libraries
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split

# ML evaluation
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentType(Enum):
    """Types of AI experiments"""
    AB_TEST = "ab_test"
    CONTROLLED_TRIAL = "controlled_trial"
    OBSERVATIONAL = "observational"
    LONGITUDINAL = "longitudinal"
    COMPARATIVE = "comparative"
    ABLATION = "ablation"

class EvaluationMetric(Enum):
    """Evaluation metrics for different AI tasks"""
    # NLP Metrics
    BLEU = "bleu"
    ROUGE = "rouge"
    BERT_SCORE = "bert_score"
    PERPLEXITY = "perplexity"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    
    # Classification Metrics
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1"
    AUC_ROC = "auc_roc"
    
    # Regression Metrics
    MSE = "mse"
    MAE = "mae"
    R2 = "r2"
    RMSE = "rmse"
    
    # Human Evaluation
    HUMAN_PREFERENCE = "human_preference"
    HELPFULNESS = "helpfulness"
    HARMFULNESS = "harmfulness"
    TRUTHFULNESS = "truthfulness"
    
    # Productivity Metrics
    TIME_TO_COMPLETION = "time_to_completion"
    OUTPUT_QUALITY = "output_quality"
    USER_SATISFACTION = "user_satisfaction"
    ERROR_RATE = "error_rate"

@dataclass
class ExperimentConfig:
    """Configuration for AI experiment"""
    name: str
    description: str
    experiment_type: ExperimentType
    hypothesis: str
    metrics: List[EvaluationMetric]
    duration_days: int = 30
    sample_size_target: int = 100
    significance_level: float = 0.05
    power: float = 0.8
    effect_size: float = 0.2
    randomization_seed: int = 42
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExperimentGroup:
    """Experimental group definition"""
    name: str
    description: str
    treatment: str
    size: int
    participants: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Measurement:
    """Single measurement in experiment"""
    participant_id: str
    group: str
    metric: EvaluationMetric
    value: Union[float, int, str, bool]
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExperimentResult:
    """Result of experiment analysis"""
    experiment_id: str
    metric: EvaluationMetric
    groups: List[str]
    means: Dict[str, float]
    std_devs: Dict[str, float]
    sample_sizes: Dict[str, int]
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    statistical_power: float
    is_significant: bool
    interpretation: str
    recommendations: List[str] = field(default_factory=list)

class StatisticalAnalyzer:
    """Statistical analysis for experiments"""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def t_test(self, group1: List[float], group2: List[float]) -> Dict[str, Any]:
        """Perform t-test between two groups"""
        statistic, p_value = ttest_ind(group1, group2)
        
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        
        # Cohen's d for effect size
        pooled_std = np.sqrt(((len(group1) - 1) * std1**2 + (len(group2) - 1) * std2**2) / 
                           (len(group1) + len(group2) - 2))
        cohens_d = (mean1 - mean2) / pooled_std
        
        # Confidence interval for difference in means
        se_diff = pooled_std * np.sqrt(1/len(group1) + 1/len(group2))
        df = len(group1) + len(group2) - 2
        t_critical = stats.t.ppf(1 - self.significance_level/2, df)
        margin_error = t_critical * se_diff
        mean_diff = mean1 - mean2
        ci_lower = mean_diff - margin_error
        ci_upper = mean_diff + margin_error
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'mean_group1': mean1,
            'mean_group2': mean2,
            'std_group1': std1,
            'std_group2': std2,
            'cohens_d': cohens_d,
            'confidence_interval': (ci_lower, ci_upper),
            'is_significant': p_value < self.significance_level
        }
    
    def mann_whitney_u(self, group1: List[float], group2: List[float]) -> Dict[str, Any]:
        """Non-parametric Mann-Whitney U test"""
        statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'median_group1': np.median(group1),
            'median_group2': np.median(group2),
            'is_significant': p_value < self.significance_level
        }
    
    def chi_square_test(self, contingency_table: np.ndarray) -> Dict[str, Any]:
        """Chi-square test for categorical data"""
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        return {
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'expected_frequencies': expected,
            'is_significant': p_value < self.significance_level
        }
    
    def calculate_power(self, effect_size: float, sample_size: int, alpha: float = 0.05) -> float:
        """Calculate statistical power for given parameters"""
        # Simplified power calculation for t-test
        from scipy.stats import norm
        
        # Critical value
        z_alpha = norm.ppf(1 - alpha/2)
        
        # Non-centrality parameter
        ncp = effect_size * np.sqrt(sample_size / 2)
        
        # Power
        power = 1 - norm.cdf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)
        
        return power
    
    def sample_size_calculation(self, effect_size: float, power: float = 0.8, 
                               alpha: float = 0.05) -> int:
        """Calculate required sample size"""
        from scipy.stats import norm
        
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(power)
        
        # Sample size per group
        n = 2 * ((z_alpha + z_beta) / effect_size)**2
        
        return int(np.ceil(n))

class LLMEvaluator:
    """Evaluator for Large Language Models"""
    
    def __init__(self):
        self.setup_clients()
    
    def setup_clients(self):
        """Setup LLM clients for evaluation"""
        self.openai_client = None
        self.anthropic_client = None
        
        if OPENAI_AVAILABLE:
            try:
                self.openai_client = openai.OpenAI()
            except Exception as e:
                logger.warning(f"Failed to setup OpenAI client: {e}")
        
        if ANTHROPIC_AVAILABLE:
            try:
                self.anthropic_client = anthropic.Anthropic()
            except Exception as e:
                logger.warning(f"Failed to setup Anthropic client: {e}")
    
    def evaluate_text_quality(self, generated_text: str, reference_text: str = "",
                             criteria: List[str] = None) -> Dict[str, float]:
        """Evaluate text quality using multiple metrics"""
        if criteria is None:
            criteria = ["coherence", "relevance", "accuracy", "fluency"]
        
        results = {}
        
        # Basic metrics
        results['length'] = len(generated_text)
        results['word_count'] = len(generated_text.split())
        results['sentence_count'] = len([s for s in generated_text.split('.') if s.strip()])
        
        # Readability (simplified)
        avg_words_per_sentence = results['word_count'] / max(results['sentence_count'], 1)
        results['readability_score'] = min(100, max(0, 100 - (avg_words_per_sentence - 15) * 2))
        
        # If reference text provided, calculate similarity
        if reference_text:
            results['reference_similarity'] = self._calculate_text_similarity(
                generated_text, reference_text
            )
        
        # LLM-based evaluation if available
        if self.openai_client:
            llm_scores = self._llm_evaluate_text(generated_text, criteria)
            results.update(llm_scores)
        
        return results
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity (simplified)"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1, text2).ratio()
    
    def _llm_evaluate_text(self, text: str, criteria: List[str]) -> Dict[str, float]:
        """Use LLM to evaluate text quality"""
        scores = {}
        
        for criterion in criteria:
            prompt = f"""
            Please evaluate the following text on the criterion of {criterion} on a scale of 1-10:
            
            Text: {text}
            
            Provide only a numeric score from 1-10, where 10 is the highest quality.
            Score:
            """
            
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0
                )
                
                score_text = response.choices[0].message.content.strip()
                score = float(score_text) / 10.0  # Normalize to 0-1
                scores[f'{criterion}_score'] = score
                
            except Exception as e:
                logger.warning(f"Failed to evaluate {criterion}: {e}")
                scores[f'{criterion}_score'] = 0.5  # Default neutral score
        
        return scores
    
    def evaluate_productivity_metrics(self, task_data: List[Dict]) -> Dict[str, Any]:
        """Evaluate productivity metrics from task completion data"""
        if not task_data:
            return {}
        
        # Extract metrics
        completion_times = [task['completion_time'] for task in task_data if 'completion_time' in task]
        quality_scores = [task['quality_score'] for task in task_data if 'quality_score' in task]
        error_rates = [task['error_rate'] for task in task_data if 'error_rate' in task]
        satisfaction_scores = [task['satisfaction'] for task in task_data if 'satisfaction' in task]
        
        results = {
            'total_tasks': len(task_data),
            'avg_completion_time': np.mean(completion_times) if completion_times else 0,
            'std_completion_time': np.std(completion_times) if completion_times else 0,
            'avg_quality_score': np.mean(quality_scores) if quality_scores else 0,
            'avg_error_rate': np.mean(error_rates) if error_rates else 0,
            'avg_satisfaction': np.mean(satisfaction_scores) if satisfaction_scores else 0,
            'productivity_index': 0
        }
        
        # Calculate productivity index (higher quality, lower time, lower errors = higher productivity)
        if completion_times and quality_scores and error_rates:
            # Normalize metrics (0-1 scale)
            norm_time = 1 - min(1, np.mean(completion_times) / 3600)  # Assume 1 hour max
            norm_quality = np.mean(quality_scores)
            norm_errors = 1 - min(1, np.mean(error_rates))
            
            results['productivity_index'] = (norm_quality + norm_time + norm_errors) / 3
        
        return results
    
    def comparative_evaluation(self, model_outputs: Dict[str, List[str]], 
                              evaluation_criteria: List[str]) -> Dict[str, Dict[str, float]]:
        """Compare multiple model outputs"""
        results = {}
        
        for model_name, outputs in model_outputs.items():
            model_scores = []
            
            for output in outputs:
                scores = self.evaluate_text_quality(output, criteria=evaluation_criteria)
                model_scores.append(scores)
            
            # Aggregate scores
            aggregated = {}
            if model_scores:
                for key in model_scores[0].keys():
                    values = [score[key] for score in model_scores if key in score]
                    if values:
                        aggregated[f'avg_{key}'] = np.mean(values)
                        aggregated[f'std_{key}'] = np.std(values)
            
            results[model_name] = aggregated
        
        return results

class ExperimentManager:
    """Manage AI experiments from design to analysis"""
    
    def __init__(self):
        self.experiments: Dict[str, Dict] = {}
        self.measurements: Dict[str, List[Measurement]] = {}
        self.analyzer = StatisticalAnalyzer()
        self.llm_evaluator = LLMEvaluator()
    
    def design_experiment(self, config: ExperimentConfig, 
                         groups: List[ExperimentGroup]) -> str:
        """Design a new experiment"""
        experiment_id = str(uuid.uuid4())
        
        # Validate experiment design
        self._validate_experiment_design(config, groups)
        
        # Calculate required sample size
        required_sample_size = self.analyzer.sample_size_calculation(
            config.effect_size, config.power, config.significance_level
        )
        
        total_planned = sum(group.size for group in groups)
        if total_planned < required_sample_size:
            logger.warning(f"Planned sample size ({total_planned}) is below recommended size ({required_sample_size})")
        
        experiment = {
            'id': experiment_id,
            'config': config,
            'groups': groups,
            'status': 'designed',
            'created_at': datetime.now(),
            'measurements': [],
            'results': None,
            'required_sample_size': required_sample_size
        }
        
        self.experiments[experiment_id] = experiment
        self.measurements[experiment_id] = []
        
        logger.info(f"Designed experiment: {config.name} ({experiment_id})")
        return experiment_id
    
    def _validate_experiment_design(self, config: ExperimentConfig, 
                                   groups: List[ExperimentGroup]):
        """Validate experiment design"""
        if len(groups) < 2:
            raise ValueError("Experiment must have at least 2 groups")
        
        if config.duration_days <= 0:
            raise ValueError("Experiment duration must be positive")
        
        if not config.metrics:
            raise ValueError("Experiment must have at least one evaluation metric")
        
        total_size = sum(group.size for group in groups)
        if total_size != config.sample_size_target:
            logger.warning(f"Total group sizes ({total_size}) don't match target ({config.sample_size_target})")
    
    def randomize_participants(self, experiment_id: str, 
                              participant_ids: List[str]) -> Dict[str, str]:
        """Randomize participants to experimental groups"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        groups = experiment['groups']
        config = experiment['config']
        
        # Set random seed for reproducibility
        random.seed(config.randomization_seed)
        np.random.seed(config.randomization_seed)
        
        # Shuffle participants
        shuffled_participants = participant_ids.copy()
        random.shuffle(shuffled_participants)
        
        # Assign to groups
        assignments = {}
        start_idx = 0
        
        for group in groups:
            end_idx = start_idx + group.size
            group_participants = shuffled_participants[start_idx:end_idx]
            
            for participant in group_participants:
                assignments[participant] = group.name
            
            group.participants.extend(group_participants)
            start_idx = end_idx
        
        experiment['status'] = 'randomized'
        logger.info(f"Randomized {len(assignments)} participants for experiment {experiment_id}")
        
        return assignments
    
    def record_measurement(self, experiment_id: str, participant_id: str,
                          group: str, metric: EvaluationMetric, value: Any,
                          context: Dict[str, Any] = None):
        """Record a measurement during the experiment"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        measurement = Measurement(
            participant_id=participant_id,
            group=group,
            metric=metric,
            value=value,
            context=context or {}
        )
        
        self.measurements[experiment_id].append(measurement)
        logger.debug(f"Recorded measurement for {participant_id}: {metric.value} = {value}")
    
    def batch_record_measurements(self, experiment_id: str, 
                                 measurements_data: List[Dict]):
        """Record multiple measurements at once"""
        for data in measurements_data:
            self.record_measurement(
                experiment_id=experiment_id,
                participant_id=data['participant_id'],
                group=data['group'],
                metric=EvaluationMetric(data['metric']),
                value=data['value'],
                context=data.get('context', {})
            )
    
    def analyze_experiment(self, experiment_id: str) -> Dict[str, ExperimentResult]:
        """Analyze experiment results"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        measurements = self.measurements[experiment_id]
        
        if not measurements:
            raise ValueError("No measurements found for experiment")
        
        results = {}
        
        # Group measurements by metric
        measurements_by_metric = {}
        for measurement in measurements:
            metric = measurement.metric
            if metric not in measurements_by_metric:
                measurements_by_metric[metric] = {}
            
            group = measurement.group
            if group not in measurements_by_metric[metric]:
                measurements_by_metric[metric][group] = []
            
            measurements_by_metric[metric][group].append(measurement.value)
        
        # Analyze each metric
        for metric, group_data in measurements_by_metric.items():
            result = self._analyze_metric(experiment_id, metric, group_data)
            results[metric.value] = result
        
        experiment['results'] = results
        experiment['status'] = 'analyzed'
        experiment['analyzed_at'] = datetime.now()
        
        return results
    
    def _analyze_metric(self, experiment_id: str, metric: EvaluationMetric,
                       group_data: Dict[str, List]) -> ExperimentResult:
        """Analyze a single metric across groups"""
        experiment = self.experiments[experiment_id]
        
        groups = list(group_data.keys())
        
        # Calculate descriptive statistics
        means = {group: np.mean(values) for group, values in group_data.items()}
        std_devs = {group: np.std(values, ddof=1) for group, values in group_data.items()}
        sample_sizes = {group: len(values) for group, values in group_data.items()}
        
        # Perform statistical test (assuming 2-group comparison for simplicity)
        if len(groups) == 2:
            group1_values = group_data[groups[0]]
            group2_values = group_data[groups[1]]
            
            # Check if data is numeric
            if all(isinstance(v, (int, float)) for v in group1_values + group2_values):
                # Perform t-test
                test_results = self.analyzer.t_test(group1_values, group2_values)
                p_value = test_results['p_value']
                effect_size = abs(test_results['cohens_d'])
                confidence_interval = test_results['confidence_interval']
                is_significant = test_results['is_significant']
                
                # Calculate statistical power
                avg_sample_size = np.mean(list(sample_sizes.values()))
                statistical_power = self.analyzer.calculate_power(effect_size, int(avg_sample_size))
                
            else:
                # Handle categorical data or use non-parametric tests
                p_value = 1.0
                effect_size = 0.0
                confidence_interval = (0.0, 0.0)
                is_significant = False
                statistical_power = 0.0
        else:
            # Multi-group analysis would require ANOVA or other methods
            p_value = 1.0
            effect_size = 0.0
            confidence_interval = (0.0, 0.0)
            is_significant = False
            statistical_power = 0.0
        
        # Generate interpretation
        interpretation = self._interpret_results(
            metric, means, p_value, effect_size, is_significant
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            metric, means, effect_size, statistical_power, sample_sizes
        )
        
        return ExperimentResult(
            experiment_id=experiment_id,
            metric=metric,
            groups=groups,
            means=means,
            std_devs=std_devs,
            sample_sizes=sample_sizes,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=confidence_interval,
            statistical_power=statistical_power,
            is_significant=is_significant,
            interpretation=interpretation,
            recommendations=recommendations
        )
    
    def _interpret_results(self, metric: EvaluationMetric, means: Dict[str, float],
                          p_value: float, effect_size: float, 
                          is_significant: bool) -> str:
        """Generate interpretation of results"""
        best_group = max(means.keys(), key=lambda k: means[k])
        worst_group = min(means.keys(), key=lambda k: means[k])
        
        interpretation = f"For {metric.value}:\n"
        interpretation += f"- Best performing group: {best_group} (mean = {means[best_group]:.3f})\n"
        interpretation += f"- Worst performing group: {worst_group} (mean = {means[worst_group]:.3f})\n"
        
        if is_significant:
            interpretation += f"- Statistically significant difference (p = {p_value:.3f})\n"
        else:
            interpretation += f"- No statistically significant difference (p = {p_value:.3f})\n"
        
        # Effect size interpretation
        if effect_size < 0.2:
            effect_desc = "negligible"
        elif effect_size < 0.5:
            effect_desc = "small"
        elif effect_size < 0.8:
            effect_desc = "medium"
        else:
            effect_desc = "large"
        
        interpretation += f"- Effect size: {effect_size:.3f} ({effect_desc})"
        
        return interpretation
    
    def _generate_recommendations(self, metric: EvaluationMetric, means: Dict[str, float],
                                 effect_size: float, statistical_power: float,
                                 sample_sizes: Dict[str, int]) -> List[str]:
        """Generate recommendations based on results"""
        recommendations = []
        
        # Power analysis recommendations
        if statistical_power < 0.8:
            recommendations.append(
                f"Statistical power is low ({statistical_power:.2f}). "
                "Consider increasing sample size for more reliable results."
            )
        
        # Sample size recommendations
        min_sample_size = min(sample_sizes.values())
        if min_sample_size < 30:
            recommendations.append(
                "Small sample sizes detected. Results may not be generalizable."
            )
        
        # Effect size recommendations
        if effect_size > 0.5:
            best_group = max(means.keys(), key=lambda k: means[k])
            recommendations.append(
                f"Large effect size detected. Consider implementing the {best_group} treatment."
            )
        elif effect_size < 0.2:
            recommendations.append(
                "Small effect size. Consider if the practical significance justifies implementation."
            )
        
        # Metric-specific recommendations
        if metric in [EvaluationMetric.TIME_TO_COMPLETION, EvaluationMetric.ERROR_RATE]:
            # For these metrics, lower is better
            best_group = min(means.keys(), key=lambda k: means[k])
            recommendations.append(f"Lower {metric.value} is better. {best_group} shows best performance.")
        
        return recommendations
    
    def generate_experiment_report(self, experiment_id: str) -> str:
        """Generate comprehensive experiment report"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        config = experiment['config']
        
        report = f"""
# Experiment Report: {config.name}

## Experiment Overview
- **Experiment ID**: {experiment_id}
- **Type**: {config.experiment_type.value}
- **Hypothesis**: {config.hypothesis}
- **Duration**: {config.duration_days} days
- **Status**: {experiment['status']}

## Experimental Design
- **Sample Size Target**: {config.sample_size_target}
- **Significance Level**: {config.significance_level}
- **Statistical Power Target**: {config.power}
- **Effect Size Target**: {config.effect_size}

## Groups
"""
        
        for group in experiment['groups']:
            report += f"- **{group.name}**: {group.description} (n={group.size})\n"
        
        if experiment.get('results'):
            report += "\n## Results Summary\n"
            
            for metric_name, result in experiment['results'].items():
                report += f"\n### {metric_name}\n"
                report += f"**Statistical Test Results:**\n"
                report += f"- p-value: {result.p_value:.4f}\n"
                report += f"- Effect size: {result.effect_size:.3f}\n"
                report += f"- Statistical power: {result.statistical_power:.3f}\n"
                report += f"- Significant: {'Yes' if result.is_significant else 'No'}\n\n"
                
                report += f"**Group Means:**\n"
                for group, mean in result.means.items():
                    report += f"- {group}: {mean:.3f} (±{result.std_devs[group]:.3f})\n"
                
                report += f"\n**Interpretation:**\n{result.interpretation}\n"
                
                if result.recommendations:
                    report += f"\n**Recommendations:**\n"
                    for rec in result.recommendations:
                        report += f"- {rec}\n"
        
        return report
    
    def export_experiment_data(self, experiment_id: str) -> Dict[str, Any]:
        """Export experiment data for external analysis"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        measurements = self.measurements[experiment_id]
        
        # Convert measurements to DataFrame-friendly format
        measurement_data = []
        for measurement in measurements:
            measurement_data.append({
                'participant_id': measurement.participant_id,
                'group': measurement.group,
                'metric': measurement.metric.value,
                'value': measurement.value,
                'timestamp': measurement.timestamp.isoformat(),
                'context': json.dumps(measurement.context)
            })
        
        return {
            'experiment': {
                'id': experiment['id'],
                'config': {
                    'name': experiment['config'].name,
                    'description': experiment['config'].description,
                    'experiment_type': experiment['config'].experiment_type.value,
                    'hypothesis': experiment['config'].hypothesis,
                    'metrics': [m.value for m in experiment['config'].metrics],
                    'duration_days': experiment['config'].duration_days,
                    'sample_size_target': experiment['config'].sample_size_target,
                    'significance_level': experiment['config'].significance_level,
                    'power': experiment['config'].power,
                    'effect_size': experiment['config'].effect_size
                },
                'groups': [
                    {
                        'name': group.name,
                        'description': group.description,
                        'treatment': group.treatment,
                        'size': group.size,
                        'participants': group.participants
                    }
                    for group in experiment['groups']
                ],
                'status': experiment['status'],
                'created_at': experiment['created_at'].isoformat()
            },
            'measurements': measurement_data,
            'results': experiment.get('results', {})
        }

# Demo and example usage
def demo_experiment_framework():
    """Demonstrate the experiment framework"""
    print("=== AI Experiment Design & Evaluation Framework Demo ===\n")
    
    # Create experiment manager
    manager = ExperimentManager()
    
    # Design experiment
    print("1. Designing productivity experiment...")
    
    config = ExperimentConfig(
        name="LLM-Assisted Content Creation Productivity",
        description="Measuring productivity gains from using LLMs for content creation",
        experiment_type=ExperimentType.AB_TEST,
        hypothesis="Writers using LLM assistance will complete tasks 20% faster with equal quality",
        metrics=[
            EvaluationMetric.TIME_TO_COMPLETION,
            EvaluationMetric.OUTPUT_QUALITY,
            EvaluationMetric.USER_SATISFACTION
        ],
        duration_days=14,
        sample_size_target=100,
        effect_size=0.3  # Expecting medium effect size
    )
    
    groups = [
        ExperimentGroup(
            name="control",
            description="Writers without LLM assistance",
            treatment="standard_writing_tools",
            size=50
        ),
        ExperimentGroup(
            name="treatment",
            description="Writers with LLM assistance",
            treatment="llm_assisted_writing",
            size=50
        )
    ]
    
    experiment_id = manager.design_experiment(config, groups)
    print(f"Experiment designed: {experiment_id}")
    
    # Simulate participant randomization
    print("\n2. Randomizing participants...")
    participants = [f"participant_{i:03d}" for i in range(100)]
    assignments = manager.randomize_participants(experiment_id, participants)
    print(f"Randomized {len(assignments)} participants")
    
    # Simulate data collection
    print("\n3. Simulating data collection...")
    
    # Generate synthetic data
    np.random.seed(42)
    
    for participant, group in assignments.items():
        # Simulate completion time (minutes)
        if group == "control":
            completion_time = np.random.normal(60, 15)  # 60 min average
        else:
            completion_time = np.random.normal(48, 12)  # 20% faster with LLM
        
        completion_time = max(15, completion_time)  # Minimum 15 minutes
        
        # Simulate quality score (1-10)
        if group == "control":
            quality_score = np.random.normal(7.0, 1.2)
        else:
            quality_score = np.random.normal(7.2, 1.1)  # Slightly better quality
        
        quality_score = np.clip(quality_score, 1, 10)
        
        # Simulate satisfaction (1-10)
        if group == "control":
            satisfaction = np.random.normal(6.5, 1.5)
        else:
            satisfaction = np.random.normal(8.0, 1.2)  # Higher satisfaction with LLM
        
        satisfaction = np.clip(satisfaction, 1, 10)
        
        # Record measurements
        manager.record_measurement(
            experiment_id, participant, group,
            EvaluationMetric.TIME_TO_COMPLETION, completion_time
        )
        manager.record_measurement(
            experiment_id, participant, group,
            EvaluationMetric.OUTPUT_QUALITY, quality_score
        )
        manager.record_measurement(
            experiment_id, participant, group,
            EvaluationMetric.USER_SATISFACTION, satisfaction
        )
    
    print("Data collection completed")
    
    # Analyze results
    print("\n4. Analyzing results...")
    results = manager.analyze_experiment(experiment_id)
    
    for metric_name, result in results.items():
        print(f"\n--- {metric_name} ---")
        print(f"p-value: {result.p_value:.4f}")
        print(f"Effect size: {result.effect_size:.3f}")
        print(f"Significant: {'Yes' if result.is_significant else 'No'}")
        
        for group, mean in result.means.items():
            print(f"{group}: {mean:.2f} ± {result.std_devs[group]:.2f}")
    
    # Generate report
    print("\n5. Generating report...")
    report = manager.generate_experiment_report(experiment_id)
    print("Report generated (truncated):")
    print(report[:500] + "..." if len(report) > 500 else report)
    
    # LLM evaluation demo
    print("\n6. LLM evaluation demo...")
    llm_evaluator = LLMEvaluator()
    
    # Sample outputs to compare
    model_outputs = {
        "model_a": [
            "The quick brown fox jumps over the lazy dog. This is a well-known pangram.",
            "Artificial intelligence is transforming industries across the globe."
        ],
        "model_b": [
            "A fast brown fox leaps above a sleepy canine. This sentence uses all letters.",
            "AI technology is revolutionizing business processes worldwide."
        ]
    }
    
    comparison_results = llm_evaluator.comparative_evaluation(
        model_outputs, ["coherence", "fluency"]
    )
    
    print("Model comparison results:")
    for model, scores in comparison_results.items():
        print(f"{model}: {scores}")
    
    return manager, experiment_id

if __name__ == "__main__":
    # Run demo
    demo_experiment_framework() 