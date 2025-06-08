# Data Quality Tools and Framework

## Overview

Comprehensive toolkit for data quality assessment, validation, profiling, and monitoring. Includes automated quality checks, data lineage tracking, and quality reporting systems for production data pipelines.

## Table of Contents

- [Data Quality Framework](#data-quality-framework)
- [Data Profiling](#data-profiling)
- [Data Validation](#data-validation)
- [Quality Monitoring](#quality-monitoring)
- [Automated Testing](#automated-testing)
- [Quality Metrics](#quality-metrics)
- [Tools and Libraries](#tools-and-libraries)
- [Best Practices](#best-practices)

## Data Quality Framework

### Quality Dimensions

```python
from enum import Enum
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class QualityDimension(Enum):
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    TIMELINESS = "timeliness"
    INTEGRITY = "integrity"

class DataQualityFramework:
    """Comprehensive data quality assessment framework"""
    
    def __init__(self):
        self.quality_rules = {}
        self.quality_results = {}
        
    def add_quality_rule(self, rule_name: str, dimension: QualityDimension, 
                        rule_function: callable):
        """Add a quality rule to the framework"""
        self.quality_rules[rule_name] = {
            'dimension': dimension,
            'function': rule_function,
            'active': True
        }
    
    def assess_quality(self, data: pd.DataFrame, rules: List[str] = None):
        """Assess data quality using specified rules"""
        if rules is None:
            rules = list(self.quality_rules.keys())
            
        results = {}
        for rule_name in rules:
            if rule_name in self.quality_rules and self.quality_rules[rule_name]['active']:
                try:
                    rule_function = self.quality_rules[rule_name]['function']
                    result = rule_function(data)
                    results[rule_name] = {
                        'dimension': self.quality_rules[rule_name]['dimension'].value,
                        'passed': result['passed'],
                        'score': result['score'],
                        'details': result.get('details', {}),
                        'timestamp': datetime.now()
                    }
                except Exception as e:
                    results[rule_name] = {
                        'dimension': self.quality_rules[rule_name]['dimension'].value,
                        'passed': False,
                        'score': 0.0,
                        'error': str(e),
                        'timestamp': datetime.now()
                    }
        
        self.quality_results = results
        return results
    
    def generate_quality_report(self):
        """Generate comprehensive quality report"""
        if not self.quality_results:
            return "No quality assessment results available"
            
        report = []
        report.append("=" * 50)
        report.append("DATA QUALITY ASSESSMENT REPORT")
        report.append("=" * 50)
        
        # Summary by dimension
        dimension_summary = {}
        for rule_name, result in self.quality_results.items():
            dimension = result['dimension']
            if dimension not in dimension_summary:
                dimension_summary[dimension] = {'passed': 0, 'total': 0, 'scores': []}
            
            dimension_summary[dimension]['total'] += 1
            if result['passed']:
                dimension_summary[dimension]['passed'] += 1
            dimension_summary[dimension]['scores'].append(result['score'])
        
        for dimension, summary in dimension_summary.items():
            avg_score = np.mean(summary['scores'])
            pass_rate = summary['passed'] / summary['total'] * 100
            report.append(f"\n{dimension.upper()}:")
            report.append(f"  Pass Rate: {pass_rate:.1f}% ({summary['passed']}/{summary['total']})")
            report.append(f"  Average Score: {avg_score:.3f}")
        
        # Detailed results
        report.append("\n" + "=" * 30)
        report.append("DETAILED RESULTS")
        report.append("=" * 30)
        
        for rule_name, result in self.quality_results.items():
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            report.append(f"\n{rule_name}: {status}")
            report.append(f"  Dimension: {result['dimension']}")
            report.append(f"  Score: {result['score']:.3f}")
            if 'details' in result:
                for key, value in result['details'].items():
                    report.append(f"  {key}: {value}")
        
        return "\n".join(report)
```

### Core Quality Rules

```python
class CoreQualityRules:
    """Standard data quality rules"""
    
    @staticmethod
    def completeness_check(data: pd.DataFrame, threshold: float = 0.95):
        """Check data completeness (non-null percentage)"""
        total_cells = data.shape[0] * data.shape[1]
        non_null_cells = total_cells - data.isnull().sum().sum()
        completeness_score = non_null_cells / total_cells
        
        return {
            'passed': completeness_score >= threshold,
            'score': completeness_score,
            'details': {
                'completeness_percentage': f"{completeness_score:.1%}",
                'total_cells': total_cells,
                'missing_cells': total_cells - non_null_cells,
                'threshold': f"{threshold:.1%}"
            }
        }
    
    @staticmethod
    def uniqueness_check(data: pd.DataFrame, columns: List[str], threshold: float = 0.95):
        """Check uniqueness of specified columns"""
        if not columns:
            columns = data.columns.tolist()
            
        results = {}
        overall_score = []
        
        for col in columns:
            if col in data.columns:
                total_rows = len(data[col].dropna())
                unique_rows = data[col].nunique()
                uniqueness_score = unique_rows / total_rows if total_rows > 0 else 0
                
                results[col] = {
                    'uniqueness_score': uniqueness_score,
                    'total_values': total_rows,
                    'unique_values': unique_rows,
                    'duplicates': total_rows - unique_rows
                }
                overall_score.append(uniqueness_score)
        
        avg_score = np.mean(overall_score) if overall_score else 0
        
        return {
            'passed': avg_score >= threshold,
            'score': avg_score,
            'details': results
        }
    
    @staticmethod
    def validity_check(data: pd.DataFrame, column: str, valid_values: List[Any]):
        """Check if column values are in valid set"""
        if column not in data.columns:
            return {'passed': False, 'score': 0.0, 'details': {'error': f'Column {column} not found'}}
        
        valid_mask = data[column].isin(valid_values)
        validity_score = valid_mask.sum() / len(data[column].dropna())
        
        invalid_values = data[column][~valid_mask & data[column].notna()].unique()
        
        return {
            'passed': validity_score == 1.0,
            'score': validity_score,
            'details': {
                'valid_percentage': f"{validity_score:.1%}",
                'invalid_count': len(data[column]) - valid_mask.sum(),
                'invalid_values': invalid_values.tolist() if len(invalid_values) > 0 else []
            }
        }
    
    @staticmethod
    def range_check(data: pd.DataFrame, column: str, min_val: float = None, max_val: float = None):
        """Check if numeric values are within specified range"""
        if column not in data.columns:
            return {'passed': False, 'score': 0.0, 'details': {'error': f'Column {column} not found'}}
        
        numeric_data = pd.to_numeric(data[column], errors='coerce').dropna()
        
        if len(numeric_data) == 0:
            return {'passed': False, 'score': 0.0, 'details': {'error': 'No numeric values found'}}
        
        in_range_mask = pd.Series([True] * len(numeric_data))
        
        if min_val is not None:
            in_range_mask &= (numeric_data >= min_val)
        if max_val is not None:
            in_range_mask &= (numeric_data <= max_val)
        
        range_score = in_range_mask.sum() / len(numeric_data)
        
        return {
            'passed': range_score == 1.0,
            'score': range_score,
            'details': {
                'in_range_percentage': f"{range_score:.1%}",
                'out_of_range_count': len(numeric_data) - in_range_mask.sum(),
                'min_value': numeric_data.min(),
                'max_value': numeric_data.max(),
                'specified_min': min_val,
                'specified_max': max_val
            }
        }
    
    @staticmethod
    def pattern_check(data: pd.DataFrame, column: str, pattern: str):
        """Check if string values match specified regex pattern"""
        import re
        
        if column not in data.columns:
            return {'passed': False, 'score': 0.0, 'details': {'error': f'Column {column} not found'}}
        
        string_data = data[column].astype(str).dropna()
        pattern_mask = string_data.str.match(pattern, na=False)
        pattern_score = pattern_mask.sum() / len(string_data) if len(string_data) > 0 else 0
        
        return {
            'passed': pattern_score == 1.0,
            'score': pattern_score,
            'details': {
                'pattern_match_percentage': f"{pattern_score:.1%}",
                'non_matching_count': len(string_data) - pattern_mask.sum(),
                'pattern': pattern
            }
        }
```

## Data Profiling

### Comprehensive Data Profiler

```python
class DataProfiler:
    """Comprehensive data profiling tool"""
    
    def __init__(self):
        self.profile_results = {}
    
    def profile_dataset(self, data: pd.DataFrame, sample_size: int = None):
        """Generate comprehensive data profile"""
        if sample_size and len(data) > sample_size:
            data_sample = data.sample(n=sample_size, random_state=42)
        else:
            data_sample = data
        
        profile = {
            'dataset_info': self._get_dataset_info(data),
            'column_profiles': {},
            'correlations': self._get_correlations(data_sample),
            'data_types': self._analyze_data_types(data_sample),
            'quality_summary': self._get_quality_summary(data_sample)
        }
        
        for column in data_sample.columns:
            profile['column_profiles'][column] = self._profile_column(data_sample, column)
        
        self.profile_results = profile
        return profile
    
    def _get_dataset_info(self, data: pd.DataFrame):
        """Get basic dataset information"""
        return {
            'shape': data.shape,
            'memory_usage': data.memory_usage(deep=True).sum(),
            'dtypes': data.dtypes.value_counts().to_dict(),
            'missing_cells': data.isnull().sum().sum(),
            'missing_percentage': (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
        }
    
    def _profile_column(self, data: pd.DataFrame, column: str):
        """Profile individual column"""
        col_data = data[column]
        
        profile = {
            'dtype': str(col_data.dtype),
            'non_null_count': col_data.count(),
            'null_count': col_data.isnull().sum(),
            'null_percentage': (col_data.isnull().sum() / len(col_data)) * 100,
            'unique_count': col_data.nunique(),
            'unique_percentage': (col_data.nunique() / col_data.count()) * 100 if col_data.count() > 0 else 0
        }
        
        # Numeric column analysis
        if pd.api.types.is_numeric_dtype(col_data):
            numeric_data = col_data.dropna()
            if len(numeric_data) > 0:
                profile.update({
                    'min': numeric_data.min(),
                    'max': numeric_data.max(),
                    'mean': numeric_data.mean(),
                    'median': numeric_data.median(),
                    'std': numeric_data.std(),
                    'quartiles': {
                        'q1': numeric_data.quantile(0.25),
                        'q3': numeric_data.quantile(0.75)
                    },
                    'outliers': self._detect_outliers(numeric_data),
                    'distribution': self._analyze_distribution(numeric_data)
                })
        
        # String column analysis
        elif pd.api.types.is_string_dtype(col_data) or col_data.dtype == 'object':
            string_data = col_data.dropna().astype(str)
            if len(string_data) > 0:
                profile.update({
                    'avg_length': string_data.str.len().mean(),
                    'min_length': string_data.str.len().min(),
                    'max_length': string_data.str.len().max(),
                    'most_common': string_data.value_counts().head().to_dict(),
                    'patterns': self._detect_patterns(string_data)
                })
        
        # Datetime column analysis
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            datetime_data = col_data.dropna()
            if len(datetime_data) > 0:
                profile.update({
                    'min_date': datetime_data.min(),
                    'max_date': datetime_data.max(),
                    'date_range_days': (datetime_data.max() - datetime_data.min()).days,
                    'frequency_analysis': self._analyze_datetime_frequency(datetime_data)
                })
        
        return profile
    
    def _detect_outliers(self, data: pd.Series):
        """Detect outliers using IQR method"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        return {
            'count': len(outliers),
            'percentage': (len(outliers) / len(data)) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    def _analyze_distribution(self, data: pd.Series):
        """Analyze data distribution"""
        from scipy import stats
        
        try:
            # Normality test
            shapiro_stat, shapiro_p = stats.shapiro(data.sample(min(5000, len(data))))
            
            return {
                'skewness': stats.skew(data),
                'kurtosis': stats.kurtosis(data),
                'normality_test': {
                    'shapiro_statistic': shapiro_stat,
                    'shapiro_p_value': shapiro_p,
                    'is_normal': shapiro_p > 0.05
                }
            }
        except:
            return {'error': 'Could not analyze distribution'}
    
    def _detect_patterns(self, data: pd.Series):
        """Detect common patterns in string data"""
        patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^\+?1?-?\.?\s?\(?\d{3}\)?-?\.?\s?\d{3}-?\.?\s?\d{4}$',
            'url': r'^https?://[^\s/$.?#].[^\s]*$',
            'numeric': r'^\d+$',
            'alphanumeric': r'^[a-zA-Z0-9]+$',
            'date_iso': r'^\d{4}-\d{2}-\d{2}$'
        }
        
        pattern_matches = {}
        for pattern_name, pattern in patterns.items():
            matches = data.str.match(pattern, na=False)
            match_count = matches.sum()
            if match_count > 0:
                pattern_matches[pattern_name] = {
                    'count': match_count,
                    'percentage': (match_count / len(data)) * 100
                }
        
        return pattern_matches
    
    def _get_correlations(self, data: pd.DataFrame):
        """Calculate correlations between numeric columns"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 1:
            corr_matrix = data[numeric_columns].corr()
            
            # Find high correlations
            high_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:  # High correlation threshold
                        high_correlations.append({
                            'column1': col1,
                            'column2': col2,
                            'correlation': corr_value
                        })
            
            return {
                'correlation_matrix': corr_matrix.to_dict(),
                'high_correlations': high_correlations
            }
        
        return {'message': 'Insufficient numeric columns for correlation analysis'}
    
    def generate_profile_report(self):
        """Generate human-readable profile report"""
        if not self.profile_results:
            return "No profiling results available"
        
        report = []
        report.append("=" * 60)
        report.append("DATA PROFILING REPORT")
        report.append("=" * 60)
        
        # Dataset overview
        info = self.profile_results['dataset_info']
        report.append(f"\nDATASET OVERVIEW:")
        report.append(f"  Shape: {info['shape'][0]} rows × {info['shape'][1]} columns")
        report.append(f"  Memory Usage: {info['memory_usage'] / 1024 / 1024:.2f} MB")
        report.append(f"  Missing Data: {info['missing_percentage']:.1f}% ({info['missing_cells']} cells)")
        
        # Data types summary
        report.append(f"\n  Data Types:")
        for dtype, count in info['dtypes'].items():
            report.append(f"    {dtype}: {count} columns")
        
        # Column profiles
        report.append(f"\nCOLUMN PROFILES:")
        for col_name, profile in self.profile_results['column_profiles'].items():
            report.append(f"\n  {col_name} ({profile['dtype']}):")
            report.append(f"    Non-null: {profile['non_null_count']} ({100-profile['null_percentage']:.1f}%)")
            report.append(f"    Unique: {profile['unique_count']} ({profile['unique_percentage']:.1f}%)")
            
            if 'min' in profile:  # Numeric column
                report.append(f"    Range: {profile['min']:.2f} to {profile['max']:.2f}")
                report.append(f"    Mean: {profile['mean']:.2f}, Median: {profile['median']:.2f}")
                if profile['outliers']['count'] > 0:
                    report.append(f"    Outliers: {profile['outliers']['count']} ({profile['outliers']['percentage']:.1f}%)")
            
            elif 'avg_length' in profile:  # String column
                report.append(f"    Avg Length: {profile['avg_length']:.1f} chars")
                report.append(f"    Length Range: {profile['min_length']} to {profile['max_length']} chars")
        
        return "\n".join(report)
```

## Data Validation

### Validation Engine

```python
class DataValidationEngine:
    """Advanced data validation engine with custom rules"""
    
    def __init__(self):
        self.validation_rules = {}
        self.validation_history = []
    
    def add_validation_rule(self, rule_name: str, rule_config: dict):
        """Add validation rule with configuration"""
        self.validation_rules[rule_name] = rule_config
    
    def validate_batch(self, data: pd.DataFrame, rule_names: List[str] = None):
        """Validate data batch against specified rules"""
        if rule_names is None:
            rule_names = list(self.validation_rules.keys())
        
        validation_results = {
            'timestamp': datetime.now(),
            'data_shape': data.shape,
            'rules_applied': rule_names,
            'results': {},
            'overall_status': 'PASS',
            'critical_failures': []
        }
        
        for rule_name in rule_names:
            if rule_name not in self.validation_rules:
                continue
                
            rule_config = self.validation_rules[rule_name]
            try:
                result = self._apply_validation_rule(data, rule_config)
                validation_results['results'][rule_name] = result
                
                # Check for critical failures
                if rule_config.get('critical', False) and not result['passed']:
                    validation_results['critical_failures'].append(rule_name)
                    validation_results['overall_status'] = 'CRITICAL_FAIL'
                elif not result['passed'] and validation_results['overall_status'] == 'PASS':
                    validation_results['overall_status'] = 'FAIL'
                    
            except Exception as e:
                validation_results['results'][rule_name] = {
                    'passed': False,
                    'error': str(e),
                    'rule_type': rule_config.get('type', 'unknown')
                }
                validation_results['overall_status'] = 'ERROR'
        
        self.validation_history.append(validation_results)
        return validation_results
    
    def _apply_validation_rule(self, data: pd.DataFrame, rule_config: dict):
        """Apply individual validation rule"""
        rule_type = rule_config['type']
        
        if rule_type == 'schema_validation':
            return self._validate_schema(data, rule_config)
        elif rule_type == 'business_rule':
            return self._validate_business_rule(data, rule_config)
        elif rule_type == 'data_quality':
            return self._validate_data_quality(data, rule_config)
        elif rule_type == 'referential_integrity':
            return self._validate_referential_integrity(data, rule_config)
        else:
            raise ValueError(f"Unknown rule type: {rule_type}")
    
    def _validate_schema(self, data: pd.DataFrame, rule_config: dict):
        """Validate data schema"""
        expected_columns = rule_config.get('columns', [])
        expected_types = rule_config.get('types', {})
        
        issues = []
        
        # Check required columns
        missing_columns = set(expected_columns) - set(data.columns)
        if missing_columns:
            issues.append(f"Missing columns: {list(missing_columns)}")
        
        # Check data types
        for column, expected_type in expected_types.items():
            if column in data.columns:
                actual_type = str(data[column].dtype)
                if expected_type not in actual_type:
                    issues.append(f"Column {column}: expected {expected_type}, got {actual_type}")
        
        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'rule_type': 'schema_validation'
        }
    
    def _validate_business_rule(self, data: pd.DataFrame, rule_config: dict):
        """Validate business rules"""
        rule_expression = rule_config['expression']
        description = rule_config.get('description', 'Business rule validation')
        
        try:
            # Evaluate business rule expression
            violations = data.query(f"not ({rule_expression})")
            violation_count = len(violations)
            
            return {
                'passed': violation_count == 0,
                'violations': violation_count,
                'violation_percentage': (violation_count / len(data)) * 100,
                'description': description,
                'rule_type': 'business_rule'
            }
        except Exception as e:
            return {
                'passed': False,
                'error': f"Error evaluating business rule: {str(e)}",
                'rule_type': 'business_rule'
            }
    
    def create_validation_report(self, validation_results: dict):
        """Create detailed validation report"""
        report = []
        report.append("=" * 50)
        report.append("DATA VALIDATION REPORT")
        report.append("=" * 50)
        
        report.append(f"\nValidation Timestamp: {validation_results['timestamp']}")
        report.append(f"Data Shape: {validation_results['data_shape']}")
        report.append(f"Overall Status: {validation_results['overall_status']}")
        
        if validation_results['critical_failures']:
            report.append(f"\n⚠️  CRITICAL FAILURES:")
            for failure in validation_results['critical_failures']:
                report.append(f"  - {failure}")
        
        report.append(f"\nRULE RESULTS:")
        for rule_name, result in validation_results['results'].items():
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            report.append(f"\n  {rule_name}: {status}")
            
            if not result['passed']:
                if 'issues' in result:
                    for issue in result['issues']:
                        report.append(f"    - {issue}")
                if 'violations' in result:
                    report.append(f"    - Violations: {result['violations']} ({result['violation_percentage']:.1f}%)")
                if 'error' in result:
                    report.append(f"    - Error: {result['error']}")
        
        return "\n".join(report)
```

## Quality Monitoring

### Real-time Quality Monitor

```python
import time
from threading import Thread
import sqlite3
import json

class QualityMonitor:
    """Real-time data quality monitoring system"""
    
    def __init__(self, db_path: str = "quality_monitor.db"):
        self.db_path = db_path
        self.monitoring = False
        self.thresholds = {}
        self.alerts = []
        self._init_database()
    
    def _init_database(self):
        """Initialize monitoring database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                dataset_name TEXT,
                metric_name TEXT,
                metric_value REAL,
                threshold_value REAL,
                status TEXT,
                details TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                alert_type TEXT,
                dataset_name TEXT,
                metric_name TEXT,
                severity TEXT,
                message TEXT,
                resolved BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def set_threshold(self, metric_name: str, threshold_value: float, 
                     comparison: str = 'min'):
        """Set quality threshold for monitoring"""
        self.thresholds[metric_name] = {
            'value': threshold_value,
            'comparison': comparison  # 'min', 'max', 'equal'
        }
    
    def start_monitoring(self, data_source_func: callable, 
                        dataset_name: str, interval: int = 60):
        """Start continuous quality monitoring"""
        self.monitoring = True
        
        def monitor_loop():
            while self.monitoring:
                try:
                    # Get fresh data
                    data = data_source_func()
                    
                    # Calculate quality metrics
                    metrics = self._calculate_metrics(data)
                    
                    # Check thresholds and log metrics
                    self._check_thresholds(metrics, dataset_name)
                    
                    # Sleep until next check
                    time.sleep(interval)
                    
                except Exception as e:
                    self._create_alert('ERROR', dataset_name, 'monitoring',
                                     'HIGH', f"Monitoring error: {str(e)}")
                    time.sleep(interval)
        
        monitor_thread = Thread(target=monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop quality monitoring"""
        self.monitoring = False
    
    def _calculate_metrics(self, data: pd.DataFrame):
        """Calculate standard quality metrics"""
        metrics = {}
        
        # Completeness
        total_cells = data.shape[0] * data.shape[1]
        non_null_cells = total_cells - data.isnull().sum().sum()
        metrics['completeness'] = non_null_cells / total_cells
        
        # Uniqueness (for each column)
        for col in data.columns:
            if data[col].dtype in ['object', 'string']:
                unique_ratio = data[col].nunique() / len(data[col].dropna())
                metrics[f'uniqueness_{col}'] = unique_ratio
        
        # Freshness (if timestamp column exists)
        timestamp_cols = data.select_dtypes(include=['datetime64']).columns
        if len(timestamp_cols) > 0:
            latest_timestamp = data[timestamp_cols[0]].max()
            hours_since_latest = (datetime.now() - latest_timestamp).total_seconds() / 3600
            metrics['freshness_hours'] = hours_since_latest
        
        # Volume
        metrics['row_count'] = len(data)
        
        return metrics
    
    def _check_thresholds(self, metrics: dict, dataset_name: str):
        """Check metrics against thresholds"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for metric_name, metric_value in metrics.items():
            status = 'OK'
            threshold_value = None
            
            if metric_name in self.thresholds:
                threshold_config = self.thresholds[metric_name]
                threshold_value = threshold_config['value']
                comparison = threshold_config['comparison']
                
                if comparison == 'min' and metric_value < threshold_value:
                    status = 'BELOW_THRESHOLD'
                    self._create_alert('THRESHOLD', dataset_name, metric_name,
                                     'MEDIUM', f"{metric_name} is below threshold: {metric_value} < {threshold_value}")
                elif comparison == 'max' and metric_value > threshold_value:
                    status = 'ABOVE_THRESHOLD'
                    self._create_alert('THRESHOLD', dataset_name, metric_name,
                                     'MEDIUM', f"{metric_name} is above threshold: {metric_value} > {threshold_value}")
            
            # Log metric
            cursor.execute('''
                INSERT INTO quality_metrics 
                (timestamp, dataset_name, metric_name, metric_value, threshold_value, status, details)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (datetime.now(), dataset_name, metric_name, metric_value, 
                  threshold_value, status, json.dumps({})))
        
        conn.commit()
        conn.close()
    
    def _create_alert(self, alert_type: str, dataset_name: str, 
                     metric_name: str, severity: str, message: str):
        """Create quality alert"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO quality_alerts 
            (timestamp, alert_type, dataset_name, metric_name, severity, message)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (datetime.now(), alert_type, dataset_name, metric_name, severity, message))
        
        conn.commit()
        conn.close()
        
        # Add to in-memory alerts
        self.alerts.append({
            'timestamp': datetime.now(),
            'type': alert_type,
            'dataset': dataset_name,
            'metric': metric_name,
            'severity': severity,
            'message': message
        })
    
    def get_quality_dashboard(self, dataset_name: str = None, 
                            hours: int = 24):
        """Get quality metrics for dashboard"""
        conn = sqlite3.connect(self.db_path)
        
        # Query recent metrics
        query = '''
            SELECT * FROM quality_metrics 
            WHERE timestamp > datetime('now', '-{} hours')
        '''.format(hours)
        
        if dataset_name:
            query += f" AND dataset_name = '{dataset_name}'"
        
        query += " ORDER BY timestamp DESC"
        
        metrics_df = pd.read_sql_query(query, conn)
        
        # Query recent alerts
        alert_query = '''
            SELECT * FROM quality_alerts 
            WHERE timestamp > datetime('now', '-{} hours')
            AND resolved = FALSE
        '''.format(hours)
        
        if dataset_name:
            alert_query += f" AND dataset_name = '{dataset_name}'"
        
        alerts_df = pd.read_sql_query(alert_query, conn)
        conn.close()
        
        return {
            'metrics': metrics_df,
            'alerts': alerts_df,
            'summary': {
                'total_metrics': len(metrics_df),
                'active_alerts': len(alerts_df),
                'datasets_monitored': metrics_df['dataset_name'].nunique() if len(metrics_df) > 0 else 0
            }
        }
```

## Implementation Examples

### Usage Examples

```python
# Example: Complete quality assessment workflow
def run_quality_assessment_example():
    # Create sample data with quality issues
    sample_data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, None, 7, 8, 9, 10],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Henry', 'Ivy', None],
        'age': [25, 30, -5, 40, 150, 35, 28, 32, 29, 31],  # Invalid ages
        'email': ['alice@example.com', 'bob@invalid', 'charlie@example.com', 
                 'david@example.com', 'eve@example.com', 'frank@example.com',
                 'grace@example.com', 'henry@example.com', 'ivy@example.com', 'jack@example.com'],
        'salary': [50000, 60000, 70000, 55000, 65000, 58000, 62000, 67000, 59000, 61000],
        'department': ['IT', 'HR', 'IT', 'Finance', 'IT', 'HR', 'Finance', 'IT', 'HR', 'Finance']
    })
    
    # Initialize quality framework
    framework = DataQualityFramework()
    
    # Add quality rules
    framework.add_quality_rule(
        'completeness_check',
        QualityDimension.COMPLETENESS,
        lambda data: CoreQualityRules.completeness_check(data, threshold=0.9)
    )
    
    framework.add_quality_rule(
        'age_range_check',
        QualityDimension.VALIDITY,
        lambda data: CoreQualityRules.range_check(data, 'age', min_val=0, max_val=120)
    )
    
    framework.add_quality_rule(
        'email_pattern_check',
        QualityDimension.VALIDITY,
        lambda data: CoreQualityRules.pattern_check(data, 'email', 
                    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    )
    
    framework.add_quality_rule(
        'department_validity',
        QualityDimension.VALIDITY,
        lambda data: CoreQualityRules.validity_check(data, 'department', 
                    ['IT', 'HR', 'Finance', 'Marketing'])
    )
    
    # Run quality assessment
    results = framework.assess_quality(sample_data)
    
    # Generate report
    report = framework.generate_quality_report()
    print(report)
    
    # Run data profiling
    profiler = DataProfiler()
    profile = profiler.profile_dataset(sample_data)
    profile_report = profiler.generate_profile_report()
    print("\n" + profile_report)

# Example: Setting up quality monitoring
def setup_quality_monitoring_example():
    # Mock data source function
    def get_latest_data():
        return pd.DataFrame({
            'id': range(1, 101),
            'value': np.random.normal(100, 15, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='H')
        })
    
    # Initialize monitor
    monitor = QualityMonitor()
    
    # Set thresholds
    monitor.set_threshold('completeness', 0.95, 'min')
    monitor.set_threshold('row_count', 50, 'min')
    monitor.set_threshold('freshness_hours', 2, 'max')
    
    # Start monitoring (in practice, this would run continuously)
    print("Starting quality monitoring...")
    monitor.start_monitoring(get_latest_data, 'sample_dataset', interval=10)
    
    # Let it run for a short time
    time.sleep(30)
    
    # Get dashboard data
    dashboard = monitor.get_quality_dashboard('sample_dataset')
    print(f"Monitoring Summary: {dashboard['summary']}")
    
    # Stop monitoring
    monitor.stop_monitoring()

if __name__ == "__main__":
    print("Running Data Quality Assessment Example...")
    run_quality_assessment_example()
    
    print("\n" + "="*60)
    print("Running Quality Monitoring Example...")
    setup_quality_monitoring_example()
```

---

*This comprehensive data quality framework provides production-ready tools for ensuring data reliability across your analytics pipeline.* 