#!/usr/bin/env python3
"""
Data Analysis Template for Python
=================================

A comprehensive template for data analysis projects including:
- Data loading and preprocessing
- Exploratory Data Analysis (EDA)
- Statistical analysis
- Visualization
- Model building basics

Author: Data Science Team
Date: 2024
"""

# Core Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path

# Statistical Analysis
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_squared_error

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

class DataAnalyzer:
    """
    A comprehensive data analysis class with common methods
    """
    
    def __init__(self, data_path=None, df=None):
        """Initialize with either file path or dataframe"""
        if data_path:
            self.df = self.load_data(data_path)
        elif df is not None:
            self.df = df.copy()
        else:
            self.df = None
        
        self.target_column = None
        self.numeric_columns = None
        self.categorical_columns = None
    
    def load_data(self, path):
        """Load data from various file formats"""
        path = Path(path)
        
        if path.suffix.lower() == '.csv':
            return pd.read_csv(path)
        elif path.suffix.lower() in ['.xlsx', '.xls']:
            return pd.read_excel(path)
        elif path.suffix.lower() == '.json':
            return pd.read_json(path)
        elif path.suffix.lower() == '.parquet':
            return pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def basic_info(self):
        """Display basic information about the dataset"""
        print("=== DATASET OVERVIEW ===")
        print(f"Shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print("\n=== COLUMN TYPES ===")
        print(self.df.dtypes)
        print("\n=== MISSING VALUES ===")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Missing %': missing_pct
        }).sort_values('Missing %', ascending=False)
        print(missing_df[missing_df['Missing Count'] > 0])
        
        return missing_df
    
    def identify_column_types(self):
        """Automatically identify numeric and categorical columns"""
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f"Numeric columns ({len(self.numeric_columns)}): {self.numeric_columns}")
        print(f"Categorical columns ({len(self.categorical_columns)}): {self.categorical_columns}")
    
    def descriptive_statistics(self):
        """Generate comprehensive descriptive statistics"""
        print("=== NUMERIC VARIABLES SUMMARY ===")
        print(self.df.describe())
        
        if self.categorical_columns:
            print("\n=== CATEGORICAL VARIABLES SUMMARY ===")
            for col in self.categorical_columns:
                print(f"\n{col}:")
                print(self.df[col].value_counts().head())
    
    def detect_outliers(self, method='iqr'):
        """Detect outliers using IQR or Z-score method"""
        outliers = {}
        
        for col in self.numeric_columns:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers[col] = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                outliers[col] = self.df[z_scores > 3]
        
        return outliers
    
    def correlation_analysis(self, figsize=(12, 8)):
        """Generate correlation matrix and heatmap"""
        if len(self.numeric_columns) > 1:
            corr_matrix = self.df[self.numeric_columns].corr()
            
            plt.figure(figsize=figsize)
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5)
            plt.title('Correlation Matrix')
            plt.tight_layout()
            plt.show()
            
            return corr_matrix
    
    def distribution_plots(self, columns=None, figsize=(15, 10)):
        """Create distribution plots for numeric variables"""
        cols_to_plot = columns or self.numeric_columns
        n_cols = 3
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i, col in enumerate(cols_to_plot):
            sns.histplot(data=self.df, x=col, kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
        
        # Hide unused subplots
        for i in range(len(cols_to_plot), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def categorical_plots(self, max_categories=10, figsize=(15, 10)):
        """Create bar plots for categorical variables"""
        if not self.categorical_columns:
            print("No categorical columns found")
            return
        
        n_cols = 2
        n_rows = (len(self.categorical_columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i, col in enumerate(self.categorical_columns):
            value_counts = self.df[col].value_counts().head(max_categories)
            value_counts.plot(kind='bar', ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
            axes[i].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(self.categorical_columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def pairplot_analysis(self, sample_size=1000):
        """Create pairplot for numeric variables"""
        if len(self.numeric_columns) > 1:
            # Sample data if too large
            df_sample = self.df.sample(min(sample_size, len(self.df)))
            sns.pairplot(df_sample[self.numeric_columns])
            plt.show()
    
    def statistical_tests(self, target_col=None):
        """Perform basic statistical tests"""
        if target_col is None:
            print("Please specify a target column for statistical tests")
            return
        
        self.target_column = target_col
        results = {}
        
        # For numeric target vs categorical features
        if self.df[target_col].dtype in ['int64', 'float64']:
            for cat_col in self.categorical_columns:
                if cat_col != target_col:
                    groups = [group[target_col].dropna() for name, group in self.df.groupby(cat_col)]
                    if len(groups) >= 2:
                        # ANOVA test
                        f_stat, p_value = stats.f_oneway(*groups)
                        results[f'{cat_col}_vs_{target_col}'] = {
                            'test': 'ANOVA',
                            'f_statistic': f_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        }
        
        return results
    
    def export_report(self, filename='data_analysis_report.html'):
        """Export analysis report to HTML"""
        # This would require additional libraries like pandas-profiling
        # For now, we'll create a simple text report
        report = []
        report.append("DATA ANALYSIS REPORT")
        report.append("=" * 50)
        report.append(f"Dataset Shape: {self.df.shape}")
        report.append(f"Numeric Columns: {len(self.numeric_columns)}")
        report.append(f"Categorical Columns: {len(self.categorical_columns)}")
        
        missing = self.df.isnull().sum().sum()
        report.append(f"Total Missing Values: {missing}")
        
        with open(filename.replace('.html', '.txt'), 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Report saved to {filename.replace('.html', '.txt')}")

# Example Usage
if __name__ == "__main__":
    # Example with sample data
    print("Creating sample dataset for demonstration...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = {
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.lognormal(10, 1, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston'], n_samples),
        'satisfaction': np.random.randint(1, 11, n_samples)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize analyzer
    analyzer = DataAnalyzer(df=df)
    
    # Run analysis
    analyzer.identify_column_types()
    analyzer.basic_info()
    analyzer.descriptive_statistics()
    
    # Visualizations (uncomment to run)
    # analyzer.distribution_plots()
    # analyzer.categorical_plots()
    # analyzer.correlation_analysis()
    
    print("\nAnalysis complete! Uncomment visualization lines to see plots.") 