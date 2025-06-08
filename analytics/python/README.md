# Python Analytics & Data Science

## Overview

Comprehensive guide to Python for data analytics, statistical analysis, machine learning, and data visualization. Covers essential libraries, workflows, and best practices for data science projects.

## Table of Contents

- [Core Analytics Libraries](#core-analytics-libraries)
- [Data Loading and Exploration](#data-loading-and-exploration)
- [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
- [Statistical Analysis](#statistical-analysis)
- [Data Visualization](#data-visualization)
- [Machine Learning Workflows](#machine-learning-workflows)
- [Time Series Analysis](#time-series-analysis)
- [Production Analytics](#production-analytics)

## Core Analytics Libraries

### Essential Library Setup

```python
# Data manipulation and analysis
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Statistical analysis
import scipy.stats as stats
from scipy import optimize
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

# Machine learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, r2_score

# Advanced analytics
import warnings
warnings.filterwarnings('ignore')

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
plt.style.use('seaborn-v0_8')

print("Analytics environment ready!")
```

### Environment Configuration

```python
class AnalyticsConfig:
    """Configuration class for analytics projects"""
    
    def __init__(self):
        self.random_state = 42
        self.figure_size = (12, 8)
        self.color_palette = sns.color_palette("husl", 8)
        self.test_size = 0.2
        
        # Set random seeds for reproducibility
        np.random.seed(self.random_state)
        
        # Configure matplotlib
        plt.rcParams['figure.figsize'] = self.figure_size
        plt.rcParams['font.size'] = 12
        
        # Configure pandas
        pd.options.display.float_format = '{:.4f}'.format
    
    def setup_notebook(self):
        """Setup for Jupyter notebook environment"""
        from IPython.display import display, HTML
        
        # Enable inline plotting
        %matplotlib inline
        
        # Custom CSS for better display
        display(HTML("""
        <style>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
        .dataframe tbody tr th {
            vertical-align: top;
        }
        .dataframe thead th {
            text-align: right;
        }
        </style>
        """))

# Initialize configuration
config = AnalyticsConfig()
```

## Data Loading and Exploration

### Data Loading Utilities

```python
class DataLoader:
    """Comprehensive data loading utilities"""
    
    @staticmethod
    def load_csv(filepath, **kwargs):
        """Load CSV with error handling and basic info"""
        try:
            df = pd.read_csv(filepath, **kwargs)
            print(f"✓ Loaded {filepath}")
            print(f"  Shape: {df.shape}")
            print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            return df
        except Exception as e:
            print(f"✗ Error loading {filepath}: {e}")
            return None
    
    @staticmethod
    def load_excel(filepath, sheet_name=None, **kwargs):
        """Load Excel files with multiple sheet support"""
        try:
            if sheet_name is None:
                # Load all sheets
                excel_file = pd.ExcelFile(filepath)
                sheets = {}
                for sheet in excel_file.sheet_names:
                    sheets[sheet] = pd.read_excel(filepath, sheet_name=sheet, **kwargs)
                    print(f"✓ Loaded sheet '{sheet}': {sheets[sheet].shape}")
                return sheets
            else:
                df = pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
                print(f"✓ Loaded {filepath} (sheet: {sheet_name})")
                print(f"  Shape: {df.shape}")
                return df
        except Exception as e:
            print(f"✗ Error loading {filepath}: {e}")
            return None
    
    @staticmethod
    def create_sample_dataset(n_rows=1000):
        """Create sample dataset for testing"""
        np.random.seed(42)
        
        # Generate sample e-commerce data
        data = {
            'customer_id': np.random.randint(1, 500, n_rows),
            'order_date': pd.date_range('2023-01-01', periods=n_rows, freq='H'),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], n_rows),
            'product_price': np.random.exponential(50, n_rows),
            'quantity': np.random.poisson(2, n_rows) + 1,
            'customer_age': np.random.normal(40, 15, n_rows).astype(int),
            'customer_gender': np.random.choice(['M', 'F'], n_rows),
            'shipping_cost': np.random.uniform(5, 25, n_rows),
            'discount_applied': np.random.choice([True, False], n_rows, p=[0.3, 0.7])
        }
        
        df = pd.DataFrame(data)
        
        # Add derived columns
        df['total_amount'] = df['product_price'] * df['quantity']
        df['final_amount'] = np.where(df['discount_applied'], 
                                    df['total_amount'] * 0.9, 
                                    df['total_amount'])
        df['order_month'] = df['order_date'].dt.month
        df['order_day_of_week'] = df['order_date'].dt.day_name()
        
        # Add some missing values for realistic data
        missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
        df.loc[missing_indices, 'customer_age'] = np.nan
        
        return df

class DataExplorer:
    """Comprehensive data exploration utilities"""
    
    def __init__(self, df):
        self.df = df
    
    def overview(self):
        """Comprehensive dataset overview"""
        print("=== DATASET OVERVIEW ===")
        print(f"Shape: {self.df.shape}")
        print(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\n=== DATA TYPES ===")
        print(self.df.dtypes.value_counts())
        
        print("\n=== MISSING VALUES ===")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Missing %': missing_pct
        })
        print(missing_df[missing_df['Missing Count'] > 0])
        
        print("\n=== NUMERIC COLUMNS SUMMARY ===")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(self.df[numeric_cols].describe())
        
        print("\n=== CATEGORICAL COLUMNS SUMMARY ===")
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            print(f"\n{col}:")
            print(f"  Unique values: {self.df[col].nunique()}")
            print(f"  Top values: {self.df[col].value_counts().head(3).to_dict()}")
    
    def correlation_analysis(self, method='pearson'):
        """Analyze correlations between numeric variables"""
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            print("Not enough numeric columns for correlation analysis")
            return None
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr(method=method)
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5)
        plt.title(f'{method.capitalize()} Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        # Find strong correlations
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_corr.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        if strong_corr:
            print("\nStrong correlations (|r| > 0.7):")
            for corr in strong_corr:
                print(f"  {corr['var1']} <-> {corr['var2']}: {corr['correlation']:.3f}")
        
        return corr_matrix
    
    def outlier_detection(self, method='iqr'):
        """Detect outliers in numeric columns"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outliers_summary = {}
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                outliers = self.df[z_scores > 3]
            
            outliers_summary[col] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(self.df) * 100,
                'indices': outliers.index.tolist()
            }
        
        # Display summary
        print(f"=== OUTLIER DETECTION ({method.upper()}) ===")
        for col, info in outliers_summary.items():
            print(f"{col}: {info['count']} outliers ({info['percentage']:.2f}%)")
        
        return outliers_summary

# Example usage
sample_data = DataLoader.create_sample_dataset(1000)
explorer = DataExplorer(sample_data)
explorer.overview()
```

## Data Cleaning and Preprocessing

### Advanced Data Cleaning

```python
class DataCleaner:
    """Comprehensive data cleaning utilities"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        self.cleaning_log = []
    
    def handle_missing_values(self, strategy='auto'):
        """Handle missing values with multiple strategies"""
        
        missing_before = self.df.isnull().sum().sum()
        
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            
            if missing_count == 0:
                continue
                
            missing_pct = missing_count / len(self.df) * 100
            
            if strategy == 'auto':
                # Auto strategy based on data type and missing percentage
                if missing_pct > 70:
                    # Drop columns with >70% missing
                    self.df.drop(columns=[col], inplace=True)
                    self.cleaning_log.append(f"Dropped column '{col}' ({missing_pct:.1f}% missing)")
                    
                elif self.df[col].dtype in ['object', 'category']:
                    # Fill categorical with mode or 'Unknown'
                    if self.df[col].value_counts().empty:
                        fill_value = 'Unknown'
                    else:
                        fill_value = self.df[col].mode().iloc[0]
                    self.df[col].fillna(fill_value, inplace=True)
                    self.cleaning_log.append(f"Filled '{col}' missing values with '{fill_value}'")
                    
                else:
                    # Fill numeric with median
                    fill_value = self.df[col].median()
                    self.df[col].fillna(fill_value, inplace=True)
                    self.cleaning_log.append(f"Filled '{col}' missing values with median ({fill_value:.2f})")
            
            elif strategy == 'drop':
                # Drop rows with any missing values
                self.df.dropna(inplace=True)
                self.cleaning_log.append("Dropped all rows with missing values")
                break
        
        missing_after = self.df.isnull().sum().sum()
        print(f"Missing values: {missing_before} → {missing_after}")
        
        return self
    
    def handle_duplicates(self, subset=None, keep='first'):
        """Remove duplicate rows"""
        duplicates_before = self.df.duplicated(subset=subset).sum()
        
        if duplicates_before > 0:
            self.df.drop_duplicates(subset=subset, keep=keep, inplace=True)
            duplicates_after = self.df.duplicated(subset=subset).sum()
            self.cleaning_log.append(f"Removed {duplicates_before} duplicate rows")
            print(f"Duplicates: {duplicates_before} → {duplicates_after}")
        else:
            print("No duplicates found")
        
        return self
    
    def handle_outliers(self, columns=None, method='iqr', action='cap'):
        """Handle outliers in numeric columns"""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        outliers_handled = 0
        
        for col in columns:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                outlier_mask = z_scores > 3
            
            outliers_count = outlier_mask.sum()
            
            if outliers_count > 0:
                if action == 'cap':
                    # Cap outliers to bounds
                    if method == 'iqr':
                        self.df.loc[self.df[col] < lower_bound, col] = lower_bound
                        self.df.loc[self.df[col] > upper_bound, col] = upper_bound
                elif action == 'remove':
                    # Remove outlier rows
                    self.df = self.df[~outlier_mask]
                
                outliers_handled += outliers_count
                self.cleaning_log.append(f"Handled {outliers_count} outliers in '{col}' using {method}/{action}")
        
        print(f"Total outliers handled: {outliers_handled}")
        return self
    
    def standardize_text(self, columns=None):
        """Standardize text columns"""
        if columns is None:
            columns = self.df.select_dtypes(include=['object']).columns
        
        for col in columns:
            if self.df[col].dtype == 'object':
                # Basic text cleaning
                self.df[col] = (self.df[col]
                               .astype(str)
                               .str.strip()
                               .str.title()
                               .replace('Nan', np.nan))
                
                self.cleaning_log.append(f"Standardized text in '{col}'")
        
        return self
    
    def encode_categorical(self, columns=None, method='auto'):
        """Encode categorical variables"""
        if columns is None:
            columns = self.df.select_dtypes(include=['object', 'category']).columns
        
        for col in columns:
            unique_count = self.df[col].nunique()
            
            if method == 'auto':
                # Auto select encoding method
                if unique_count <= 10:
                    # One-hot encoding for low cardinality
                    dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                    self.df = pd.concat([self.df, dummies], axis=1)
                    self.df.drop(columns=[col], inplace=True)
                    self.cleaning_log.append(f"One-hot encoded '{col}' ({unique_count} categories)")
                else:
                    # Label encoding for high cardinality
                    le = LabelEncoder()
                    self.df[col] = le.fit_transform(self.df[col].astype(str))
                    self.cleaning_log.append(f"Label encoded '{col}' ({unique_count} categories)")
            
            elif method == 'onehot':
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                self.df = pd.concat([self.df, dummies], axis=1)
                self.df.drop(columns=[col], inplace=True)
        
        return self
    
    def feature_scaling(self, columns=None, method='standard'):
        """Scale numeric features"""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        
        self.df[columns] = scaler.fit_transform(self.df[columns])
        self.cleaning_log.append(f"Applied {method} scaling to {len(columns)} columns")
        
        return self
    
    def get_summary(self):
        """Get cleaning summary"""
        print("=== DATA CLEANING SUMMARY ===")
        print(f"Original shape: {self.original_shape}")
        print(f"Final shape: {self.df.shape}")
        print(f"Rows removed: {self.original_shape[0] - self.df.shape[0]}")
        print(f"Columns removed: {self.original_shape[1] - self.df.shape[1]}")
        
        print("\n=== CLEANING STEPS ===")
        for step in self.cleaning_log:
            print(f"  • {step}")
        
        return self.df

# Example usage with chaining
cleaned_data = (DataCleaner(sample_data)
                .handle_missing_values()
                .handle_duplicates()
                .handle_outliers(action='cap')
                .standardize_text()
                .get_summary())
```

## Statistical Analysis

### Descriptive and Inferential Statistics

```python
class StatisticalAnalyzer:
    """Comprehensive statistical analysis toolkit"""
    
    def __init__(self, df):
        self.df = df
    
    def descriptive_stats(self, columns=None):
        """Comprehensive descriptive statistics"""
        if columns is None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        else:
            numeric_cols = columns
        
        stats_dict = {}
        
        for col in numeric_cols:
            data = self.df[col].dropna()
            
            stats_dict[col] = {
                'count': len(data),
                'mean': data.mean(),
                'median': data.median(),
                'mode': data.mode().iloc[0] if not data.mode().empty else np.nan,
                'std': data.std(),
                'variance': data.var(),
                'min': data.min(),
                'max': data.max(),
                'range': data.max() - data.min(),
                'q25': data.quantile(0.25),
                'q75': data.quantile(0.75),
                'iqr': data.quantile(0.75) - data.quantile(0.25),
                'skewness': stats.skew(data),
                'kurtosis': stats.kurtosis(data),
                'cv': data.std() / data.mean() if data.mean() != 0 else np.nan
            }
        
        # Convert to DataFrame for better display
        stats_df = pd.DataFrame(stats_dict).T
        return stats_df
    
    def distribution_analysis(self, column, bins=30):
        """Analyze distribution of a variable"""
        data = self.df[column].dropna()
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Histogram
        axes[0, 0].hist(data, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title(f'Histogram of {column}')
        axes[0, 0].set_xlabel(column)
        axes[0, 0].set_ylabel('Frequency')
        
        # Box plot
        axes[0, 1].boxplot(data, vert=True)
        axes[0, 1].set_title(f'Box Plot of {column}')
        axes[0, 1].set_ylabel(column)
        
        # Q-Q plot
        stats.probplot(data, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title(f'Q-Q Plot of {column}')
        
        # Density plot
        data.plot.density(ax=axes[1, 1], color='red')
        axes[1, 1].set_title(f'Density Plot of {column}')
        axes[1, 1].set_xlabel(column)
        
        plt.tight_layout()
        plt.show()
        
        # Statistical tests
        print(f"=== DISTRIBUTION ANALYSIS: {column} ===")
        
        # Normality test
        shapiro_stat, shapiro_p = stats.shapiro(data.sample(min(5000, len(data))))
        print(f"Shapiro-Wilk normality test: statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
        
        # Anderson-Darling test
        anderson_result = stats.anderson(data)
        print(f"Anderson-Darling test: statistic={anderson_result.statistic:.4f}")
        
        return {
            'shapiro_statistic': shapiro_stat,
            'shapiro_pvalue': shapiro_p,
            'anderson_statistic': anderson_result.statistic
        }
    
    def hypothesis_testing(self, group_col, value_col, test_type='ttest'):
        """Perform hypothesis tests between groups"""
        groups = self.df[group_col].unique()
        
        if len(groups) != 2:
            print(f"Error: {group_col} must have exactly 2 groups for t-test")
            return None
        
        group1_data = self.df[self.df[group_col] == groups[0]][value_col].dropna()
        group2_data = self.df[self.df[group_col] == groups[1]][value_col].dropna()
        
        print(f"=== HYPOTHESIS TESTING: {value_col} by {group_col} ===")
        print(f"Group 1 ({groups[0]}): n={len(group1_data)}, mean={group1_data.mean():.4f}")
        print(f"Group 2 ({groups[1]}): n={len(group2_data)}, mean={group2_data.mean():.4f}")
        
        if test_type == 'ttest':
            # Independent t-test
            stat, p_value = stats.ttest_ind(group1_data, group2_data)
            test_name = "Independent t-test"
            
        elif test_type == 'mannwhitney':
            # Mann-Whitney U test (non-parametric)
            stat, p_value = stats.mannwhitneyu(group1_data, group2_data)
            test_name = "Mann-Whitney U test"
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(group1_data) - 1) * group1_data.var() + 
                             (len(group2_data) - 1) * group2_data.var()) / 
                            (len(group1_data) + len(group2_data) - 2))
        cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_std
        
        print(f"\n{test_name} results:")
        print(f"  Statistic: {stat:.4f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Cohen's d: {cohens_d:.4f}")
        
        # Interpretation
        alpha = 0.05
        if p_value < alpha:
            print(f"  Result: Significant difference (p < {alpha})")
        else:
            print(f"  Result: No significant difference (p >= {alpha})")
        
        return {
            'statistic': stat,
            'pvalue': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < alpha
        }
    
    def correlation_significance(self, col1, col2, method='pearson'):
        """Test correlation significance"""
        data1 = self.df[col1].dropna()
        data2 = self.df[col2].dropna()
        
        # Align data (remove rows where either is missing)
        combined = pd.DataFrame({col1: self.df[col1], col2: self.df[col2]}).dropna()
        data1 = combined[col1]
        data2 = combined[col2]
        
        if method == 'pearson':
            corr, p_value = stats.pearsonr(data1, data2)
        elif method == 'spearman':
            corr, p_value = stats.spearmanr(data1, data2)
        
        print(f"=== CORRELATION ANALYSIS ===")
        print(f"{method.capitalize()} correlation between {col1} and {col2}:")
        print(f"  Correlation coefficient: {corr:.4f}")
        print(f"  P-value: {p_value:.4f}")
        
        # Confidence interval for Pearson correlation
        if method == 'pearson':
            n = len(data1)
            z_score = 0.5 * np.log((1 + corr) / (1 - corr))
            se = 1 / np.sqrt(n - 3)
            z_crit = stats.norm.ppf(0.975)  # 95% CI
            
            z_lower = z_score - z_crit * se
            z_upper = z_score + z_crit * se
            
            ci_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
            ci_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
            
            print(f"  95% Confidence Interval: ({ci_lower:.4f}, {ci_upper:.4f})")
        
        return {
            'correlation': corr,
            'pvalue': p_value,
            'significant': p_value < 0.05
        }

# Example usage
stats_analyzer = StatisticalAnalyzer(sample_data)
desc_stats = stats_analyzer.descriptive_stats()
print(desc_stats)
```

---

*This comprehensive Python analytics guide provides complete coverage of data analysis workflows from loading to advanced statistical analysis.* 