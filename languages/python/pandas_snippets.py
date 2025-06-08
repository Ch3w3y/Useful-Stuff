# Pandas Code Snippets for Data Science
# =====================================
# A collection of commonly used pandas operations for data manipulation and analysis

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. DATA LOADING AND INSPECTION
# ==========================================

def load_and_inspect_data():
    """Common data loading and inspection patterns"""
    
    # Load data from various sources
    df_csv = pd.read_csv('data.csv')
    df_excel = pd.read_excel('data.xlsx', sheet_name='Sheet1')
    df_json = pd.read_json('data.json')
    df_parquet = pd.read_parquet('data.parquet')
    
    # Quick inspection
    df = df_csv  # Example dataframe
    
    print("Shape:", df.shape)
    print("\nInfo:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())
    print("\nDescriptive statistics:")
    print(df.describe())
    print("\nNull values:")
    print(df.isnull().sum())
    print("\nData types:")
    print(df.dtypes)
    print("\nUnique values per column:")
    print(df.nunique())

# ==========================================
# 2. DATA CLEANING
# ==========================================

def clean_data(df):
    """Common data cleaning operations"""
    
    # Remove duplicates
    df_clean = df.drop_duplicates()
    
    # Remove rows with any null values
    df_clean = df_clean.dropna()
    
    # Remove rows with null values in specific columns
    df_clean = df_clean.dropna(subset=['important_column'])
    
    # Fill null values
    df_clean = df.fillna(0)  # Fill with 0
    df_clean = df.fillna(df.mean())  # Fill with mean (numeric columns)
    df_clean = df.fillna(method='ffill')  # Forward fill
    df_clean = df.fillna(method='bfill')  # Backward fill
    df_clean = df.fillna({'column1': 0, 'column2': 'Unknown'})  # Fill specific columns
    
    # Remove outliers using IQR method
    def remove_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    # Clean text columns
    df['text_column'] = df['text_column'].str.strip()  # Remove whitespace
    df['text_column'] = df['text_column'].str.lower()  # Convert to lowercase
    df['text_column'] = df['text_column'].str.replace('[^a-zA-Z0-9 ]', '', regex=True)  # Remove special chars
    
    return df_clean

# ==========================================
# 3. DATA TRANSFORMATION
# ==========================================

def transform_data(df):
    """Common data transformation operations"""
    
    # Create new columns
    df['new_column'] = df['column1'] + df['column2']
    df['ratio'] = df['numerator'] / df['denominator']
    df['is_high'] = df['value'] > df['value'].mean()
    
    # Apply functions to columns
    df['transformed'] = df['column'].apply(lambda x: x * 2)
    df['categorized'] = df['score'].apply(lambda x: 'High' if x > 80 else 'Medium' if x > 50 else 'Low')
    
    # Map values
    mapping = {'A': 1, 'B': 2, 'C': 3}
    df['mapped'] = df['category'].map(mapping)
    
    # Replace values
    df['column'] = df['column'].replace({'old_value': 'new_value'})
    df['column'] = df['column'].replace([1, 2, 3], ['one', 'two', 'three'])
    
    # Binning/Discretization
    df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 100], labels=['Child', 'Young', 'Adult', 'Senior'])
    df['score_quartile'] = pd.qcut(df['score'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    
    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=['category'], prefix=['cat'])
    
    # Label encoding for ordinal data
    df['size_encoded'] = df['size'].map({'S': 1, 'M': 2, 'L': 3, 'XL': 4})
    
    return df

# ==========================================
# 4. DATE AND TIME OPERATIONS
# ==========================================

def datetime_operations(df):
    """Common datetime operations"""
    
    # Convert to datetime
    df['date'] = pd.to_datetime(df['date_string'])
    df['date'] = pd.to_datetime(df['date_string'], format='%Y-%m-%d')
    
    # Extract components
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.day_name()
    df['quarter'] = df['date'].dt.quarter
    df['week_of_year'] = df['date'].dt.isocalendar().week
    
    # Calculate differences
    df['days_since'] = (datetime.now() - df['date']).dt.days
    df['business_days'] = np.busday_count(df['start_date'].values.astype('datetime64[D]'),
                                         df['end_date'].values.astype('datetime64[D]'))
    
    # Resample time series data
    df_resampled = df.set_index('date').resample('D').mean()  # Daily average
    df_resampled = df.set_index('date').resample('M').sum()   # Monthly sum
    
    # Create date ranges
    date_range = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    business_days = pd.bdate_range('2024-01-01', '2024-12-31')
    
    return df

# ==========================================
# 5. GROUPING AND AGGREGATION
# ==========================================

def groupby_operations(df):
    """Common groupby and aggregation operations"""
    
    # Basic groupby
    grouped = df.groupby('category').mean()
    grouped = df.groupby(['category', 'subcategory']).sum()
    
    # Multiple aggregations
    agg_result = df.groupby('category').agg({
        'value1': 'sum',
        'value2': 'mean',
        'value3': ['min', 'max', 'std']
    })
    
    # Custom aggregations
    def custom_agg(x):
        return x.max() - x.min()
    
    df.groupby('category')['value'].agg(custom_agg)
    
    # Apply custom functions
    def process_group(group):
        group['pct_of_total'] = group['value'] / group['value'].sum()
        return group
    
    df_processed = df.groupby('category').apply(process_group)
    
    # Transform (keeps original shape)
    df['group_mean'] = df.groupby('category')['value'].transform('mean')
    df['group_rank'] = df.groupby('category')['value'].rank()
    df['cumulative_sum'] = df.groupby('category')['value'].cumsum()
    
    # Rolling windows
    df['rolling_mean'] = df['value'].rolling(window=7).mean()
    df['expanding_sum'] = df['value'].expanding().sum()
    
    return df_processed

# ==========================================
# 6. MERGING AND JOINING
# ==========================================

def merge_and_join_operations():
    """Common merge and join operations"""
    
    # Sample dataframes
    df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
    df2 = pd.DataFrame({'key': ['A', 'B', 'D'], 'value2': [4, 5, 6]})
    
    # Inner join (intersection)
    inner_join = pd.merge(df1, df2, on='key', how='inner')
    
    # Left join (all from left)
    left_join = pd.merge(df1, df2, on='key', how='left')
    
    # Right join (all from right)
    right_join = pd.merge(df1, df2, on='key', how='right')
    
    # Outer join (union)
    outer_join = pd.merge(df1, df2, on='key', how='outer')
    
    # Merge on multiple columns
    merged = pd.merge(df1, df2, on=['key1', 'key2'])
    
    # Merge with different column names
    merged = pd.merge(df1, df2, left_on='key1', right_on='key2')
    
    # Concatenate dataframes
    concatenated = pd.concat([df1, df2])  # Vertical
    concatenated = pd.concat([df1, df2], axis=1)  # Horizontal
    
    # Append rows
    appended = df1.append(df2, ignore_index=True)
    
    return merged

# ==========================================
# 7. PIVOTING AND RESHAPING
# ==========================================

def pivot_and_reshape(df):
    """Common pivoting and reshaping operations"""
    
    # Pivot table
    pivot = df.pivot_table(values='value', index='row_var', columns='col_var', aggfunc='sum')
    
    # Pivot with multiple values
    pivot_multi = df.pivot_table(values=['value1', 'value2'], 
                                index='row_var', 
                                columns='col_var', 
                                aggfunc={'value1': 'sum', 'value2': 'mean'})
    
    # Melt (unpivot)
    melted = df.melt(id_vars=['id'], value_vars=['col1', 'col2'], 
                     var_name='variable', value_name='value')
    
    # Stack and unstack
    stacked = df.set_index(['col1', 'col2']).stack()
    unstacked = stacked.unstack()
    
    # Cross-tabulation
    crosstab = pd.crosstab(df['category1'], df['category2'])
    
    return pivot

# ==========================================
# 8. FILTERING AND SELECTION
# ==========================================

def filtering_operations(df):
    """Common filtering and selection operations"""
    
    # Boolean indexing
    filtered = df[df['value'] > 100]
    filtered = df[(df['value'] > 50) & (df['category'] == 'A')]
    filtered = df[df['category'].isin(['A', 'B', 'C'])]
    
    # String operations
    filtered = df[df['text'].str.contains('pattern')]
    filtered = df[df['text'].str.startswith('prefix')]
    filtered = df[df['text'].str.endswith('suffix')]
    
    # Select columns
    selected = df[['col1', 'col2']]
    selected = df.loc[:, 'col1':'col5']  # Range of columns
    selected = df.select_dtypes(include=['number'])  # Numeric columns only
    
    # Select rows
    selected = df.loc[0:10]  # First 10 rows
    selected = df.iloc[0:10, 0:5]  # First 10 rows, first 5 columns
    
    # Query method
    filtered = df.query('value > 100 and category == "A"')
    
    # Sample data
    sample = df.sample(n=100)  # Random 100 rows
    sample = df.sample(frac=0.1)  # Random 10% of data
    
    return filtered

# ==========================================
# 9. PERFORMANCE OPTIMIZATION
# ==========================================

def optimization_tips():
    """Performance optimization tips and tricks"""
    
    # Use categorical data type for repeated strings
    df['category'] = df['category'].astype('category')
    
    # Use appropriate data types
    df['int_col'] = df['int_col'].astype('int32')  # Instead of int64
    df['float_col'] = df['float_col'].astype('float32')  # Instead of float64
    
    # Vectorized operations (faster than apply)
    df['result'] = df['col1'] * df['col2']  # Instead of df.apply(lambda x: x['col1'] * x['col2'])
    
    # Use .loc for setting values
    df.loc[df['condition'], 'column'] = 'new_value'
    
    # Chunk processing for large files
    chunk_list = []
    for chunk in pd.read_csv('large_file.csv', chunksize=10000):
        processed_chunk = chunk.groupby('category').sum()
        chunk_list.append(processed_chunk)
    df_processed = pd.concat(chunk_list)
    
    # Use eval for complex expressions
    df.eval('new_col = col1 + col2 * col3', inplace=True)

# ==========================================
# 10. UTILITY FUNCTIONS
# ==========================================

def useful_utilities(df):
    """Useful utility functions"""
    
    # Memory usage
    memory_usage = df.memory_usage(deep=True).sum() / 1024**2  # MB
    print(f"Memory usage: {memory_usage:.2f} MB")
    
    # Column statistics
    def column_stats(df):
        stats = pd.DataFrame({
            'dtype': df.dtypes,
            'null_count': df.isnull().sum(),
            'null_percentage': (df.isnull().sum() / len(df)) * 100,
            'unique_count': df.nunique(),
            'unique_percentage': (df.nunique() / len(df)) * 100
        })
        return stats
    
    # Find correlations
    def find_high_correlations(df, threshold=0.8):
        corr_matrix = df.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        )
        high_corr = [(column, row, upper_triangle.loc[row, column]) 
                     for column in upper_triangle.columns 
                     for row in upper_triangle.index 
                     if upper_triangle.loc[row, column] > threshold]
        return high_corr
    
    # Data quality check
    def data_quality_report(df):
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'duplicated_rows': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'missing_data': df.isnull().sum().sum(),
            'column_stats': column_stats(df)
        }
        return report
    
    return data_quality_report(df)

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    sample_data = {
        'date': pd.date_range('2024-01-01', periods=1000, freq='D'),
        'category': np.random.choice(['A', 'B', 'C'], 1000),
        'value': np.random.normal(100, 15, 1000),
        'score': np.random.randint(0, 100, 1000),
        'text': ['sample_text_' + str(i) for i in range(1000)]
    }
    df = pd.DataFrame(sample_data)
    
    print("Sample DataFrame created with shape:", df.shape)
    print("\nRunning data quality report...")
    quality_report = useful_utilities(df)
    print(quality_report) 