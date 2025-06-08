# 🐍 Python for Data Science

> Comprehensive collection of Python scripts, templates, and resources for data science, machine learning, and analytics.

## 📁 Directory Structure

```
python/
├── data-analysis/           # Data analysis and EDA scripts
├── machine-learning/        # ML algorithms and workflows
├── visualization/           # Plotting and dashboard creation
├── web-scraping/           # Data collection and scraping
├── automation/             # Task automation and scripting
├── apis/                   # API development and integration
├── databases/              # Database connectivity and operations
├── testing/                # Unit tests and test frameworks
├── deployment/             # Packaging and deployment scripts
├── snippets/               # Reusable code snippets
└── templates/              # Project templates and boilerplate
```

## 🔧 Core Libraries & Frameworks

### Data Manipulation & Analysis
```python
import pandas as pd              # Data manipulation and analysis
import numpy as np               # Numerical computing
import polars as pl             # Fast DataFrame library
import dask.dataframe as dd     # Parallel computing
import modin.pandas as mpd      # Distributed pandas
```

### Machine Learning
```python
import scikit-learn as sklearn   # General-purpose ML library
import xgboost as xgb           # Gradient boosting
import lightgbm as lgb          # Light gradient boosting
import catboost as cb           # Categorical boosting
import optuna                   # Hyperparameter optimization
```

### Deep Learning
```python
import torch                    # PyTorch deep learning
import tensorflow as tf        # TensorFlow/Keras
import jax                     # JAX for high-performance ML
import transformers            # Hugging Face transformers
import lightning as pl         # PyTorch Lightning
```

### Visualization
```python
import matplotlib.pyplot as plt # Static plotting
import seaborn as sns          # Statistical visualization
import plotly.express as px    # Interactive plotting
import altair as alt           # Grammar of graphics
import bokeh                   # Interactive web plots
```

### Web Development & APIs
```python
import fastapi                 # Modern API framework
import flask                   # Lightweight web framework
import streamlit as st         # Data app framework
import dash                    # Interactive web apps
import requests                # HTTP library
```

## 📊 Data Analysis Templates

### Exploratory Data Analysis
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def explore_dataset(df):
    """Comprehensive EDA function"""
    print("Dataset Shape:", df.shape)
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nSummary Statistics:")
    print(df.describe())
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()
    
    return df.info()

# Usage
df = pd.read_csv('data.csv')
explore_dataset(df)
```

### Data Cleaning Pipeline
```python
def clean_data(df):
    """Standard data cleaning pipeline"""
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Fill numeric missing values with median
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Fill categorical missing values with mode
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Remove outliers using IQR method
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df
```

## 🤖 Machine Learning Templates

### Classification Pipeline
```python
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

class MLClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(random_state=42)
        self.label_encoder = LabelEncoder()
    
    def preprocess(self, X, y=None, fit=True):
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            if y is not None:
                y_encoded = self.label_encoder.fit_transform(y)
                return X_scaled, y_encoded
        else:
            X_scaled = self.scaler.transform(X)
            return X_scaled
    
    def train(self, X, y):
        X_processed, y_processed = self.preprocess(X, y)
        self.model.fit(X_processed, y_processed)
    
    def predict(self, X):
        X_processed = self.preprocess(X, fit=False)
        predictions = self.model.predict(X_processed)
        return self.label_encoder.inverse_transform(predictions)
    
    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        print(classification_report(y_test, predictions))
        return confusion_matrix(y_test, predictions)

# Usage
classifier = MLClassifier()
classifier.train(X_train, y_train)
classifier.evaluate(X_test, y_test)
```

### Regression Pipeline
```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class MLRegressor:
    def __init__(self, model_type='rf'):
        self.scaler = StandardScaler()
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'rf':
            self.model = RandomForestRegressor(random_state=42)
    
    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        
        return {'mse': mse, 'rmse': rmse, 'r2': r2}
```

## 📈 Visualization Templates

### Statistical Plots
```python
import matplotlib.pyplot as plt
import seaborn as sns

def create_statistical_plots(df, target_col):
    """Create comprehensive statistical visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Distribution plot
    sns.histplot(df[target_col], kde=True, ax=axes[0,0])
    axes[0,0].set_title(f'Distribution of {target_col}')
    
    # Box plot
    sns.boxplot(y=df[target_col], ax=axes[0,1])
    axes[0,1].set_title(f'Box Plot of {target_col}')
    
    # Correlation heatmap
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[1,0])
    axes[1,0].set_title('Correlation Matrix')
    
    # Pairplot (sample of columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]
    sns.scatterplot(data=df, x=numeric_cols[0], y=target_col, ax=axes[1,1])
    axes[1,1].set_title(f'{numeric_cols[0]} vs {target_col}')
    
    plt.tight_layout()
    plt.show()
```

### Interactive Dashboards
```python
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_interactive_dashboard(df):
    """Create interactive dashboard with Plotly"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Distribution', 'Scatter Plot', 'Box Plot', 'Time Series'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Add histogram
    fig.add_trace(
        go.Histogram(x=df['column1'], name='Distribution'),
        row=1, col=1
    )
    
    # Add scatter plot
    fig.add_trace(
        go.Scatter(x=df['column1'], y=df['column2'], mode='markers', name='Scatter'),
        row=1, col=2
    )
    
    # Add box plot
    fig.add_trace(
        go.Box(y=df['column1'], name='Box Plot'),
        row=2, col=1
    )
    
    # Add time series (if applicable)
    if 'date' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['value'], mode='lines', name='Time Series'),
            row=2, col=2
        )
    
    fig.update_layout(height=600, showlegend=False, title_text="Interactive Dashboard")
    fig.show()
```

## 🌐 Web Development Templates

### FastAPI Application
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="ML Model API", version="1.0.0")

# Load pre-trained model
model = joblib.load('model.pkl')

class PredictionRequest(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Make prediction
        prediction = model.predict([request.features])[0]
        confidence = max(model.predict_proba([request.features])[0])
        
        return PredictionResponse(
            prediction=float(prediction),
            confidence=float(confidence)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Streamlit Dashboard
```python
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Data Analysis Dashboard", layout="wide")

st.title("📊 Data Analysis Dashboard")

# Sidebar for file upload
st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Main dashboard
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Data preview
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    # Visualizations
    st.subheader("Visualizations")
    
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    if len(numeric_columns) >= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            x_axis = st.selectbox("Select X-axis", numeric_columns)
            y_axis = st.selectbox("Select Y-axis", numeric_columns)
            
            fig = px.scatter(df, x=x_axis, y=y_axis)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            selected_column = st.selectbox("Select column for histogram", numeric_columns)
            fig = px.histogram(df, x=selected_column)
            st.plotly_chart(fig, use_container_width=True)
```

## 🔧 Utility Functions

### File Operations
```python
import os
import json
import pickle
from pathlib import Path

def save_data(data, filepath, format='pickle'):
    """Save data in various formats"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    elif format == 'json':
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    elif format == 'csv' and hasattr(data, 'to_csv'):
        data.to_csv(filepath, index=False)

def load_data(filepath, format='pickle'):
    """Load data from various formats"""
    if format == 'pickle':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    elif format == 'json':
        with open(filepath, 'r') as f:
            return json.load(f)
    elif format == 'csv':
        return pd.read_csv(filepath)
```

### Configuration Management
```python
import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """Configuration class for ML projects"""
    data_path: str
    model_path: str
    output_path: str
    random_seed: int = 42
    test_size: float = 0.2
    n_estimators: int = 100
    learning_rate: Optional[float] = None

def load_config(config_path: str) -> Config:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)

# Example config.yaml
"""
data_path: "data/raw/dataset.csv"
model_path: "models/trained_model.pkl"
output_path: "results/"
random_seed: 42
test_size: 0.2
n_estimators: 100
learning_rate: 0.01
"""
```

## 🧪 Testing Templates

### Unit Testing
```python
import unittest
import pandas as pd
import numpy as np
from your_module import MLClassifier, clean_data

class TestMLClassifier(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.classifier = MLClassifier()
        self.X_train = np.random.rand(100, 5)
        self.y_train = np.random.randint(0, 2, 100)
        self.X_test = np.random.rand(20, 5)
        self.y_test = np.random.randint(0, 2, 20)
    
    def test_training(self):
        """Test model training"""
        self.classifier.train(self.X_train, self.y_train)
        self.assertIsNotNone(self.classifier.model)
    
    def test_prediction(self):
        """Test model prediction"""
        self.classifier.train(self.X_train, self.y_train)
        predictions = self.classifier.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
    
    def test_data_cleaning(self):
        """Test data cleaning function"""
        # Create test data with missing values
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': ['a', 'b', 'c', np.nan, 'e'],
            'C': [1.1, 2.2, 3.3, 4.4, 100.0]  # Contains outlier
        })
        
        cleaned_df = clean_data(df)
        self.assertFalse(cleaned_df.isnull().any().any())

if __name__ == '__main__':
    unittest.main()
```

## 📦 Project Templates

### Standard Project Structure
```
project_name/
├── data/
│   ├── raw/                 # Original, immutable data
│   ├── interim/             # Intermediate data
│   └── processed/           # Final, canonical data sets
├── notebooks/               # Jupyter notebooks
├── src/
│   ├── __init__.py
│   ├── data/               # Data loading and processing
│   ├── features/           # Feature engineering
│   ├── models/             # Model training and prediction
│   └── visualization/      # Plotting and visualization
├── tests/                  # Unit tests
├── requirements.txt        # Dependencies
├── setup.py               # Package setup
├── config.yaml            # Configuration file
└── README.md              # Project documentation
```

## 🚀 Best Practices

### Code Quality
1. **PEP 8 Compliance**: Follow Python style guidelines
2. **Type Hints**: Use type annotations for better code clarity
3. **Docstrings**: Document functions and classes
4. **Error Handling**: Implement proper exception handling
5. **Logging**: Use logging instead of print statements

### Performance Optimization
1. **Vectorization**: Use NumPy/Pandas vectorized operations
2. **Memory Management**: Monitor memory usage with large datasets
3. **Parallel Processing**: Use multiprocessing for CPU-bound tasks
4. **Caching**: Cache expensive computations
5. **Profiling**: Use cProfile and line_profiler for optimization

### Development Workflow
1. **Virtual Environments**: Use conda or venv for isolation
2. **Version Control**: Git best practices and branching
3. **Testing**: Write unit tests and integration tests
4. **CI/CD**: Automated testing and deployment
5. **Documentation**: Maintain comprehensive documentation

## 🔗 Useful Resources

### Documentation
- [Python Official Documentation](https://docs.python.org/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)

### Learning Resources
- [Python for Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
- [Automate the Boring Stuff with Python](https://automatetheboringstuff.com/)
- [Real Python Tutorials](https://realpython.com/)
- [Kaggle Learn](https://www.kaggle.com/learn)

### Tools & Libraries
- [Awesome Python](https://github.com/vinta/awesome-python)
- [Python Package Index (PyPI)](https://pypi.org/)
- [Conda Forge](https://conda-forge.org/)
- [GitHub Python Topics](https://github.com/topics/python) 