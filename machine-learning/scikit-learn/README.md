# Scikit-Learn Machine Learning

## Overview

Comprehensive guide to scikit-learn for machine learning, covering algorithms, preprocessing, model selection, evaluation, and production workflows with practical examples.

## Table of Contents

- [Data Preprocessing](#data-preprocessing)
- [Supervised Learning](#supervised-learning)
- [Unsupervised Learning](#unsupervised-learning)
- [Model Selection](#model-selection)
- [Feature Engineering](#feature-engineering)
- [Pipeline Development](#pipeline-development)
- [Model Evaluation](#model-evaluation)
- [Production Deployment](#production-deployment)

## Data Preprocessing

### Data Cleaning and Transformation

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder,
    PolynomialFeatures, PowerTransformer
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Comprehensive data preprocessing pipeline"""
    
    def __init__(self):
        self.preprocessor = None
        self.feature_names = None
        
    def create_preprocessor(self, df, target_column=None):
        """Create preprocessing pipeline based on data types"""
        
        # Separate features from target
        if target_column:
            X = df.drop(columns=[target_column])
        else:
            X = df.copy()
            
        # Identify column types
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Numeric pipeline
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        # Combine pipelines
        self.preprocessor = ColumnTransformer([
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ])
        
        return self.preprocessor
    
    def fit_transform(self, X_train, y_train=None):
        """Fit preprocessor and transform training data"""
        X_transformed = self.preprocessor.fit_transform(X_train)
        
        # Get feature names
        self.feature_names = self._get_feature_names()
        
        return X_transformed
    
    def transform(self, X):
        """Transform new data"""
        return self.preprocessor.transform(X)
    
    def _get_feature_names(self):
        """Get feature names after transformation"""
        feature_names = []
        
        for name, transformer, features in self.preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(features)
            elif name == 'cat':
                if hasattr(transformer.named_steps['encoder'], 'get_feature_names_out'):
                    cat_names = transformer.named_steps['encoder'].get_feature_names_out(features)
                    feature_names.extend(cat_names)
                else:
                    feature_names.extend(features)
        
        return feature_names

# Advanced preprocessing techniques
class AdvancedPreprocessing:
    """Advanced preprocessing methods"""
    
    @staticmethod
    def handle_outliers(df, columns, method='iqr', factor=1.5):
        """Handle outliers using IQR or Z-score method"""
        df_clean = df.copy()
        
        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                # Cap outliers
                df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
                
            elif method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                threshold = factor
                
                # Remove outliers beyond threshold standard deviations
                df_clean = df_clean[np.abs((df_clean[col] - mean) / std) < threshold]
        
        return df_clean
    
    @staticmethod
    def create_time_features(df, date_column):
        """Create time-based features from datetime column"""
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Extract time components
        df[f'{date_column}_year'] = df[date_column].dt.year
        df[f'{date_column}_month'] = df[date_column].dt.month
        df[f'{date_column}_day'] = df[date_column].dt.day
        df[f'{date_column}_dayofweek'] = df[date_column].dt.dayofweek
        df[f'{date_column}_quarter'] = df[date_column].dt.quarter
        df[f'{date_column}_is_weekend'] = df[date_column].dt.dayofweek.isin([5, 6]).astype(int)
        
        return df
    
    @staticmethod
    def create_interaction_features(df, feature_pairs):
        """Create interaction features between specified pairs"""
        df_new = df.copy()
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                # Multiplicative interaction
                df_new[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                
                # Ratio features (avoid division by zero)
                df_new[f'{feat1}_div_{feat2}'] = df[feat1] / (df[feat2] + 1e-8)
                df_new[f'{feat2}_div_{feat1}'] = df[feat2] / (df[feat1] + 1e-8)
        
        return df_new
```

## Supervised Learning

### Classification Algorithms

```python
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

class ClassificationSuite:
    """Complete classification model suite"""
    
    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self.results = {}
        
    def setup_models(self):
        """Setup classification models with default parameters"""
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'knn': KNeighborsClassifier(n_neighbors=5),
            'naive_bayes': GaussianNB(),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'ada_boost': AdaBoostClassifier(random_state=42),
            'extra_trees': ExtraTreesClassifier(n_estimators=100, random_state=42)
        }
        
        return self.models
    
    def train_all_models(self, X_train, y_train):
        """Train all models"""
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model
        
        return self.trained_models
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        for name, model in self.trained_models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # ROC AUC for binary classification
            roc_auc = None
            if y_pred_proba is not None and len(np.unique(y_test)) == 2:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc
            }
        
        return self.results
    
    def get_best_model(self, metric='f1_score'):
        """Get best model based on specified metric"""
        valid_results = {k: v for k, v in self.results.items() if v[metric] is not None}
        best_model_name = max(valid_results, key=lambda x: valid_results[x][metric])
        return best_model_name, self.trained_models[best_model_name]
    
    def plot_results(self):
        """Plot model comparison"""
        import matplotlib.pyplot as plt
        
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [self.results[model][metric] for model in models]
            axes[i].bar(models, values)
            axes[i].set_title(f'{metric.capitalize()} Comparison')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

# Advanced classification techniques
class AdvancedClassification:
    """Advanced classification methods"""
    
    @staticmethod
    def ensemble_voting(models, X_train, y_train, X_test, voting='soft'):
        """Create voting ensemble"""
        from sklearn.ensemble import VotingClassifier
        
        voting_clf = VotingClassifier(
            estimators=[(name, model) for name, model in models.items()],
            voting=voting
        )
        
        voting_clf.fit(X_train, y_train)
        predictions = voting_clf.predict(X_test)
        
        if voting == 'soft':
            probabilities = voting_clf.predict_proba(X_test)
            return predictions, probabilities
        
        return predictions
    
    @staticmethod
    def stacking_ensemble(base_models, meta_model, X_train, y_train, X_test, cv=5):
        """Create stacking ensemble"""
        from sklearn.ensemble import StackingClassifier
        from sklearn.model_selection import cross_val_predict
        
        stacking_clf = StackingClassifier(
            estimators=[(name, model) for name, model in base_models.items()],
            final_estimator=meta_model,
            cv=cv
        )
        
        stacking_clf.fit(X_train, y_train)
        predictions = stacking_clf.predict(X_test)
        probabilities = stacking_clf.predict_proba(X_test)
        
        return predictions, probabilities
```

### Regression Algorithms

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class RegressionSuite:
    """Complete regression model suite"""
    
    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self.results = {}
    
    def setup_models(self):
        """Setup regression models"""
        self.models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0, random_state=42),
            'lasso': Lasso(alpha=1.0, random_state=42),
            'elastic_net': ElasticNet(alpha=1.0, random_state=42),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42),
            'svr': SVR(kernel='rbf'),
            'knn': KNeighborsRegressor(n_neighbors=5),
            'decision_tree': DecisionTreeRegressor(random_state=42)
        }
        
        return self.models
    
    def train_all_models(self, X_train, y_train):
        """Train all regression models"""
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model
        
        return self.trained_models
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        for name, model in self.trained_models.items():
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.results[name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2_score': r2
            }
        
        return self.results
    
    def get_feature_importance(self, model_name, feature_names):
        """Get feature importance for tree-based models"""
        model = self.trained_models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            print(f"Model {model_name} does not have feature_importances_ attribute")
            return None
```

## Unsupervised Learning

### Clustering

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score

class ClusteringSuite:
    """Comprehensive clustering analysis"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def setup_clustering_models(self, n_clusters=3):
        """Setup clustering models"""
        self.models = {
            'kmeans': KMeans(n_clusters=n_clusters, random_state=42),
            'gaussian_mixture': GaussianMixture(n_components=n_clusters, random_state=42),
            'hierarchical': AgglomerativeClustering(n_clusters=n_clusters),
            'dbscan': DBSCAN(eps=0.5, min_samples=5)
        }
        
        return self.models
    
    def fit_all_clusters(self, X):
        """Fit all clustering models"""
        for name, model in self.models.items():
            print(f"Fitting {name}...")
            if name == 'gaussian_mixture':
                model.fit(X)
                labels = model.predict(X)
            else:
                labels = model.fit_predict(X)
            
            # Calculate metrics
            if len(np.unique(labels)) > 1:  # Need at least 2 clusters
                silhouette = silhouette_score(X, labels)
                calinski_harabasz = calinski_harabasz_score(X, labels)
            else:
                silhouette = -1
                calinski_harabasz = -1
            
            self.results[name] = {
                'labels': labels,
                'n_clusters': len(np.unique(labels[labels != -1])),  # Exclude noise for DBSCAN
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski_harabasz
            }
        
        return self.results
    
    def find_optimal_clusters(self, X, max_clusters=10, method='elbow'):
        """Find optimal number of clusters"""
        scores = []
        K_range = range(2, max_clusters + 1)
        
        for k in K_range:
            if method == 'elbow':
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(X)
                scores.append(kmeans.inertia_)
            elif method == 'silhouette':
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(X)
                score = silhouette_score(X, labels)
                scores.append(score)
        
        return K_range, scores
    
    def plot_clusters(self, X, labels, title="Clustering Results"):
        """Plot clustering results (for 2D data)"""
        import matplotlib.pyplot as plt
        
        if X.shape[1] == 2:
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
            plt.colorbar(scatter)
            plt.title(title)
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.show()
        else:
            print("Plotting available only for 2D data")

# Dimensionality reduction
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.manifold import TSNE, UMAP

class DimensionalityReduction:
    """Dimensionality reduction techniques"""
    
    @staticmethod
    def apply_pca(X, n_components=0.95, plot=True):
        """Apply PCA with explained variance analysis"""
        pca = PCA(n_components=n_components, random_state=42)
        X_transformed = pca.fit_transform(X)
        
        if plot and n_components > 1:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 4))
            
            # Explained variance plot
            plt.subplot(1, 2, 1)
            plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                    np.cumsum(pca.explained_variance_ratio_))
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Explained Variance')
            plt.title('PCA Explained Variance')
            plt.grid(True)
            
            # Scatter plot for first 2 components
            if X_transformed.shape[1] >= 2:
                plt.subplot(1, 2, 2)
                plt.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.6)
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                plt.title('PCA - First Two Components')
            
            plt.tight_layout()
            plt.show()
        
        return X_transformed, pca
    
    @staticmethod
    def apply_tsne(X, n_components=2, perplexity=30):
        """Apply t-SNE for visualization"""
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        X_transformed = tsne.fit_transform(X)
        
        return X_transformed, tsne
    
    @staticmethod
    def compare_methods(X, y=None, sample_size=1000):
        """Compare different dimensionality reduction methods"""
        import matplotlib.pyplot as plt
        
        # Sample data if too large
        if len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[indices]
            y_sample = y[indices] if y is not None else None
        else:
            X_sample = X
            y_sample = y
        
        # Apply different methods
        methods = {
            'PCA': PCA(n_components=2, random_state=42),
            't-SNE': TSNE(n_components=2, random_state=42),
            'TruncatedSVD': TruncatedSVD(n_components=2, random_state=42)
        }
        
        fig, axes = plt.subplots(1, len(methods), figsize=(15, 5))
        
        for i, (name, method) in enumerate(methods.items()):
            X_transformed = method.fit_transform(X_sample)
            
            if y_sample is not None:
                scatter = axes[i].scatter(X_transformed[:, 0], X_transformed[:, 1], 
                                        c=y_sample, cmap='viridis', alpha=0.6)
                plt.colorbar(scatter, ax=axes[i])
            else:
                axes[i].scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.6)
            
            axes[i].set_title(f'{name}')
            axes[i].set_xlabel('Component 1')
            axes[i].set_ylabel('Component 2')
        
        plt.tight_layout()
        plt.show()
```

## Model Selection and Hyperparameter Tuning

```python
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score,
    StratifiedKFold, train_test_split, validation_curve
)
from sklearn.metrics import make_scorer

class ModelSelection:
    """Advanced model selection and hyperparameter tuning"""
    
    def __init__(self, scoring='accuracy', cv=5):
        self.scoring = scoring
        self.cv = cv
        self.best_models = {}
        
    def grid_search_tuning(self, model, param_grid, X_train, y_train):
        """Perform grid search hyperparameter tuning"""
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        return grid_search
    
    def random_search_tuning(self, model, param_distributions, X_train, y_train, n_iter=100):
        """Perform randomized search hyperparameter tuning"""
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        random_search.fit(X_train, y_train)
        
        return random_search
    
    def cross_validate_model(self, model, X, y):
        """Perform cross-validation"""
        if self.scoring in ['accuracy', 'f1', 'precision', 'recall']:
            cv_strategy = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        else:
            cv_strategy = self.cv
        
        scores = cross_val_score(model, X, y, scoring=self.scoring, cv=cv_strategy, n_jobs=-1)
        
        return {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std(),
            'confidence_interval': (scores.mean() - 1.96*scores.std()/np.sqrt(len(scores)),
                                  scores.mean() + 1.96*scores.std()/np.sqrt(len(scores)))
        }
    
    def learning_curve_analysis(self, model, X, y, train_sizes=None):
        """Generate learning curves"""
        from sklearn.model_selection import learning_curve
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y,
            train_sizes=train_sizes,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=-1
        )
        
        return {
            'train_sizes': train_sizes,
            'train_scores': train_scores,
            'val_scores': val_scores,
            'train_mean': train_scores.mean(axis=1),
            'train_std': train_scores.std(axis=1),
            'val_mean': val_scores.mean(axis=1),
            'val_std': val_scores.std(axis=1)
        }
    
    def plot_learning_curve(self, learning_curve_data, title="Learning Curve"):
        """Plot learning curves"""
        import matplotlib.pyplot as plt
        
        train_sizes = learning_curve_data['train_sizes']
        train_mean = learning_curve_data['train_mean']
        train_std = learning_curve_data['train_std']
        val_mean = learning_curve_data['val_mean']
        val_std = learning_curve_data['val_std']
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                        alpha=0.1, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                        alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

# Automated hyperparameter tuning
class AutoMLTuning:
    """Automated machine learning tuning"""
    
    @staticmethod
    def get_default_param_grids():
        """Get default parameter grids for common algorithms"""
        return {
            'RandomForestClassifier': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'LogisticRegression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga']
            },
            'SVC': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'poly', 'sigmoid'],
                'gamma': ['scale', 'auto', 0.1, 1]
            },
            'GradientBoostingClassifier': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }
    
    @staticmethod
    def auto_tune_models(models, X_train, y_train, X_test, y_test, scoring='accuracy'):
        """Automatically tune multiple models"""
        param_grids = AutoMLTuning.get_default_param_grids()
        tuned_models = {}
        results = {}
        
        for name, model in models.items():
            model_class = model.__class__.__name__
            
            if model_class in param_grids:
                print(f"Tuning {name}...")
                
                grid_search = GridSearchCV(
                    model, param_grids[model_class],
                    scoring=scoring, cv=5, n_jobs=-1
                )
                
                grid_search.fit(X_train, y_train)
                
                # Evaluate on test set
                test_score = grid_search.score(X_test, y_test)
                
                tuned_models[name] = grid_search.best_estimator_
                results[name] = {
                    'best_params': grid_search.best_params_,
                    'best_cv_score': grid_search.best_score_,
                    'test_score': test_score
                }
            else:
                print(f"No parameter grid available for {model_class}")
        
        return tuned_models, results
```

---

*This comprehensive scikit-learn guide provides complete coverage of machine learning workflows from data preprocessing to model deployment.* 