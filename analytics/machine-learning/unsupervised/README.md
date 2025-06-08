# Unsupervised Machine Learning

## Overview

Unsupervised learning discovers hidden patterns and structures in data without labeled examples. This directory covers clustering algorithms, dimensionality reduction techniques, anomaly detection, and association rule mining with practical implementations in Python and R.

## Table of Contents

- [Clustering Algorithms](#clustering-algorithms)
- [Dimensionality Reduction](#dimensionality-reduction)
- [Anomaly Detection](#anomaly-detection)
- [Association Rule Mining](#association-rule-mining)
- [Evaluation Metrics](#evaluation-metrics)
- [Feature Engineering](#feature-engineering)
- [Case Studies](#case-studies)
- [Best Practices](#best-practices)

## Clustering Algorithms

### K-Means Clustering

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Generate sample data
X, y_true = make_blobs(n_samples=300, centers=4, n_features=2, 
                       random_state=0, cluster_std=0.6)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means
kmeans = KMeans(n_clusters=4, random_state=0, n_init=10)
y_pred = kmeans.fit_predict(X_scaled)

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis')
plt.title('True Clusters')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           c='red', marker='x', s=200, linewidths=3)
plt.title('K-Means Clustering')
plt.show()

print(f"Inertia: {kmeans.inertia_:.2f}")
```

### Elbow Method for Optimal K

```python
def find_optimal_clusters(X, max_k=10):
    """Find optimal number of clusters using elbow method"""
    
    inertias = []
    K_range = range(1, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertias, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.show()
    
    return inertias

# Find optimal clusters
inertias = find_optimal_clusters(X_scaled)
```

### DBSCAN Clustering

```python
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

def dbscan_clustering(X, eps=0.5, min_samples=5):
    """Apply DBSCAN clustering"""
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(X)
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f'Estimated number of clusters: {n_clusters}')
    print(f'Estimated number of noise points: {n_noise}')
    
    return cluster_labels

# Find optimal eps using k-distance graph
def find_optimal_eps(X, k=4):
    """Find optimal eps parameter for DBSCAN"""
    
    neigh = NearestNeighbors(n_neighbors=k)
    nbrs = neigh.fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    # Sort distances
    distances = np.sort(distances, axis=0)
    distances = distances[:, k-1]
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{k}-NN distance')
    plt.title('k-distance Graph for Optimal eps')
    plt.grid(True)
    plt.show()
    
    return distances

# Apply DBSCAN
eps_distances = find_optimal_eps(X_scaled)
dbscan_labels = dbscan_clustering(X_scaled, eps=0.3, min_samples=5)

# Plot DBSCAN results
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis')
plt.title('DBSCAN Clustering')
plt.show()
```

### Hierarchical Clustering

```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering

def hierarchical_clustering(X, method='ward', n_clusters=4):
    """Apply hierarchical clustering"""
    
    # Calculate linkage matrix
    Z = linkage(X, method=method)
    
    # Plot dendrogram
    plt.figure(figsize=(15, 8))
    dendrogram(Z, truncate_mode='level', p=5)
    plt.title(f'Hierarchical Clustering Dendrogram ({method})')
    plt.xlabel('Sample index or (cluster size)')
    plt.ylabel('Distance')
    plt.show()
    
    # Get cluster labels
    cluster_labels = fcluster(Z, n_clusters, criterion='maxclust')
    
    return cluster_labels, Z

# Apply hierarchical clustering
hier_labels, linkage_matrix = hierarchical_clustering(X_scaled)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=hier_labels, cmap='viridis')
plt.title('Hierarchical Clustering')
plt.show()
```

### Gaussian Mixture Models

```python
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV

def gmm_clustering(X, n_components_range=range(1, 11)):
    """Apply Gaussian Mixture Model clustering"""
    
    # Find optimal number of components using BIC
    bic_scores = []
    aic_scores = []
    
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, random_state=0)
        gmm.fit(X)
        bic_scores.append(gmm.bic(X))
        aic_scores.append(gmm.aic(X))
    
    # Plot BIC and AIC scores
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(n_components_range, bic_scores, 'bo-', label='BIC')
    plt.plot(n_components_range, aic_scores, 'ro-', label='AIC')
    plt.xlabel('Number of components')
    plt.ylabel('Information criterion')
    plt.title('Model Selection for GMM')
    plt.legend()
    plt.grid(True)
    
    # Fit optimal model
    optimal_components = n_components_range[np.argmin(bic_scores)]
    gmm_optimal = GaussianMixture(n_components=optimal_components, random_state=0)
    gmm_labels = gmm_optimal.fit_predict(X)
    
    # Plot clustering results
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=gmm_labels, cmap='viridis')
    plt.title(f'GMM Clustering (k={optimal_components})')
    
    plt.tight_layout()
    plt.show()
    
    return gmm_labels, gmm_optimal

# Apply GMM clustering
gmm_labels, gmm_model = gmm_clustering(X_scaled)
```

## Dimensionality Reduction

### Principal Component Analysis (PCA)

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

# Load high-dimensional dataset
digits = load_digits()
X_digits = digits.data
y_digits = digits.target

print(f"Original shape: {X_digits.shape}")

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_digits)

# Plot explained variance ratio
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('PCA Explained Variance')
plt.grid(True)

# Choose number of components for 95% variance
n_components_95 = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
print(f"Components for 95% variance: {n_components_95}")

# Apply PCA with selected components
pca_reduced = PCA(n_components=n_components_95)
X_pca_reduced = pca_reduced.fit_transform(X_digits)

# Visualize first two components
plt.subplot(1, 2, 2)
plt.scatter(X_pca_reduced[:, 0], X_pca_reduced[:, 1], c=y_digits, cmap='tab10')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Visualization')
plt.colorbar()

plt.tight_layout()
plt.show()

print(f"Reduced shape: {X_pca_reduced.shape}")
print(f"Variance explained: {pca_reduced.explained_variance_ratio_.sum():.3f}")
```

### t-SNE (t-Distributed Stochastic Neighbor Embedding)

```python
from sklearn.manifold import TSNE

def apply_tsne(X, perplexity=30, n_iter=1000, random_state=0):
    """Apply t-SNE dimensionality reduction"""
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, 
                n_iter=n_iter, random_state=random_state)
    X_tsne = tsne.fit_transform(X)
    
    return X_tsne

# Apply t-SNE to digits dataset (use PCA first for efficiency)
X_pca_50 = PCA(n_components=50).fit_transform(X_digits)
X_tsne = apply_tsne(X_pca_50)

# Visualize t-SNE results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_digits, cmap='tab10')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNE Visualization of Digits Dataset')
plt.colorbar(scatter)
plt.show()
```

### UMAP (Uniform Manifold Approximation and Projection)

```python
import umap

def apply_umap(X, n_neighbors=15, min_dist=0.1, random_state=0):
    """Apply UMAP dimensionality reduction"""
    
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, 
                       random_state=random_state)
    X_umap = reducer.fit_transform(X)
    
    return X_umap

# Apply UMAP
X_umap = apply_umap(X_digits)

# Compare PCA, t-SNE, and UMAP
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# PCA
axes[0].scatter(X_pca_reduced[:, 0], X_pca_reduced[:, 1], c=y_digits, cmap='tab10')
axes[0].set_title('PCA')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')

# t-SNE
axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_digits, cmap='tab10')
axes[1].set_title('t-SNE')
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')

# UMAP
scatter = axes[2].scatter(X_umap[:, 0], X_umap[:, 1], c=y_digits, cmap='tab10')
axes[2].set_title('UMAP')
axes[2].set_xlabel('UMAP 1')
axes[2].set_ylabel('UMAP 2')

plt.colorbar(scatter, ax=axes, shrink=0.8)
plt.tight_layout()
plt.show()
```

### Factor Analysis

```python
from sklearn.decomposition import FactorAnalysis

def factor_analysis(X, n_factors=None, max_factors=10):
    """Apply Factor Analysis"""
    
    if n_factors is None:
        # Find optimal number of factors using log-likelihood
        ll_scores = []
        factor_range = range(1, max_factors + 1)
        
        for n in factor_range:
            fa = FactorAnalysis(n_components=n, random_state=0)
            fa.fit(X)
            ll_scores.append(fa.score(X))
        
        # Plot log-likelihood scores
        plt.figure(figsize=(10, 6))
        plt.plot(factor_range, ll_scores, 'bo-')
        plt.xlabel('Number of factors')
        plt.ylabel('Log-likelihood')
        plt.title('Factor Analysis Model Selection')
        plt.grid(True)
        plt.show()
        
        # Choose optimal number of factors
        n_factors = factor_range[np.argmax(ll_scores)]
    
    # Apply Factor Analysis
    fa = FactorAnalysis(n_components=n_factors, random_state=0)
    X_fa = fa.fit_transform(X)
    
    print(f"Optimal number of factors: {n_factors}")
    
    return X_fa, fa

# Apply Factor Analysis
X_fa, fa_model = factor_analysis(X_digits)
```

## Anomaly Detection

### Isolation Forest

```python
from sklearn.ensemble import IsolationForest

def isolation_forest_anomaly_detection(X, contamination=0.1):
    """Detect anomalies using Isolation Forest"""
    
    # Fit Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=0)
    anomaly_labels = iso_forest.fit_predict(X)
    
    # Convert to binary labels (1: normal, 0: anomaly)
    anomaly_binary = (anomaly_labels == 1).astype(int)
    
    return anomaly_binary, iso_forest

# Generate dataset with anomalies
np.random.seed(0)
normal_data = np.random.normal(0, 1, (1000, 2))
anomaly_data = np.random.normal(3, 0.5, (50, 2))
X_anomaly = np.vstack([normal_data, anomaly_data])

# Detect anomalies
anomaly_pred, iso_model = isolation_forest_anomaly_detection(X_anomaly)

# Visualize results
plt.figure(figsize=(10, 6))
normal_mask = anomaly_pred == 1
anomaly_mask = anomaly_pred == 0

plt.scatter(X_anomaly[normal_mask, 0], X_anomaly[normal_mask, 1], 
           c='blue', label='Normal', alpha=0.6)
plt.scatter(X_anomaly[anomaly_mask, 0], X_anomaly[anomaly_mask, 1], 
           c='red', label='Anomaly', alpha=0.8)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Isolation Forest Anomaly Detection')
plt.legend()
plt.show()

print(f"Number of detected anomalies: {np.sum(anomaly_mask)}")
```

### Local Outlier Factor (LOF)

```python
from sklearn.neighbors import LocalOutlierFactor

def lof_anomaly_detection(X, n_neighbors=20, contamination=0.1):
    """Detect anomalies using Local Outlier Factor"""
    
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, 
                            contamination=contamination)
    anomaly_labels = lof.fit_predict(X)
    
    # Get outlier scores
    outlier_scores = lof.negative_outlier_factor_
    
    return anomaly_labels, outlier_scores

# Apply LOF
lof_labels, lof_scores = lof_anomaly_detection(X_anomaly)

# Visualize results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
normal_mask = lof_labels == 1
anomaly_mask = lof_labels == -1

plt.scatter(X_anomaly[normal_mask, 0], X_anomaly[normal_mask, 1], 
           c='blue', label='Normal', alpha=0.6)
plt.scatter(X_anomaly[anomaly_mask, 0], X_anomaly[anomaly_mask, 1], 
           c='red', label='Anomaly', alpha=0.8)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('LOF Anomaly Detection')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_anomaly[:, 0], X_anomaly[:, 1], c=-lof_scores, cmap='viridis')
plt.colorbar(label='Outlier Score')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('LOF Outlier Scores')

plt.tight_layout()
plt.show()
```

### One-Class SVM

```python
from sklearn.svm import OneClassSVM

def oneclass_svm_anomaly_detection(X, nu=0.05, gamma='scale'):
    """Detect anomalies using One-Class SVM"""
    
    oc_svm = OneClassSVM(nu=nu, gamma=gamma)
    anomaly_labels = oc_svm.fit_predict(X)
    
    return anomaly_labels, oc_svm

# Apply One-Class SVM
ocsvm_labels, ocsvm_model = oneclass_svm_anomaly_detection(X_anomaly)

# Visualize results
plt.figure(figsize=(10, 6))
normal_mask = ocsvm_labels == 1
anomaly_mask = ocsvm_labels == -1

plt.scatter(X_anomaly[normal_mask, 0], X_anomaly[normal_mask, 1], 
           c='blue', label='Normal', alpha=0.6)
plt.scatter(X_anomaly[anomaly_mask, 0], X_anomaly[anomaly_mask, 1], 
           c='red', label='Anomaly', alpha=0.8)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('One-Class SVM Anomaly Detection')
plt.legend()
plt.show()
```

## Association Rule Mining

### Market Basket Analysis

```python
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def market_basket_analysis(transactions_df, min_support=0.01, metric='confidence', min_threshold=0.5):
    """Perform market basket analysis"""
    
    # Find frequent itemsets
    frequent_itemsets = apriori(transactions_df, min_support=min_support, use_colnames=True)
    
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
    
    return frequent_itemsets, rules

# Create sample transaction data
transactions = [
    ['Milk', 'Eggs', 'Bread', 'Cheese'],
    ['Eggs', 'Bread'],
    ['Milk', 'Bread'],
    ['Eggs', 'Bread', 'Butter'],
    ['Milk', 'Eggs', 'Bread', 'Butter'],
    ['Milk', 'Eggs', 'Butter'],
    ['Eggs', 'Bread'],
    ['Milk', 'Cheese'],
    ['Milk', 'Eggs', 'Cheese'],
    ['Bread', 'Butter']
]

# Convert to binary matrix
from sklearn.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_transactions = pd.DataFrame(te_ary, columns=te.columns_)

print("Transaction matrix:")
print(df_transactions.head())

# Perform market basket analysis
frequent_items, rules = market_basket_analysis(df_transactions)

print("\nFrequent itemsets:")
print(frequent_items)

print("\nAssociation rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())
```

## Evaluation Metrics

### Clustering Evaluation

```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

def evaluate_clustering(X, labels_true, labels_pred):
    """Evaluate clustering performance"""
    
    metrics = {}
    
    # Internal metrics (don't require true labels)
    metrics['silhouette_score'] = silhouette_score(X, labels_pred)
    metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels_pred)
    metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels_pred)
    
    # External metrics (require true labels)
    if labels_true is not None:
        metrics['adjusted_rand_score'] = adjusted_rand_score(labels_true, labels_pred)
        metrics['adjusted_mutual_info_score'] = adjusted_mutual_info_score(labels_true, labels_pred)
    
    return metrics

# Evaluate different clustering algorithms
clustering_results = {
    'K-Means': y_pred,
    'DBSCAN': dbscan_labels,
    'Hierarchical': hier_labels - 1,  # Convert to 0-based indexing
    'GMM': gmm_labels
}

print("Clustering Evaluation Results:")
print("-" * 60)

for name, labels in clustering_results.items():
    if len(np.unique(labels)) > 1:  # Skip if only one cluster
        metrics = evaluate_clustering(X_scaled, y_true, labels)
        print(f"\n{name}:")
        for metric_name, score in metrics.items():
            print(f"  {metric_name}: {score:.3f}")
```

### Dimensionality Reduction Evaluation

```python
def evaluate_dimensionality_reduction(X_original, X_reduced, method_name):
    """Evaluate dimensionality reduction quality"""
    
    # Reconstruction error (for linear methods)
    if method_name in ['PCA', 'Factor Analysis']:
        # For PCA, we can reconstruct
        if method_name == 'PCA':
            pca_temp = PCA(n_components=X_reduced.shape[1])
            pca_temp.fit(X_original)
            X_reconstructed = pca_temp.inverse_transform(X_reduced)
            reconstruction_error = np.mean((X_original - X_reconstructed) ** 2)
            print(f"{method_name} Reconstruction Error: {reconstruction_error:.4f}")
    
    # Variance preserved (for PCA)
    if method_name == 'PCA':
        pca_temp = PCA(n_components=X_reduced.shape[1])
        pca_temp.fit(X_original)
        variance_ratio = pca_temp.explained_variance_ratio_.sum()
        print(f"{method_name} Variance Preserved: {variance_ratio:.3f}")
    
    # Dimensionality reduction ratio
    reduction_ratio = X_reduced.shape[1] / X_original.shape[1]
    print(f"{method_name} Dimensionality Reduction: {X_original.shape[1]} -> {X_reduced.shape[1]} ({reduction_ratio:.3f})")

# Evaluate dimensionality reduction methods
evaluate_dimensionality_reduction(X_digits, X_pca_reduced, 'PCA')
evaluate_dimensionality_reduction(X_digits, X_fa, 'Factor Analysis')
```

## Feature Engineering

### Feature Selection for Unsupervised Learning

```python
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.decomposition import PCA

def unsupervised_feature_selection(X, variance_threshold=0.01, correlation_threshold=0.95):
    """Perform feature selection for unsupervised learning"""
    
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Remove low variance features
    selector = VarianceThreshold(threshold=variance_threshold)
    X_variance_filtered = selector.fit_transform(X)
    selected_features = np.array(feature_names)[selector.get_support()]
    
    print(f"Features after variance filtering: {X_variance_filtered.shape[1]}")
    
    # Remove highly correlated features
    if X_variance_filtered.shape[1] > 1:
        corr_matrix = np.corrcoef(X_variance_filtered.T)
        upper_tri = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        high_corr_pairs = np.where((np.abs(corr_matrix) > correlation_threshold) & upper_tri)
        
        features_to_remove = set()
        for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
            features_to_remove.add(j)
        
        features_to_keep = [i for i in range(X_variance_filtered.shape[1]) if i not in features_to_remove]
        X_corr_filtered = X_variance_filtered[:, features_to_keep]
        
        print(f"Features after correlation filtering: {X_corr_filtered.shape[1]}")
    else:
        X_corr_filtered = X_variance_filtered
    
    return X_corr_filtered

# Apply feature selection
X_selected = unsupervised_feature_selection(X_digits)
```

### Automated Feature Engineering

```python
def create_polynomial_features(X, degree=2, include_bias=False):
    """Create polynomial features"""
    from sklearn.preprocessing import PolynomialFeatures
    
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    X_poly = poly.fit_transform(X)
    
    return X_poly, poly

def create_interaction_features(X):
    """Create interaction features between all pairs"""
    
    n_features = X.shape[1]
    interactions = []
    
    for i in range(n_features):
        for j in range(i+1, n_features):
            interaction = X[:, i] * X[:, j]
            interactions.append(interaction)
    
    if interactions:
        X_interactions = np.column_stack(interactions)
        X_extended = np.column_stack([X, X_interactions])
    else:
        X_extended = X
    
    return X_extended

# Example with smaller dataset
X_small = X_digits[:100, :8]  # Use first 8 features for demo

# Create polynomial features
X_poly, poly_transformer = create_polynomial_features(X_small, degree=2)
print(f"Original features: {X_small.shape[1]}")
print(f"Polynomial features: {X_poly.shape[1]}")

# Create interaction features
X_interactions = create_interaction_features(X_small)
print(f"With interactions: {X_interactions.shape[1]}")
```

## Case Studies

### Customer Segmentation

```python
def customer_segmentation_analysis():
    """Complete customer segmentation pipeline"""
    
    # Generate synthetic customer data
    np.random.seed(42)
    n_customers = 1000
    
    # Customer features
    age = np.random.normal(40, 15, n_customers)
    income = np.random.normal(50000, 20000, n_customers)
    spending_score = np.random.normal(50, 25, n_customers)
    frequency = np.random.poisson(10, n_customers)
    
    customer_data = np.column_stack([age, income, spending_score, frequency])
    feature_names = ['Age', 'Income', 'Spending_Score', 'Frequency']
    
    # Preprocessing
    scaler = StandardScaler()
    customer_scaled = scaler.fit_transform(customer_data)
    
    # Apply multiple clustering algorithms
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans_labels = kmeans.fit_predict(customer_scaled)
    
    gmm = GaussianMixture(n_components=4, random_state=42)
    gmm_labels = gmm.fit_predict(customer_scaled)
    
    # Dimensionality reduction for visualization
    pca_viz = PCA(n_components=2)
    customer_pca = pca_viz.fit_transform(customer_scaled)
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # K-Means results
    axes[0, 0].scatter(customer_pca[:, 0], customer_pca[:, 1], c=kmeans_labels, cmap='viridis')
    axes[0, 0].set_title('K-Means Customer Segmentation')
    axes[0, 0].set_xlabel('PC1')
    axes[0, 0].set_ylabel('PC2')
    
    # GMM results
    axes[0, 1].scatter(customer_pca[:, 0], customer_pca[:, 1], c=gmm_labels, cmap='viridis')
    axes[0, 1].set_title('GMM Customer Segmentation')
    axes[0, 1].set_xlabel('PC1')
    axes[0, 1].set_ylabel('PC2')
    
    # Feature importance (PCA loadings)
    loadings = pca_viz.components_.T * np.sqrt(pca_viz.explained_variance_)
    axes[1, 0].bar(feature_names, loadings[:, 0])
    axes[1, 0].set_title('PC1 Feature Loadings')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    axes[1, 1].bar(feature_names, loadings[:, 1])
    axes[1, 1].set_title('PC2 Feature Loadings')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Segment analysis
    df_customers = pd.DataFrame(customer_data, columns=feature_names)
    df_customers['KMeans_Segment'] = kmeans_labels
    df_customers['GMM_Segment'] = gmm_labels
    
    print("K-Means Segment Statistics:")
    print(df_customers.groupby('KMeans_Segment')[feature_names].mean())
    
    return df_customers

# Run customer segmentation analysis
customer_segments = customer_segmentation_analysis()
```

### Fraud Detection

```python
def fraud_detection_pipeline():
    """Anomaly detection for fraud detection"""
    
    # Generate synthetic transaction data
    np.random.seed(42)
    n_transactions = 5000
    
    # Normal transactions
    normal_amount = np.random.lognormal(mean=3, sigma=1, size=int(n_transactions * 0.95))
    normal_time = np.random.uniform(6, 22, size=int(n_transactions * 0.95))  # Business hours
    normal_merchant_cat = np.random.choice([1, 2, 3, 4, 5], size=int(n_transactions * 0.95))
    
    # Fraudulent transactions
    fraud_amount = np.random.lognormal(mean=5, sigma=0.5, size=int(n_transactions * 0.05))
    fraud_time = np.random.uniform(0, 6, size=int(n_transactions * 0.05))  # Unusual hours
    fraud_merchant_cat = np.random.choice([6, 7, 8], size=int(n_transactions * 0.05))
    
    # Combine data
    amounts = np.concatenate([normal_amount, fraud_amount])
    times = np.concatenate([normal_time, fraud_time])
    merchant_cats = np.concatenate([normal_merchant_cat, fraud_merchant_cat])
    
    # Create features
    transaction_data = np.column_stack([amounts, times, merchant_cats])
    
    # True labels (for evaluation)
    true_labels = np.concatenate([np.ones(int(n_transactions * 0.95)), 
                                 np.zeros(int(n_transactions * 0.05))])
    
    # Feature engineering
    scaler = StandardScaler()
    transaction_scaled = scaler.fit_transform(transaction_data)
    
    # Apply multiple anomaly detection methods
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_pred = iso_forest.fit_predict(transaction_scaled)
    iso_pred_binary = (iso_pred == 1).astype(int)
    
    lof = LocalOutlierFactor(contamination=0.05)
    lof_pred = lof.fit_predict(transaction_scaled)
    lof_pred_binary = (lof_pred == 1).astype(int)
    
    # Evaluation
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("Isolation Forest Results:")
    print(classification_report(true_labels, iso_pred_binary))
    
    print("\nLocal Outlier Factor Results:")
    print(classification_report(true_labels, lof_pred_binary))
    
    # Visualization
    pca_fraud = PCA(n_components=2)
    transaction_pca = pca_fraud.fit_transform(transaction_scaled)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # True labels
    axes[0].scatter(transaction_pca[true_labels==1, 0], transaction_pca[true_labels==1, 1], 
                   c='blue', label='Normal', alpha=0.6)
    axes[0].scatter(transaction_pca[true_labels==0, 0], transaction_pca[true_labels==0, 1], 
                   c='red', label='Fraud', alpha=0.8)
    axes[0].set_title('True Labels')
    axes[0].legend()
    
    # Isolation Forest
    axes[1].scatter(transaction_pca[iso_pred_binary==1, 0], transaction_pca[iso_pred_binary==1, 1], 
                   c='blue', label='Normal', alpha=0.6)
    axes[1].scatter(transaction_pca[iso_pred_binary==0, 0], transaction_pca[iso_pred_binary==0, 1], 
                   c='red', label='Fraud', alpha=0.8)
    axes[1].set_title('Isolation Forest')
    axes[1].legend()
    
    # LOF
    axes[2].scatter(transaction_pca[lof_pred_binary==1, 0], transaction_pca[lof_pred_binary==1, 1], 
                   c='blue', label='Normal', alpha=0.6)
    axes[2].scatter(transaction_pca[lof_pred_binary==0, 0], transaction_pca[lof_pred_binary==0, 1], 
                   c='red', label='Fraud', alpha=0.8)
    axes[2].set_title('Local Outlier Factor')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()

# Run fraud detection pipeline
fraud_detection_pipeline()
```

## Best Practices

### 1. Data Preprocessing
- **Scaling**: Always scale features for distance-based algorithms
- **Handling Missing Values**: Use appropriate imputation strategies
- **Outlier Treatment**: Consider robust scaling or outlier removal
- **Feature Selection**: Remove irrelevant or redundant features

### 2. Algorithm Selection
- **K-Means**: Good for spherical clusters, requires pre-specified k
- **DBSCAN**: Handles arbitrary shapes, automatically determines clusters
- **Hierarchical**: Good for understanding cluster hierarchy
- **GMM**: Handles overlapping clusters with probabilistic assignments

### 3. Validation Strategies
- **Cross-Validation**: Use appropriate CV for unsupervised learning
- **Stability Analysis**: Check consistency across different runs
- **Multiple Metrics**: Use various evaluation metrics
- **Domain Knowledge**: Validate results with subject matter experts

### 4. Hyperparameter Tuning
- **Grid Search**: Systematic parameter exploration
- **Random Search**: Efficient for high-dimensional spaces
- **Bayesian Optimization**: Advanced parameter optimization
- **Validation Curves**: Understand parameter sensitivity

### 5. Interpretability
- **Feature Importance**: Understand which features drive patterns
- **Visualization**: Use dimensionality reduction for insights
- **Cluster Profiling**: Analyze cluster characteristics
- **Documentation**: Record assumptions and decisions

---

*This comprehensive guide covers the essential aspects of unsupervised machine learning. Experiment with different algorithms and techniques to find the best approaches for your specific use cases.* 