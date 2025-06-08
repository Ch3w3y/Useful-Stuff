# Development & Data Science Best Practices

## Overview

Comprehensive collection of industry best practices, coding standards, and methodological guidelines for software development, data science, machine learning, and system architecture. Includes security practices, testing strategies, and operational excellence principles.

## Table of Contents

- [Software Development Best Practices](#software-development-best-practices)
- [Data Science Best Practices](#data-science-best-practices)
- [Machine Learning Best Practices](#machine-learning-best-practices)
- [Security Best Practices](#security-best-practices)
- [Architecture Best Practices](#architecture-best-practices)
- [Testing Best Practices](#testing-best-practices)
- [DevOps & Operations](#devops--operations)
- [Code Quality & Standards](#code-quality--standards)

## Software Development Best Practices

### Clean Code Principles

#### Code Organization
```python
# Good: Clear, descriptive naming and single responsibility
class UserAuthenticationService:
    def __init__(self, password_validator, token_generator):
        self._password_validator = password_validator
        self._token_generator = token_generator
    
    def authenticate_user(self, email: str, password: str) -> AuthResult:
        """Authenticate user with email and password."""
        if not self._is_valid_email(email):
            return AuthResult.invalid_email()
        
        user = self._get_user_by_email(email)
        if not user:
            return AuthResult.user_not_found()
        
        if not self._password_validator.verify(password, user.password_hash):
            return AuthResult.invalid_password()
        
        token = self._token_generator.generate_token(user)
        return AuthResult.success(user, token)

# Bad: Poor naming, mixed responsibilities
class Auth:
    def do_auth(self, e, p):
        u = get_user(e)
        if check_pass(p, u.p):
            return make_token(u)
        return None
```

#### Function Design
```python
# Good: Pure functions with clear contracts
def calculate_discount(price: Decimal, discount_rate: float, 
                      membership_tier: str) -> Decimal:
    """
    Calculate discount amount for a given price.
    
    Args:
        price: Original price (must be positive)
        discount_rate: Discount rate between 0.0 and 1.0
        membership_tier: Customer membership level
    
    Returns:
        Discount amount to be applied
    
    Raises:
        ValueError: If price is negative or discount_rate is invalid
    """
    if price < 0:
        raise ValueError("Price must be positive")
    
    if not 0 <= discount_rate <= 1:
        raise ValueError("Discount rate must be between 0 and 1")
    
    base_discount = price * Decimal(str(discount_rate))
    
    membership_multiplier = {
        'bronze': Decimal('1.0'),
        'silver': Decimal('1.1'),
        'gold': Decimal('1.2'),
        'platinum': Decimal('1.3')
    }.get(membership_tier.lower(), Decimal('1.0'))
    
    return base_discount * membership_multiplier

# Bad: Side effects, unclear behavior
def calc_disc(p, d, m):
    global last_discount
    last_discount = p * d
    if m == 'gold':
        last_discount *= 1.2
    return last_discount
```

### SOLID Principles Implementation

#### Single Responsibility Principle
```python
# Good: Each class has one reason to change
class EmailValidator:
    def validate(self, email: str) -> bool:
        """Validate email format."""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

class PasswordHasher:
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        import bcrypt
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    
    def verify_password(self, password: str, hash_str: str) -> bool:
        """Verify password against hash."""
        import bcrypt
        return bcrypt.checkpw(password.encode(), hash_str.encode())

class UserRepository:
    def save_user(self, user: User) -> None:
        """Save user to database."""
        pass
    
    def find_by_email(self, email: str) -> Optional[User]:
        """Find user by email address."""
        pass

# Bad: Multiple responsibilities in one class
class UserManager:
    def validate_email(self, email):
        # Email validation logic
        pass
    
    def hash_password(self, password):
        # Password hashing logic
        pass
    
    def save_to_database(self, user):
        # Database operations
        pass
    
    def send_welcome_email(self, user):
        # Email sending logic
        pass
```

#### Open/Closed Principle
```python
# Good: Open for extension, closed for modification
from abc import ABC, abstractmethod

class PaymentProcessor(ABC):
    @abstractmethod
    def process_payment(self, amount: Decimal, payment_data: dict) -> PaymentResult:
        pass

class CreditCardProcessor(PaymentProcessor):
    def process_payment(self, amount: Decimal, payment_data: dict) -> PaymentResult:
        # Credit card processing logic
        return PaymentResult.success(transaction_id="cc_123")

class PayPalProcessor(PaymentProcessor):
    def process_payment(self, amount: Decimal, payment_data: dict) -> PaymentResult:
        # PayPal processing logic
        return PaymentResult.success(transaction_id="pp_456")

class PaymentService:
    def __init__(self, processors: List[PaymentProcessor]):
        self._processors = {
            processor.__class__.__name__.lower(): processor 
            for processor in processors
        }
    
    def process_payment(self, payment_method: str, amount: Decimal, 
                       payment_data: dict) -> PaymentResult:
        processor = self._processors.get(payment_method)
        if not processor:
            return PaymentResult.error("Unsupported payment method")
        
        return processor.process_payment(amount, payment_data)
```

### Error Handling Best Practices

```python
# Good: Specific exception handling with proper logging
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class ServiceError(Exception):
    """Base exception for service errors."""
    pass

class UserNotFoundError(ServiceError):
    """Raised when user is not found."""
    pass

class DatabaseConnectionError(ServiceError):
    """Raised when database connection fails."""
    pass

class UserService:
    def get_user(self, user_id: int) -> User:
        """
        Get user by ID.
        
        Raises:
            UserNotFoundError: When user doesn't exist
            DatabaseConnectionError: When database is unavailable
        """
        try:
            return self._repository.find_by_id(user_id)
        except ConnectionError as e:
            logger.error(f"Database connection failed: {e}")
            raise DatabaseConnectionError("Unable to connect to database") from e
        except NotFoundException as e:
            logger.info(f"User {user_id} not found")
            raise UserNotFoundError(f"User with ID {user_id} not found") from e
        except Exception as e:
            logger.exception(f"Unexpected error retrieving user {user_id}")
            raise ServiceError("An unexpected error occurred") from e

# Good: Using context managers for resource management
def process_file(file_path: str) -> ProcessingResult:
    """Process file with proper resource management."""
    try:
        with open(file_path, 'r') as file:
            data = file.read()
            return process_data(data)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return ProcessingResult.error("File not found")
    except PermissionError:
        logger.error(f"Permission denied: {file_path}")
        return ProcessingResult.error("Permission denied")
    except Exception as e:
        logger.exception(f"Error processing file {file_path}")
        return ProcessingResult.error("Processing failed")

# Bad: Catching all exceptions, poor error handling
def bad_get_user(user_id):
    try:
        return db.get_user(user_id)
    except:
        return None  # Lost all error information
```

## Data Science Best Practices

### Data Validation and Quality

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """Comprehensive data validation for datasets."""
    
    def __init__(self, schema: Dict[str, Dict]):
        """
        Initialize validator with schema.
        
        Args:
            schema: Dictionary defining expected data types and constraints
        """
        self.schema = schema
        self.validation_results = {}
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate entire dataset against schema.
        
        Returns:
            Dictionary containing validation results
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        # Check schema compliance
        schema_results = self._validate_schema(df)
        results['errors'].extend(schema_results['errors'])
        
        # Check data quality
        quality_results = self._validate_quality(df)
        results['warnings'].extend(quality_results['warnings'])
        results['metrics'].update(quality_results['metrics'])
        
        # Check for duplicates
        duplicate_results = self._check_duplicates(df)
        results['warnings'].extend(duplicate_results['warnings'])
        results['metrics'].update(duplicate_results['metrics'])
        
        results['is_valid'] = len(results['errors']) == 0
        
        return results
    
    def _validate_schema(self, df: pd.DataFrame) -> Dict[str, List]:
        """Validate dataframe against expected schema."""
        errors = []
        
        # Check required columns
        missing_cols = set(self.schema.keys()) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check data types
        for col, constraints in self.schema.items():
            if col not in df.columns:
                continue
                
            expected_type = constraints.get('type')
            if expected_type and not df[col].dtype == expected_type:
                errors.append(f"Column {col} has incorrect type: "
                            f"expected {expected_type}, got {df[col].dtype}")
        
        return {'errors': errors}
    
    def _validate_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """Check data quality metrics."""
        warnings = []
        metrics = {}
        
        # Missing values analysis
        missing_pct = (df.isnull().sum() / len(df)) * 100
        metrics['missing_percentage'] = missing_pct.to_dict()
        
        high_missing = missing_pct[missing_pct > 10]
        if not high_missing.empty:
            warnings.append(f"High missing values in columns: {high_missing.to_dict()}")
        
        # Outlier detection for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_metrics = {}
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_pct = (len(outliers) / len(df)) * 100
            outlier_metrics[col] = outlier_pct
            
            if outlier_pct > 5:
                warnings.append(f"High outlier percentage in {col}: {outlier_pct:.2f}%")
        
        metrics['outlier_percentage'] = outlier_metrics
        
        return {'warnings': warnings, 'metrics': metrics}
    
    def _check_duplicates(self, df: pd.DataFrame) -> Dict[str, any]:
        """Check for duplicate records."""
        warnings = []
        metrics = {}
        
        duplicate_count = df.duplicated().sum()
        duplicate_pct = (duplicate_count / len(df)) * 100
        
        metrics['duplicate_count'] = duplicate_count
        metrics['duplicate_percentage'] = duplicate_pct
        
        if duplicate_pct > 1:
            warnings.append(f"High duplicate percentage: {duplicate_pct:.2f}%")
        
        return {'warnings': warnings, 'metrics': metrics}

# Usage example
schema = {
    'user_id': {'type': 'int64', 'nullable': False},
    'age': {'type': 'float64', 'min': 0, 'max': 120},
    'income': {'type': 'float64', 'min': 0},
    'category': {'type': 'object', 'values': ['A', 'B', 'C']}
}

validator = DataValidator(schema)
validation_results = validator.validate_dataset(df)

if not validation_results['is_valid']:
    logger.error(f"Data validation failed: {validation_results['errors']}")
```

### Reproducible Research Practices

```python
import random
import numpy as np
import pandas as pd
from typing import Optional
import os
import json
from datetime import datetime

class ExperimentConfig:
    """Configuration management for reproducible experiments."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = {}
        self.random_seed = None
        self.experiment_id = None
        
        if config_path:
            self.load_config(config_path)
    
    def set_random_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        self.random_seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Set seed for other libraries if available
        try:
            import torch
            torch.manual_seed(seed)
        except ImportError:
            pass
        
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass
    
    def set_experiment_id(self, exp_id: Optional[str] = None) -> str:
        """Set unique experiment identifier."""
        if exp_id is None:
            exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_id = exp_id
        return exp_id
    
    def log_experiment(self, results: dict, model_params: dict, 
                      data_info: dict) -> None:
        """Log experiment details for reproducibility."""
        experiment_log = {
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'random_seed': self.random_seed,
            'config': self.config,
            'model_parameters': model_params,
            'data_info': data_info,
            'results': results,
            'environment': {
                'python_version': os.sys.version,
                'working_directory': os.getcwd(),
                'git_commit': self._get_git_commit()
            }
        }
        
        log_file = f"experiments/{self.experiment_id}_log.json"
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        with open(log_file, 'w') as f:
            json.dump(experiment_log, f, indent=2, default=str)
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True)
            return result.stdout.strip() if result.returncode == 0 else None
        except:
            return None

# Feature engineering with version control
class FeatureEngineering:
    """Versioned feature engineering pipeline."""
    
    def __init__(self, version: str = "v1"):
        self.version = version
        self.transformations = []
    
    def add_transformation(self, name: str, func: callable, 
                          params: dict = None) -> None:
        """Add transformation to pipeline."""
        self.transformations.append({
            'name': name,
            'function': func,
            'parameters': params or {},
            'version': self.version
        })
    
    def apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all transformations to dataframe."""
        result_df = df.copy()
        
        for transformation in self.transformations:
            logger.info(f"Applying {transformation['name']} "
                       f"(v{transformation['version']})")
            
            func = transformation['function']
            params = transformation['parameters']
            result_df = func(result_df, **params)
        
        return result_df
    
    def get_feature_metadata(self) -> dict:
        """Get metadata about feature transformations."""
        return {
            'version': self.version,
            'transformations': [
                {
                    'name': t['name'],
                    'parameters': t['parameters'],
                    'version': t['version']
                }
                for t in self.transformations
            ]
        }

# Example usage
config = ExperimentConfig()
config.set_random_seed(42)
exp_id = config.set_experiment_id("customer_segmentation_v1")

# Feature engineering
fe = FeatureEngineering(version="v1.2")
fe.add_transformation("log_transform", lambda df: df.assign(
    log_income=np.log1p(df['income'])
))
fe.add_transformation("age_binning", lambda df: df.assign(
    age_group=pd.cut(df['age'], bins=[0, 25, 35, 50, 100], 
                    labels=['young', 'adult', 'middle_age', 'senior'])
))
```

## Machine Learning Best Practices

### Model Development Lifecycle

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np

class MLPipeline:
    """Complete ML pipeline with best practices."""
    
    def __init__(self, model, preprocessor=None, config=None):
        self.model = model
        self.preprocessor = preprocessor
        self.config = config or {}
        self.is_fitted = False
        self.training_metadata = {}
        self.validation_results = {}
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              validation_strategy: str = "cross_val") -> Dict[str, Any]:
        """
        Train model with comprehensive validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            validation_strategy: 'cross_val', 'holdout', or 'time_series'
        
        Returns:
            Training results and validation metrics
        """
        # Data validation
        self._validate_input_data(X, y)
        
        # Preprocessing
        if self.preprocessor:
            X_processed = self.preprocessor.fit_transform(X)
        else:
            X_processed = X
        
        # Model training
        self.model.fit(X_processed, y)
        self.is_fitted = True
        
        # Validation
        validation_results = self._validate_model(X_processed, y, validation_strategy)
        
        # Store metadata
        self.training_metadata = {
            'training_samples': len(X),
            'features': list(X.columns),
            'target_distribution': y.value_counts().to_dict(),
            'validation_strategy': validation_strategy
        }
        
        self.validation_results = validation_results
        
        return {
            'training_metadata': self.training_metadata,
            'validation_results': validation_results
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with input validation."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        # Validate input features match training features
        if list(X.columns) != self.training_metadata['features']:
            raise ValueError("Input features don't match training features")
        
        if self.preprocessor:
            X_processed = self.preprocessor.transform(X)
        else:
            X_processed = X
        
        return self.model.predict(X_processed)
    
    def _validate_input_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Validate input data quality."""
        # Check for missing values
        if X.isnull().any().any():
            raise ValueError("Input features contain missing values")
        
        if y.isnull().any():
            raise ValueError("Target variable contains missing values")
        
        # Check data types
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) != len(X.columns):
            non_numeric = set(X.columns) - set(numeric_columns)
            raise ValueError(f"Non-numeric columns found: {non_numeric}")
        
        # Check for infinite values
        if np.isinf(X.select_dtypes(include=[np.number])).any().any():
            raise ValueError("Input features contain infinite values")
    
    def _validate_model(self, X: np.ndarray, y: pd.Series, 
                       strategy: str) -> Dict[str, Any]:
        """Validate model performance."""
        if strategy == "cross_val":
            return self._cross_validation(X, y)
        elif strategy == "holdout":
            return self._holdout_validation(X, y)
        else:
            raise ValueError(f"Unknown validation strategy: {strategy}")
    
    def _cross_validation(self, X: np.ndarray, y: pd.Series, 
                         cv_folds: int = 5) -> Dict[str, Any]:
        """Perform cross-validation."""
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Get cross-validation scores
        cv_scores = cross_val_score(self.model, X, y, cv=cv, 
                                   scoring='accuracy', n_jobs=-1)
        
        return {
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_folds': cv_folds
        }
    
    def save_model(self, filepath: str) -> None:
        """Save model with metadata."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        model_package = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'config': self.config,
            'training_metadata': self.training_metadata,
            'validation_results': self.validation_results,
            'version': '1.0'
        }
        
        joblib.dump(model_package, filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'MLPipeline':
        """Load model with metadata."""
        model_package = joblib.load(filepath)
        
        pipeline = cls(
            model=model_package['model'],
            preprocessor=model_package['preprocessor'],
            config=model_package['config']
        )
        
        pipeline.is_fitted = True
        pipeline.training_metadata = model_package['training_metadata']
        pipeline.validation_results = model_package['validation_results']
        
        return pipeline

# Model monitoring and drift detection
class ModelMonitor:
    """Monitor model performance and detect drift."""
    
    def __init__(self, reference_data: pd.DataFrame):
        self.reference_data = reference_data
        self.reference_stats = self._calculate_stats(reference_data)
    
    def detect_drift(self, new_data: pd.DataFrame, 
                    threshold: float = 0.05) -> Dict[str, Any]:
        """
        Detect data drift using statistical tests.
        
        Args:
            new_data: New data to compare against reference
            threshold: Significance threshold for drift detection
        
        Returns:
            Drift detection results
        """
        new_stats = self._calculate_stats(new_data)
        
        drift_results = {}
        
        for column in self.reference_data.columns:
            if column not in new_data.columns:
                continue
            
            # Kolmogorov-Smirnov test for numerical columns
            if np.issubdtype(self.reference_data[column].dtype, np.number):
                from scipy import stats
                ks_stat, p_value = stats.ks_2samp(
                    self.reference_data[column].dropna(),
                    new_data[column].dropna()
                )
                
                drift_detected = p_value < threshold
                
                drift_results[column] = {
                    'test': 'kolmogorov_smirnov',
                    'statistic': ks_stat,
                    'p_value': p_value,
                    'drift_detected': drift_detected,
                    'threshold': threshold
                }
        
        return drift_results
    
    def _calculate_stats(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Calculate statistical summaries for data."""
        stats = {}
        
        for column in data.columns:
            if np.issubdtype(data[column].dtype, np.number):
                stats[column] = {
                    'mean': data[column].mean(),
                    'std': data[column].std(),
                    'min': data[column].min(),
                    'max': data[column].max(),
                    'median': data[column].median(),
                    'q25': data[column].quantile(0.25),
                    'q75': data[column].quantile(0.75)
                }
            else:
                stats[column] = {
                    'value_counts': data[column].value_counts().to_dict(),
                    'unique_count': data[column].nunique(),
                    'missing_count': data[column].isnull().sum()
                }
        
        return stats
```

## Security Best Practices

### Secure Coding Practices

```python
import hashlib
import secrets
import hmac
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class SecurityUtils:
    """Utility class for common security operations."""
    
    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate cryptographically secure random token."""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def hash_password(password: str, salt: Optional[bytes] = None) -> Dict[str, str]:
        """
        Hash password using PBKDF2 with SHA-256.
        
        Args:
            password: Plain text password
            salt: Optional salt (generated if not provided)
        
        Returns:
            Dictionary with hashed password and salt
        """
        if salt is None:
            salt = secrets.token_bytes(32)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        
        return {
            'hash': key.decode(),
            'salt': base64.urlsafe_b64encode(salt).decode()
        }
    
    @staticmethod
    def verify_password(password: str, stored_hash: str, salt: str) -> bool:
        """Verify password against stored hash."""
        try:
            salt_bytes = base64.urlsafe_b64decode(salt.encode())
            stored_hash_bytes = base64.urlsafe_b64decode(stored_hash.encode())
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt_bytes,
                iterations=100000,
            )
            
            derived_key = kdf.derive(password.encode())
            return hmac.compare_digest(derived_key, stored_hash_bytes)
            
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    @staticmethod
    def encrypt_data(data: str, key: bytes) -> str:
        """Encrypt data using Fernet symmetric encryption."""
        fernet = Fernet(key)
        encrypted_data = fernet.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()
    
    @staticmethod
    def decrypt_data(encrypted_data: str, key: bytes) -> str:
        """Decrypt data using Fernet symmetric encryption."""
        fernet = Fernet(key)
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted_data = fernet.decrypt(encrypted_bytes)
        return decrypted_data.decode()
    
    @staticmethod
    def generate_encryption_key() -> bytes:
        """Generate encryption key for Fernet."""
        return Fernet.generate_key()

# Input validation and sanitization
class InputValidator:
    """Comprehensive input validation."""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def sanitize_html(html_input: str) -> str:
        """Sanitize HTML input to prevent XSS."""
        import html
        return html.escape(html_input)
    
    @staticmethod
    def validate_sql_injection(query_input: str) -> bool:
        """Basic SQL injection detection."""
        dangerous_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b)",
            r"(\b(UNION|JOIN)\b)",
            r"(--|\||#)",
            r"(\b(OR|AND)\b.*=.*)",
        ]
        
        import re
        for pattern in dangerous_patterns:
            if re.search(pattern, query_input, re.IGNORECASE):
                return False
        return True
    
    @staticmethod
    def validate_file_upload(filename: str, allowed_extensions: List[str], 
                           max_size: int) -> Dict[str, Any]:
        """Validate file upload security."""
        import os
        
        results = {
            'is_valid': True,
            'errors': []
        }
        
        # Check file extension
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in allowed_extensions:
            results['is_valid'] = False
            results['errors'].append(f"File extension {file_ext} not allowed")
        
        # Check for path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            results['is_valid'] = False
            results['errors'].append("Invalid characters in filename")
        
        return results

# Rate limiting
from collections import defaultdict
import time

class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = defaultdict(list)
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for given identifier."""
        now = time.time()
        
        # Clean old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if now - req_time < self.time_window
        ]
        
        # Check if under limit
        if len(self.requests[identifier]) < self.max_requests:
            self.requests[identifier].append(now)
            return True
        
        return False

# Secure API endpoint example
from flask import Flask, request, jsonify
import jwt
from functools import wraps

app = Flask(__name__)
rate_limiter = RateLimiter(max_requests=100, time_window=3600)  # 100 requests per hour

def require_auth(f):
    """Decorator for requiring authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        
        try:
            # Remove 'Bearer ' prefix
            token = token.replace('Bearer ', '')
            decoded = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            request.user_id = decoded['user_id']
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(*args, **kwargs)
    
    return decorated

def require_rate_limit(f):
    """Decorator for rate limiting."""
    @wraps(f)
    def decorated(*args, **kwargs):
        client_ip = request.remote_addr
        
        if not rate_limiter.is_allowed(client_ip):
            return jsonify({'error': 'Rate limit exceeded'}), 429
        
        return f(*args, **kwargs)
    
    return decorated

@app.route('/api/secure-endpoint', methods=['POST'])
@require_rate_limit
@require_auth
def secure_endpoint():
    """Example of secure API endpoint."""
    # Validate input
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    # Sanitize inputs
    email = data.get('email', '').strip()
    message = data.get('message', '').strip()
    
    if not InputValidator.validate_email(email):
        return jsonify({'error': 'Invalid email format'}), 400
    
    # Sanitize HTML
    message = InputValidator.sanitize_html(message)
    
    # Process request
    result = process_secure_request(email, message, request.user_id)
    
    return jsonify(result)
```

---

*This comprehensive best practices guide provides industry-standard approaches to software development, data science, machine learning, and security with practical, production-ready examples and methodologies.* 