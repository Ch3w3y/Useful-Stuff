# Productivity Tools and Utilities

## Overview

Comprehensive collection of productivity tools, automation scripts, development utilities, and workflow optimizations for data science and software development teams.

## Table of Contents

- [Development Workflow Tools](#development-workflow-tools)
- [Code Generation](#code-generation)
- [Automation Scripts](#automation-scripts)
- [Environment Management](#environment-management)
- [Documentation Tools](#documentation-tools)
- [Project Templates](#project-templates)
- [Time Tracking](#time-tracking)
- [Best Practices](#best-practices)

## Development Workflow Tools

### Git Workflow Automation

```bash
#!/bin/bash
# Git workflow automation script

# Enhanced git aliases and functions
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.st status
git config --global alias.unstage 'reset HEAD --'
git config --global alias.last 'log -1 HEAD'
git config --global alias.visual '!gitk'
git config --global alias.lgraph 'log --oneline --graph --decorate --all'
git config --global alias.cleanup '!git branch --merged | grep -v "\*\|main\|master\|develop" | xargs -n 1 git branch -d'

# Smart commit function
smart_commit() {
    local message="$1"
    local files="$2"
    
    if [ -z "$message" ]; then
        echo "Usage: smart_commit 'commit message' [files]"
        return 1
    fi
    
    # Add files (all if not specified)
    if [ -z "$files" ]; then
        git add .
    else
        git add $files
    fi
    
    # Show what will be committed
    echo "Files to be committed:"
    git diff --cached --name-only
    
    # Confirm commit
    read -p "Proceed with commit? (y/N): " confirm
    if [[ $confirm == [yY] ]]; then
        git commit -m "$message"
        
        # Ask about push
        read -p "Push to remote? (y/N): " push_confirm
        if [[ $push_confirm == [yY] ]]; then
            git push
        fi
    fi
}

# Feature branch workflow
start_feature() {
    local feature_name="$1"
    if [ -z "$feature_name" ]; then
        echo "Usage: start_feature feature_name"
        return 1
    fi
    
    git checkout main
    git pull origin main
    git checkout -b "feature/$feature_name"
    echo "Started feature branch: feature/$feature_name"
}

finish_feature() {
    local current_branch=$(git branch --show-current)
    
    if [[ $current_branch != feature/* ]]; then
        echo "Not on a feature branch"
        return 1
    fi
    
    echo "Finishing feature branch: $current_branch"
    
    # Rebase on main
    git checkout main
    git pull origin main
    git checkout $current_branch
    git rebase main
    
    # Push feature branch
    git push origin $current_branch
    
    echo "Feature branch ready for PR: $current_branch"
    echo "Create PR at: $(git remote get-url origin | sed 's/\.git$//')/compare/$current_branch"
}

# Release workflow
create_release() {
    local version="$1"
    if [ -z "$version" ]; then
        echo "Usage: create_release version"
        return 1
    fi
    
    git checkout main
    git pull origin main
    git tag -a "v$version" -m "Release version $version"
    git push origin "v$version"
    echo "Created release: v$version"
}
```

### Code Quality Automation

```python
#!/usr/bin/env python3
"""
Code quality automation tools
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

class CodeQualityChecker:
    """Automated code quality checking"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.results = {}
    
    def run_flake8(self) -> Dict[str, Any]:
        """Run flake8 linting"""
        try:
            result = subprocess.run(
                ["flake8", "--max-line-length=88", "--extend-ignore=E203,W503", 
                 str(self.project_root)],
                capture_output=True, text=True
            )
            
            return {
                "tool": "flake8",
                "passed": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr
            }
        except FileNotFoundError:
            return {"tool": "flake8", "passed": False, "error": "flake8 not installed"}
    
    def run_black(self, check_only: bool = True) -> Dict[str, Any]:
        """Run black code formatting"""
        try:
            cmd = ["black"]
            if check_only:
                cmd.append("--check")
            cmd.extend(["--line-length=88", str(self.project_root)])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            return {
                "tool": "black",
                "passed": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr
            }
        except FileNotFoundError:
            return {"tool": "black", "passed": False, "error": "black not installed"}
    
    def run_isort(self, check_only: bool = True) -> Dict[str, Any]:
        """Run isort import sorting"""
        try:
            cmd = ["isort"]
            if check_only:
                cmd.append("--check-only")
            cmd.extend(["--profile=black", str(self.project_root)])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            return {
                "tool": "isort",
                "passed": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr
            }
        except FileNotFoundError:
            return {"tool": "isort", "passed": False, "error": "isort not installed"}
    
    def run_mypy(self) -> Dict[str, Any]:
        """Run mypy type checking"""
        try:
            result = subprocess.run(
                ["mypy", str(self.project_root)],
                capture_output=True, text=True
            )
            
            return {
                "tool": "mypy",
                "passed": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr
            }
        except FileNotFoundError:
            return {"tool": "mypy", "passed": False, "error": "mypy not installed"}
    
    def run_pytest(self) -> Dict[str, Any]:
        """Run pytest tests"""
        try:
            result = subprocess.run(
                ["pytest", "-v", "--tb=short"],
                capture_output=True, text=True, cwd=self.project_root
            )
            
            return {
                "tool": "pytest",
                "passed": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr
            }
        except FileNotFoundError:
            return {"tool": "pytest", "passed": False, "error": "pytest not installed"}
    
    def run_all_checks(self, fix_formatting: bool = False) -> Dict[str, Any]:
        """Run all quality checks"""
        print("Running code quality checks...")
        
        # Linting
        print("• Running flake8...")
        self.results["flake8"] = self.run_flake8()
        
        # Formatting
        print("• Running black...")
        self.results["black"] = self.run_black(check_only=not fix_formatting)
        
        print("• Running isort...")
        self.results["isort"] = self.run_isort(check_only=not fix_formatting)
        
        # Type checking
        print("• Running mypy...")
        self.results["mypy"] = self.run_mypy()
        
        # Tests
        print("• Running pytest...")
        self.results["pytest"] = self.run_pytest()
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate quality report"""
        if not self.results:
            return "No quality checks have been run."
        
        report = ["=" * 50, "CODE QUALITY REPORT", "=" * 50, ""]
        
        total_checks = len(self.results)
        passed_checks = sum(1 for r in self.results.values() if r.get("passed", False))
        
        report.append(f"Overall: {passed_checks}/{total_checks} checks passed")
        report.append("")
        
        for tool, result in self.results.items():
            status = "✓ PASS" if result.get("passed", False) else "✗ FAIL"
            report.append(f"{tool.upper()}: {status}")
            
            if not result.get("passed", False):
                if "error" in result:
                    report.append(f"  Error: {result['error']}")
                elif result.get("output"):
                    lines = result["output"].strip().split('\n')[:5]  # First 5 lines
                    for line in lines:
                        report.append(f"  {line}")
                    if len(result["output"].strip().split('\n')) > 5:
                        report.append("  ...")
            report.append("")
        
        return "\n".join(report)

# Pre-commit hook script
def create_pre_commit_hook():
    """Create pre-commit hook script"""
    hook_content = '''#!/bin/bash
# Pre-commit hook for code quality

echo "Running pre-commit quality checks..."

# Run quality checks
python -c "
from code_quality import CodeQualityChecker
checker = CodeQualityChecker()
results = checker.run_all_checks()
report = checker.generate_report()
print(report)

# Check if all passed
all_passed = all(r.get('passed', False) for r in results.values())
exit(0 if all_passed else 1)
"

if [ $? -ne 0 ]; then
    echo "Pre-commit checks failed. Please fix issues and try again."
    exit 1
fi

echo "All pre-commit checks passed!"
'''
    
    hook_path = Path(".git/hooks/pre-commit")
    hook_path.parent.mkdir(exist_ok=True)
    hook_path.write_text(hook_content)
    hook_path.chmod(0o755)
    print(f"Created pre-commit hook: {hook_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Code quality checker")
    parser.add_argument("--fix", action="store_true", help="Fix formatting issues")
    parser.add_argument("--install-hook", action="store_true", help="Install pre-commit hook")
    
    args = parser.parse_args()
    
    if args.install_hook:
        create_pre_commit_hook()
        sys.exit(0)
    
    checker = CodeQualityChecker()
    results = checker.run_all_checks(fix_formatting=args.fix)
    report = checker.generate_report()
    print(report)
    
    # Exit with error if any checks failed
    all_passed = all(r.get("passed", False) for r in results.values())
    sys.exit(0 if all_passed else 1)
```

## Code Generation

### Project Template Generator

```python
"""
Project template generator for different types of projects
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Any
from jinja2 import Environment, FileSystemLoader
import yaml

class ProjectGenerator:
    """Generate project templates"""
    
    def __init__(self, templates_dir: str = "templates"):
        self.templates_dir = Path(templates_dir)
        self.env = Environment(loader=FileSystemLoader(str(self.templates_dir)))
    
    def generate_python_package(self, package_name: str, author: str, 
                               description: str, python_version: str = "3.8") -> Path:
        """Generate Python package structure"""
        project_dir = Path(package_name)
        project_dir.mkdir(exist_ok=True)
        
        # Package directory
        package_dir = project_dir / package_name
        package_dir.mkdir(exist_ok=True)
        
        # Create files from templates
        context = {
            "package_name": package_name,
            "author": author,
            "description": description,
            "python_version": python_version
        }
        
        files_to_create = {
            "setup.py": "python_package/setup.py.j2",
            "README.md": "python_package/README.md.j2",
            "requirements.txt": "python_package/requirements.txt.j2",
            "requirements-dev.txt": "python_package/requirements-dev.txt.j2",
            ".gitignore": "python_package/gitignore.j2",
            "pyproject.toml": "python_package/pyproject.toml.j2",
            f"{package_name}/__init__.py": "python_package/__init__.py.j2",
            f"{package_name}/main.py": "python_package/main.py.j2",
            "tests/__init__.py": "python_package/test_init.py.j2",
            f"tests/test_{package_name}.py": "python_package/test_package.py.j2"
        }
        
        for file_path, template_path in files_to_create.items():
            self._create_file_from_template(
                project_dir / file_path, template_path, context
            )
        
        # Create additional directories
        (project_dir / "docs").mkdir(exist_ok=True)
        (project_dir / "scripts").mkdir(exist_ok=True)
        
        return project_dir
    
    def generate_data_science_project(self, project_name: str, author: str,
                                    description: str) -> Path:
        """Generate data science project structure"""
        project_dir = Path(project_name)
        project_dir.mkdir(exist_ok=True)
        
        # Create directory structure
        directories = [
            "data/raw",
            "data/processed",
            "data/interim",
            "data/external",
            "notebooks",
            "src",
            "models",
            "reports/figures",
            "scripts",
            "tests",
            "configs"
        ]
        
        for directory in directories:
            (project_dir / directory).mkdir(parents=True, exist_ok=True)
        
        context = {
            "project_name": project_name,
            "author": author,
            "description": description
        }
        
        files_to_create = {
            "README.md": "data_science/README.md.j2",
            "requirements.txt": "data_science/requirements.txt.j2",
            ".gitignore": "data_science/gitignore.j2",
            "Makefile": "data_science/Makefile.j2",
            "src/__init__.py": "common/__init__.py.j2",
            "src/data/__init__.py": "common/__init__.py.j2",
            "src/data/make_dataset.py": "data_science/make_dataset.py.j2",
            "src/features/__init__.py": "common/__init__.py.j2",
            "src/features/build_features.py": "data_science/build_features.py.j2",
            "src/models/__init__.py": "common/__init__.py.j2",
            "src/models/train_model.py": "data_science/train_model.py.j2",
            "src/models/predict_model.py": "data_science/predict_model.py.j2",
            "src/visualization/__init__.py": "common/__init__.py.j2",
            "src/visualization/visualize.py": "data_science/visualize.py.j2",
            "notebooks/01-exploratory-data-analysis.ipynb": "data_science/eda_notebook.j2",
            "configs/config.yaml": "data_science/config.yaml.j2"
        }
        
        for file_path, template_path in files_to_create.items():
            self._create_file_from_template(
                project_dir / file_path, template_path, context
            )
        
        return project_dir
    
    def generate_ml_pipeline(self, project_name: str, author: str,
                           ml_type: str = "classification") -> Path:
        """Generate ML pipeline project"""
        project_dir = Path(project_name)
        project_dir.mkdir(exist_ok=True)
        
        directories = [
            "src/data",
            "src/features",
            "src/models",
            "src/evaluation",
            "src/deployment",
            "configs",
            "tests",
            "models",
            "data",
            "notebooks",
            "scripts",
            "docker"
        ]
        
        for directory in directories:
            (project_dir / directory).mkdir(parents=True, exist_ok=True)
        
        context = {
            "project_name": project_name,
            "author": author,
            "ml_type": ml_type
        }
        
        files_to_create = {
            "README.md": "ml_pipeline/README.md.j2",
            "requirements.txt": "ml_pipeline/requirements.txt.j2",
            "setup.py": "ml_pipeline/setup.py.j2",
            "Dockerfile": "ml_pipeline/Dockerfile.j2",
            "docker-compose.yml": "ml_pipeline/docker-compose.yml.j2",
            ".gitignore": "ml_pipeline/gitignore.j2",
            "src/__init__.py": "common/__init__.py.j2",
            "src/config.py": "ml_pipeline/config.py.j2",
            "src/data/data_loader.py": "ml_pipeline/data_loader.py.j2",
            "src/features/feature_engineering.py": "ml_pipeline/feature_engineering.py.j2",
            "src/models/model.py": f"ml_pipeline/model_{ml_type}.py.j2",
            "src/evaluation/metrics.py": "ml_pipeline/metrics.py.j2",
            "src/deployment/api.py": "ml_pipeline/api.py.j2",
            "scripts/train.py": "ml_pipeline/train.py.j2",
            "scripts/predict.py": "ml_pipeline/predict.py.j2",
            "configs/model_config.yaml": "ml_pipeline/model_config.yaml.j2"
        }
        
        for file_path, template_path in files_to_create.items():
            self._create_file_from_template(
                project_dir / file_path, template_path, context
            )
        
        return project_dir
    
    def _create_file_from_template(self, file_path: Path, template_path: str, 
                                 context: Dict[str, Any]):
        """Create file from Jinja2 template"""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            template = self.env.get_template(template_path)
            content = template.render(**context)
            file_path.write_text(content)
        except Exception as e:
            # Create basic file if template doesn't exist
            if not file_path.exists():
                file_path.touch()

# Code snippet generator
class CodeSnippetGenerator:
    """Generate common code snippets"""
    
    @staticmethod
    def generate_data_loader(framework: str = "pandas") -> str:
        """Generate data loader code"""
        if framework == "pandas":
            return '''
import pandas as pd
from pathlib import Path
from typing import Optional

def load_data(file_path: str, **kwargs) -> pd.DataFrame:
    """Load data from various file formats"""
    file_path = Path(file_path)
    
    if file_path.suffix == '.csv':
        return pd.read_csv(file_path, **kwargs)
    elif file_path.suffix == '.json':
        return pd.read_json(file_path, **kwargs)
    elif file_path.suffix in ['.xlsx', '.xls']:
        return pd.read_excel(file_path, **kwargs)
    elif file_path.suffix == '.parquet':
        return pd.read_parquet(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

def save_data(df: pd.DataFrame, file_path: str, **kwargs):
    """Save data to various file formats"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if file_path.suffix == '.csv':
        df.to_csv(file_path, index=False, **kwargs)
    elif file_path.suffix == '.json':
        df.to_json(file_path, **kwargs)
    elif file_path.suffix in ['.xlsx', '.xls']:
        df.to_excel(file_path, index=False, **kwargs)
    elif file_path.suffix == '.parquet':
        df.to_parquet(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
'''
        
        elif framework == "pytorch":
            return '''
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Any

class CustomDataset(Dataset):
    """Custom PyTorch dataset"""
    
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        sample = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, target

def create_data_loader(dataset: Dataset, batch_size: int = 32, 
                      shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    """Create PyTorch data loader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
'''
    
    @staticmethod
    def generate_model_template(model_type: str = "sklearn") -> str:
        """Generate model template"""
        if model_type == "sklearn":
            return '''
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib
from typing import Any, Tuple
import pandas as pd
import numpy as np

class ModelPipeline:
    """ML model pipeline"""
    
    def __init__(self, model, preprocessor=None):
        self.model = model
        self.preprocessor = preprocessor
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ModelPipeline':
        """Fit the pipeline"""
        if self.preprocessor:
            X_processed = self.preprocessor.fit_transform(X)
        else:
            X_processed = X
        
        self.model.fit(X_processed, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.preprocessor:
            X_processed = self.preprocessor.transform(X)
        else:
            X_processed = X
        
        return self.model.predict(X_processed)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Evaluate model performance"""
        predictions = self.predict(X)
        
        return {
            'accuracy': accuracy_score(y, predictions),
            'classification_report': classification_report(y, predictions)
        }
    
    def save(self, filepath: str):
        """Save the pipeline"""
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'ModelPipeline':
        """Load the pipeline"""
        return joblib.load(filepath)
'''
    
    @staticmethod
    def generate_config_template() -> str:
        """Generate configuration template"""
        return '''
import yaml
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, asdict

@dataclass
class Config:
    """Configuration class"""
    # Data paths
    data_dir: str = "data"
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    
    # Model parameters
    model_type: str = "random_forest"
    test_size: float = 0.2
    random_state: int = 42
    
    # Training parameters
    n_estimators: int = 100
    max_depth: int = None
    
    # Output paths
    model_path: str = "models/model.joblib"
    results_path: str = "results"
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """Load config from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, config_path: str):
        """Save config to YAML file"""
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
    
    def update(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
'''

# CLI for code generation
def main():
    """CLI for project and code generation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Project and code generator")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Project generation
    project_parser = subparsers.add_parser("project", help="Generate project template")
    project_parser.add_argument("type", choices=["python", "datascience", "ml"], 
                               help="Project type")
    project_parser.add_argument("name", help="Project name")
    project_parser.add_argument("--author", default="Your Name", help="Author name")
    project_parser.add_argument("--description", default="", help="Project description")
    
    # Code generation
    code_parser = subparsers.add_parser("code", help="Generate code snippets")
    code_parser.add_argument("type", choices=["dataloader", "model", "config"],
                            help="Code type")
    code_parser.add_argument("--framework", default="pandas", 
                            choices=["pandas", "pytorch", "sklearn"],
                            help="Framework for code generation")
    
    args = parser.parse_args()
    
    if args.command == "project":
        generator = ProjectGenerator()
        
        if args.type == "python":
            project_dir = generator.generate_python_package(
                args.name, args.author, args.description
            )
        elif args.type == "datascience":
            project_dir = generator.generate_data_science_project(
                args.name, args.author, args.description
            )
        elif args.type == "ml":
            project_dir = generator.generate_ml_pipeline(
                args.name, args.author
            )
        
        print(f"Created project: {project_dir}")
    
    elif args.command == "code":
        snippet_gen = CodeSnippetGenerator()
        
        if args.type == "dataloader":
            code = snippet_gen.generate_data_loader(args.framework)
        elif args.type == "model":
            code = snippet_gen.generate_model_template(args.framework)
        elif args.type == "config":
            code = snippet_gen.generate_config_template()
        
        print(code)

if __name__ == "__main__":
    main()
```

## Automation Scripts

### Development Environment Setup

```bash
#!/bin/bash
# Development environment setup script

setup_python_env() {
    local env_name="$1"
    local python_version="$2"
    
    echo "Setting up Python environment: $env_name"
    
    # Create conda environment
    conda create -n "$env_name" python="$python_version" -y
    conda activate "$env_name"
    
    # Install common packages
    pip install -r requirements-dev.txt
    
    # Install pre-commit hooks
    pre-commit install
    
    echo "Python environment '$env_name' created successfully"
}

setup_git_hooks() {
    echo "Setting up Git hooks..."
    
    # Install pre-commit
    pip install pre-commit
    
    # Create .pre-commit-config.yaml
    cat > .pre-commit-config.yaml << EOF
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
  
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
        language_version: python3
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]
  
  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203,W503]
EOF
    
    # Install hooks
    pre-commit install
    
    echo "Git hooks installed successfully"
}

setup_docker_env() {
    echo "Setting up Docker environment..."
    
    # Create Dockerfile
    cat > Dockerfile << EOF
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
EOF
    
    # Create docker-compose.yml
    cat > docker-compose.yml << EOF
version: '3.8'
services:
  app:
    build: .
    volumes:
      - .:/app
    environment:
      - PYTHONPATH=/app
    ports:
      - "8000:8000"
EOF
    
    echo "Docker environment setup complete"
}

setup_vscode_config() {
    echo "Setting up VS Code configuration..."
    
    mkdir -p .vscode
    
    # Settings
    cat > .vscode/settings.json << EOF
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.provider": "isort",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
EOF
    
    # Extensions
    cat > .vscode/extensions.json << EOF
{
    "recommendations": [
        "ms-python.python",
        "ms-python.flake8",
        "ms-python.black-formatter",
        "ms-python.isort",
        "ms-toolsai.jupyter",
        "GitHub.copilot",
        "eamodio.gitlens"
    ]
}
EOF
    
    echo "VS Code configuration created"
}

# Main setup function
main() {
    echo "Development Environment Setup"
    echo "============================="
    
    read -p "Enter environment name: " env_name
    read -p "Enter Python version (default: 3.9): " python_version
    python_version=${python_version:-3.9}
    
    # Setup steps
    setup_python_env "$env_name" "$python_version"
    setup_git_hooks
    setup_docker_env
    setup_vscode_config
    
    echo ""
    echo "Setup complete! Next steps:"
    echo "1. Activate environment: conda activate $env_name"
    echo "2. Install dependencies: pip install -r requirements.txt"
    echo "3. Run tests: pytest"
    echo "4. Start development!"
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
```

### Data Pipeline Automation

```python
#!/usr/bin/env python3
"""
Data pipeline automation and scheduling
"""

import schedule
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Any
import yaml
import subprocess
import smtplib
from email.mime.text import MIMEText

class PipelineScheduler:
    """Automated pipeline scheduler"""
    
    def __init__(self, config_path: str = "pipeline_config.yaml"):
        self.config = self._load_config(config_path)
        self.setup_logging()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load pipeline configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(self.config.get('logging', {}).get('log_dir', 'logs'))
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'pipeline.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_command(self, command: str, name: str = None) -> bool:
        """Run shell command"""
        self.logger.info(f"Running {name or 'command'}: {command}")
        
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True
            )
            
            if result.returncode == 0:
                self.logger.info(f"Command completed successfully: {name}")
                return True
            else:
                self.logger.error(f"Command failed: {name}")
                self.logger.error(f"Error: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Exception running command {name}: {e}")
            return False
    
    def run_python_script(self, script_path: str, args: str = "") -> bool:
        """Run Python script"""
        command = f"python {script_path} {args}"
        return self.run_command(command, f"Python script: {script_path}")
    
    def send_notification(self, subject: str, message: str):
        """Send email notification"""
        email_config = self.config.get('notifications', {}).get('email', {})
        
        if not email_config.get('enabled', False):
            return
        
        try:
            msg = MIMEText(message)
            msg['Subject'] = subject
            msg['From'] = email_config['from']
            msg['To'] = ', '.join(email_config['to'])
            
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Notification sent: {subject}")
            
        except Exception as e:
            self.logger.error(f"Failed to send notification: {e}")
    
    def run_pipeline(self, pipeline_name: str):
        """Run complete pipeline"""
        pipeline_config = self.config['pipelines'][pipeline_name]
        self.logger.info(f"Starting pipeline: {pipeline_name}")
        
        start_time = datetime.now()
        success = True
        
        for step in pipeline_config['steps']:
            step_name = step['name']
            step_type = step['type']
            
            self.logger.info(f"Running step: {step_name}")
            
            if step_type == 'command':
                success = self.run_command(step['command'], step_name)
            elif step_type == 'python':
                success = self.run_python_script(
                    step['script'], step.get('args', '')
                )
            
            if not success and step.get('critical', True):
                self.logger.error(f"Critical step failed: {step_name}")
                break
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        if success:
            message = f"Pipeline '{pipeline_name}' completed successfully in {duration}"
            self.logger.info(message)
            self.send_notification(f"Pipeline Success: {pipeline_name}", message)
        else:
            message = f"Pipeline '{pipeline_name}' failed after {duration}"
            self.logger.error(message)
            self.send_notification(f"Pipeline Failure: {pipeline_name}", message)
    
    def schedule_pipelines(self):
        """Schedule all pipelines based on configuration"""
        for pipeline_name, config in self.config['pipelines'].items():
            schedule_config = config.get('schedule')
            
            if not schedule_config:
                continue
            
            schedule_type = schedule_config['type']
            
            if schedule_type == 'daily':
                time_str = schedule_config['time']
                schedule.every().day.at(time_str).do(
                    self.run_pipeline, pipeline_name
                )
                self.logger.info(f"Scheduled {pipeline_name} daily at {time_str}")
                
            elif schedule_type == 'hourly':
                schedule.every().hour.do(self.run_pipeline, pipeline_name)
                self.logger.info(f"Scheduled {pipeline_name} hourly")
                
            elif schedule_type == 'weekly':
                day = schedule_config['day']
                time_str = schedule_config['time']
                getattr(schedule.every(), day.lower()).at(time_str).do(
                    self.run_pipeline, pipeline_name
                )
                self.logger.info(f"Scheduled {pipeline_name} weekly on {day} at {time_str}")
    
    def run_scheduler(self):
        """Run the scheduler"""
        self.logger.info("Starting pipeline scheduler")
        self.schedule_pipelines()
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

# Configuration file template
def create_config_template():
    """Create pipeline configuration template"""
    config = {
        'logging': {
            'log_dir': 'logs',
            'level': 'INFO'
        },
        'notifications': {
            'email': {
                'enabled': False,
                'smtp_server': 'smtp.example.com',
                'smtp_port': 587,
                'username': 'your_email@example.com',
                'password': 'your_password',
                'from': 'pipeline@example.com',
                'to': ['admin@example.com']
            }
        },
        'pipelines': {
            'daily_etl': {
                'description': 'Daily ETL pipeline',
                'schedule': {
                    'type': 'daily',
                    'time': '02:00'
                },
                'steps': [
                    {
                        'name': 'extract_data',
                        'type': 'python',
                        'script': 'scripts/extract.py',
                        'critical': True
                    },
                    {
                        'name': 'transform_data',
                        'type': 'python',
                        'script': 'scripts/transform.py',
                        'critical': True
                    },
                    {
                        'name': 'load_data',
                        'type': 'python',
                        'script': 'scripts/load.py',
                        'critical': True
                    }
                ]
            },
            'model_training': {
                'description': 'Weekly model training',
                'schedule': {
                    'type': 'weekly',
                    'day': 'Sunday',
                    'time': '00:00'
                },
                'steps': [
                    {
                        'name': 'prepare_training_data',
                        'type': 'python',
                        'script': 'scripts/prepare_data.py',
                        'critical': True
                    },
                    {
                        'name': 'train_model',
                        'type': 'python',
                        'script': 'scripts/train.py',
                        'critical': True
                    },
                    {
                        'name': 'evaluate_model',
                        'type': 'python',
                        'script': 'scripts/evaluate.py',
                        'critical': False
                    }
                ]
            }
        }
    }
    
    with open('pipeline_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("Created pipeline_config.yaml template")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline scheduler")
    parser.add_argument("--create-config", action="store_true",
                       help="Create configuration template")
    parser.add_argument("--run", help="Run specific pipeline")
    parser.add_argument("--schedule", action="store_true",
                       help="Start scheduler")
    
    args = parser.parse_args()
    
    if args.create_config:
        create_config_template()
    elif args.run:
        scheduler = PipelineScheduler()
        scheduler.run_pipeline(args.run)
    elif args.schedule:
        scheduler = PipelineScheduler()
        scheduler.run_scheduler()
    else:
        parser.print_help()
```

## Quick Setup Scripts

### One-liner setups for common scenarios

```bash
# Complete data science environment setup
curl -sSL https://raw.githubusercontent.com/your-repo/productivity-tools/main/setup-ds.sh | bash

# Quick Python project setup
create_python_project() {
    local name="$1"
    mkdir "$name" && cd "$name"
    python -m venv venv
    source venv/bin/activate
    pip install black flake8 pytest
    echo "# $name" > README.md
    echo "venv/\n__pycache__/\n*.pyc" > .gitignore
    git init && git add . && git commit -m "Initial commit"
    echo "Project $name created successfully!"
}

# Quick Jupyter lab setup with extensions
setup_jupyter() {
    pip install jupyterlab
    jupyter labextension install @jupyter-widgets/jupyterlab-manager
    jupyter labextension install plotlywidget
    jupyter lab --generate-config
    echo "Jupyter Lab setup complete!"
}
```

---

*This productivity toolkit provides comprehensive automation and utilities to streamline your development workflow and boost team productivity.* 