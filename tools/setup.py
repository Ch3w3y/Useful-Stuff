#!/usr/bin/env python3
"""
Data Science Toolkit Setup Script
=================================

This script helps you set up your environment for data science work by:
- Checking system requirements
- Installing necessary packages
- Setting up configurations
- Downloading common datasets
- Initializing project structure

Usage:
    python setup.py [--option]

Options:
    --minimal     Install only core packages
    --full        Install all packages (default)
    --gpu         Include GPU support packages
    --r           Set up R environment
    --check       Only check system requirements
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path
import json

class DataScienceSetup:
    def __init__(self):
        self.system = platform.system()
        self.python_version = sys.version_info
        self.project_root = Path(__file__).parent.parent
        
    def check_requirements(self):
        """Check system requirements"""
        print("🔍 Checking system requirements...")
        
        # Python version check
        if self.python_version < (3, 8):
            print("❌ Python 3.8+ is required. Current version:", 
                  f"{self.python_version.major}.{self.python_version.minor}")
            return False
        else:
            print(f"✅ Python {self.python_version.major}.{self.python_version.minor} detected")
        
        # Check for pip
        try:
            subprocess.run(["pip", "--version"], check=True, capture_output=True)
            print("✅ pip is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ pip is not available")
            return False
        
        # Check for git
        try:
            subprocess.run(["git", "--version"], check=True, capture_output=True)
            print("✅ git is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  git is not available (optional but recommended)")
        
        # Check available memory
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            print(f"💾 Available RAM: {memory_gb:.1f} GB")
            if memory_gb < 4:
                print("⚠️  Less than 4GB RAM detected. Some operations may be slow.")
        except ImportError:
            print("📊 Could not check memory (psutil not installed)")
        
        return True
    
    def install_packages(self, install_type="full"):
        """Install Python packages"""
        print(f"📦 Installing packages ({install_type} installation)...")
        
        # Define package groups
        core_packages = [
            "pandas", "numpy", "scipy", "scikit-learn",
            "matplotlib", "seaborn", "plotly", "jupyter",
            "tqdm", "requests"
        ]
        
        ml_packages = [
            "xgboost", "lightgbm", "optuna", "mlflow"
        ]
        
        deep_learning_packages = [
            "torch", "torchvision", "transformers", "datasets"
        ]
        
        gpu_packages = [
            "torch[gpu]", "tensorflow[gpu]"
        ]
        
        all_packages = core_packages.copy()
        
        if install_type in ["full", "ml"]:
            all_packages.extend(ml_packages)
        
        if install_type in ["full", "deep"]:
            all_packages.extend(deep_learning_packages)
        
        if install_type == "gpu":
            all_packages.extend(gpu_packages)
        
        # Install packages
        try:
            print("Installing packages...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "--upgrade", "pip"
            ], check=True)
            
            subprocess.run([
                sys.executable, "-m", "pip", "install"
            ] + all_packages, check=True)
            
            print("✅ Packages installed successfully!")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing packages: {e}")
            return False
        
        return True
    
    def setup_r_environment(self):
        """Set up R environment"""
        print("🔧 Setting up R environment...")
        
        # Check if R is available
        try:
            subprocess.run(["Rscript", "--version"], check=True, capture_output=True)
            print("✅ R is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ R is not installed. Please install R first.")
            return False
        
        # Install R packages
        r_packages = [
            "tidyverse", "caret", "randomForest", "xgboost",
            "ggplot2", "dplyr", "readr", "plotly", "shiny"
        ]
        
        r_script = f"""
        install.packages(c({', '.join([f'"{pkg}"' for pkg in r_packages])}), 
                        repos='https://cran.rstudio.com/')
        """
        
        try:
            subprocess.run([
                "Rscript", "-e", r_script
            ], check=True)
            print("✅ R packages installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Error installing R packages")
            return False
        
        return True
    
    def create_project_structure(self):
        """Create recommended project structure"""
        print("📁 Creating project structure...")
        
        project_dirs = [
            "data/raw",
            "data/processed",
            "data/external",
            "notebooks/exploratory",
            "notebooks/experiments",
            "src/data",
            "src/features",
            "src/models",
            "src/visualization",
            "models",
            "reports",
            "configs"
        ]
        
        for dir_path in project_dirs:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            
            # Create .gitkeep files for empty directories
            gitkeep_path = full_path / ".gitkeep"
            if not any(full_path.iterdir()):
                gitkeep_path.touch()
        
        print("✅ Project structure created!")
    
    def setup_jupyter_config(self):
        """Set up Jupyter configuration"""
        print("📓 Setting up Jupyter configuration...")
        
        try:
            # Create Jupyter config directory
            jupyter_dir = Path.home() / ".jupyter"
            jupyter_dir.mkdir(exist_ok=True)
            
            # Basic Jupyter config
            config_content = """
# Jupyter Lab Configuration
c.ServerApp.open_browser = False
c.ServerApp.port = 8888
c.ServerApp.allow_remote_access = True
c.ServerApp.ip = '0.0.0.0'

# Enable extensions
c.ServerApp.jpserver_extensions = {
    'jupyterlab': True
}
"""
            
            config_file = jupyter_dir / "jupyter_lab_config.py"
            with open(config_file, 'w') as f:
                f.write(config_content)
            
            print("✅ Jupyter configuration created!")
            
        except Exception as e:
            print(f"⚠️  Could not create Jupyter config: {e}")
    
    def download_sample_data(self):
        """Download sample datasets for practice"""
        print("📊 Downloading sample datasets...")
        
        try:
            import pandas as pd
            import requests
            
            data_dir = self.project_root / "data" / "sample"
            data_dir.mkdir(exist_ok=True)
            
            # Download Titanic dataset
            titanic_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
            response = requests.get(titanic_url)
            
            if response.status_code == 200:
                with open(data_dir / "titanic.csv", 'wb') as f:
                    f.write(response.content)
                print("✅ Downloaded Titanic dataset")
            
            # Create sample synthetic dataset
            import numpy as np
            np.random.seed(42)
            
            synthetic_data = pd.DataFrame({
                'feature_1': np.random.normal(0, 1, 1000),
                'feature_2': np.random.exponential(2, 1000),
                'feature_3': np.random.choice(['A', 'B', 'C'], 1000),
                'target': np.random.randint(0, 2, 1000)
            })
            
            synthetic_data.to_csv(data_dir / "synthetic_data.csv", index=False)
            print("✅ Created synthetic dataset")
            
        except Exception as e:
            print(f"⚠️  Could not download sample data: {e}")
    
    def create_config_files(self):
        """Create configuration files"""
        print("⚙️  Creating configuration files...")
        
        # Create .gitignore
        gitignore_content = """
# Data Science Project .gitignore

# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Data files
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Models
models/*.pkl
models/*.joblib
models/*.h5

# Jupyter Notebook checkpoints
.ipynb_checkpoints

# Environment files
.env
.venv
env/
venv/

# IDE files
.vscode/
.idea/
*.swp
*.swo

# OS files
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Temporary files
tmp/
temp/
"""
        
        gitignore_path = self.project_root / ".gitignore"
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
        
        # Create environment template
        env_content = """
# Environment Variables for Data Science Project
# Copy this file to .env and fill in your values

# API Keys
OPENAI_API_KEY=your_openai_key_here
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key

# Database connections
DATABASE_URL=your_database_url
POSTGRES_USER=your_postgres_user
POSTGRES_PASSWORD=your_postgres_password

# Cloud credentials
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key

# MLflow tracking
MLFLOW_TRACKING_URI=your_mlflow_server_url
"""
        
        env_template_path = self.project_root / ".env.template"
        with open(env_template_path, 'w') as f:
            f.write(env_content)
        
        print("✅ Configuration files created!")
    
    def run_setup(self, args):
        """Run the complete setup process"""
        print("🚀 Starting Data Science Toolkit Setup")
        print("=" * 50)
        
        # Check requirements
        if not self.check_requirements():
            print("❌ System requirements not met. Please fix issues and try again.")
            return False
        
        if args.check:
            print("✅ System check complete!")
            return True
        
        # Install packages
        install_type = "minimal" if args.minimal else "full"
        if args.gpu:
            install_type = "gpu"
        
        if not self.install_packages(install_type):
            print("❌ Package installation failed")
            return False
        
        # Set up R environment
        if args.r:
            self.setup_r_environment()
        
        # Create project structure
        self.create_project_structure()
        
        # Set up configurations
        self.setup_jupyter_config()
        self.create_config_files()
        
        # Download sample data
        self.download_sample_data()
        
        print("\n🎉 Setup complete!")
        print("\nNext steps:")
        print("1. Activate your environment: source venv/bin/activate")
        print("2. Start Jupyter Lab: jupyter lab")
        print("3. Check out the sample notebooks in notebooks/")
        print("4. Read the documentation in wiki/")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Data Science Toolkit Setup")
    parser.add_argument("--minimal", action="store_true", 
                       help="Install only core packages")
    parser.add_argument("--full", action="store_true", 
                       help="Install all packages (default)")
    parser.add_argument("--gpu", action="store_true", 
                       help="Include GPU support packages")
    parser.add_argument("--r", action="store_true", 
                       help="Set up R environment")
    parser.add_argument("--check", action="store_true", 
                       help="Only check system requirements")
    
    args = parser.parse_args()
    
    setup = DataScienceSetup()
    success = setup.run_setup(args)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 