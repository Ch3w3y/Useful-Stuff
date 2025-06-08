# PyTorch GPU Environment for Deep Learning
# =========================================
# This Docker image provides a complete environment for PyTorch-based ML projects
# with GPU support, common libraries, and Jupyter notebook setup.

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

LABEL maintainer="Data Science Team"
LABEL description="PyTorch GPU environment with ML libraries and Jupyter"
LABEL version="1.0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV JUPYTER_ENABLE_LAB=yes

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    tree \
    unzip \
    build-essential \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install additional ML libraries
RUN pip install --no-cache-dir \
    # Core ML libraries
    scikit-learn==1.3.2 \
    xgboost==2.0.2 \
    lightgbm==4.1.0 \
    catboost==1.2.2 \
    # Deep Learning
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    transformers==4.35.2 \
    datasets==2.14.6 \
    accelerate==0.24.1 \
    # Computer Vision
    opencv-python==4.8.1.78 \
    pillow==10.1.0 \
    albumentations==1.3.1 \
    # NLP
    spacy==3.7.2 \
    nltk==3.8.1 \
    gensim==4.3.2 \
    # Visualization
    matplotlib==3.8.2 \
    seaborn==0.13.0 \
    plotly==5.17.0 \
    bokeh==3.3.2 \
    # Data handling
    pandas==2.1.4 \
    numpy==1.25.2 \
    polars==0.19.19 \
    # Jupyter and extensions
    jupyterlab==4.0.9 \
    jupyterlab-git==0.50.0 \
    jupyter-widgets==8.1.1 \
    ipywidgets==8.1.1 \
    # MLOps
    mlflow==2.8.1 \
    wandb==0.16.0 \
    tensorboard==2.15.1 \
    # Utilities
    tqdm==4.66.1 \
    click==8.1.7 \
    pyyaml==6.0.1 \
    python-dotenv==1.0.0

# Install spaCy models
RUN python -m spacy download en_core_web_sm

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Create directories for common ML tasks
RUN mkdir -p /workspace/{data,notebooks,models,experiments,configs,scripts}

# Copy configuration files
COPY jupyter_lab_config.py /root/.jupyter/
COPY .bashrc /root/

# Set up Jupyter Lab extensions
RUN jupyter lab build

# Create a sample notebook for testing
COPY sample_pytorch_notebook.ipynb /workspace/notebooks/

# Expose Jupyter port
EXPOSE 8888

# Create startup script
RUN echo '#!/bin/bash\n\
echo "Starting PyTorch GPU Environment..."\n\
echo "GPU Status:"\n\
nvidia-smi\n\
echo ""\n\
echo "Python packages:"\n\
pip list | grep -E "(torch|tensorflow|sklearn|pandas|numpy)"\n\
echo ""\n\
echo "Starting Jupyter Lab on port 8888..."\n\
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir=/workspace\n\
' > /start.sh && chmod +x /start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8888/lab || exit 1

# Default command
CMD ["/start.sh"] 