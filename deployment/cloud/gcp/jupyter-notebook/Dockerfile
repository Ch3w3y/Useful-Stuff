# Jupyter Notebook for GCP
# Secure Jupyter deployment with data science stack and GCP integration
FROM jupyter/datascience-notebook:latest

# Switch to root for installations
USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update -y && apt-get install google-cloud-sdk -y && \
    rm -rf /var/lib/apt/lists/*

# Switch back to jovyan user
USER $NB_UID

# Install Python packages for GCP and data science
RUN pip install --no-cache-dir \
    google-cloud-storage \
    google-cloud-bigquery \
    google-cloud-bigquery-storage \
    google-cloud-aiplatform \
    google-cloud-monitoring \
    google-cloud-logging \
    google-auth \
    google-auth-oauthlib \
    pandas-gbq \
    pyarrow \
    fastparquet \
    dask[complete] \
    ray[default] \
    mlflow \
    wandb \
    optuna \
    shap \
    lime \
    plotly \
    bokeh \
    altair \
    streamlit \
    gradio \
    jupyter-ai \
    jupyterlab-git \
    jupyterlab-lsp \
    python-lsp-server[all] \
    black \
    isort \
    flake8 \
    mypy

# Install R packages for Jupyter
RUN conda install -c r -c conda-forge \
    r-irkernel \
    r-tidyverse \
    r-bigrquery \
    r-googlecloudstorager \
    r-reticulate \
    r-plotly \
    r-leaflet \
    r-dt

# Install JupyterLab extensions
RUN jupyter labextension install \
    @jupyterlab/git \
    @jupyterlab/github \
    @jupyter-widgets/jupyterlab-manager \
    @pyviz/jupyterlab_pyviz \
    jupyterlab-plotly \
    @bokeh/jupyter_bokeh

# Create directories for notebooks and config
RUN mkdir -p /home/$NB_USER/notebooks /home/$NB_USER/config /home/$NB_USER/data

# Copy configuration files
COPY config/jupyter_notebook_config.py /etc/jupyter/
COPY config/custom.css /home/$NB_USER/.jupyter/custom/
COPY notebooks/ /home/$NB_USER/notebooks/
COPY scripts/ /home/$NB_USER/scripts/

# Set up environment variables
ENV JUPYTER_ENABLE_LAB=yes
ENV JUPYTER_TOKEN=""
ENV JUPYTER_PASSWORD=""
ENV GCP_PROJECT_ID=""
ENV GOOGLE_APPLICATION_CREDENTIALS="/home/${NB_USER}/config/service-account-key.json"

# Create startup script
COPY start-notebook.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/start-notebook.sh

# Set working directory
WORKDIR /home/$NB_USER/notebooks

# Expose Jupyter port
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8888/api || exit 1

# Start Jupyter
CMD ["start-notebook.sh", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=''", "--ServerApp.password=''"] 