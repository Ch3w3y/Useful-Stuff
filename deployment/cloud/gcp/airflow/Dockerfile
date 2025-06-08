# Apache Airflow for GCP
# Production-ready Airflow with GCP providers and security configurations
FROM apache/airflow:2.7.3-python3.11

# Switch to root for system installations
USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    lsb-release \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update -y && apt-get install google-cloud-sdk -y && \
    rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

# Copy requirements file
COPY requirements.txt /tmp/requirements.txt

# Install Python packages
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install Airflow providers and additional packages
RUN pip install --no-cache-dir \
    apache-airflow[google,postgres,celery,redis,kubernetes,docker] \
    apache-airflow-providers-google==10.10.1 \
    apache-airflow-providers-postgres==5.7.0 \
    apache-airflow-providers-slack==7.3.0 \
    apache-airflow-providers-email==1.1.0 \
    apache-airflow-providers-http==4.5.1 \
    pandas \
    numpy \
    scikit-learn \
    pyarrow \
    fastparquet \
    sqlalchemy-bigquery \
    dbt-core \
    dbt-bigquery \
    great-expectations \
    soda-core[bigquery] \
    pandera

# Create directories
RUN mkdir -p /opt/airflow/dags/sql /opt/airflow/dags/scripts /opt/airflow/config /opt/airflow/logs

# Copy DAGs and configuration
COPY dags/ /opt/airflow/dags/
COPY config/airflow.cfg /opt/airflow/
COPY scripts/ /opt/airflow/scripts/

# Set environment variables
ENV AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=True
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV AIRFLOW__WEBSERVER__EXPOSE_CONFIG=False
ENV AIRFLOW__WEBSERVER__RBAC=True
ENV AIRFLOW__CORE__REMOTE_LOGGING=True
ENV AIRFLOW__LOGGING__REMOTE_LOGGING=True
ENV AIRFLOW__LOGGING__REMOTE_BASE_LOG_FOLDER=gs://your-airflow-logs-bucket/logs
ENV AIRFLOW__CORE__SECURITY=oauth
ENV AIRFLOW__WEBSERVER__AUTHENTICATE=True

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=5 \
    CMD airflow jobs check --job-type SchedulerJob --hostname "$${HOSTNAME}"

# Default command
CMD ["airflow", "webserver"]