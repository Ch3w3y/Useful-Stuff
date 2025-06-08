# R Script Runner for GCP
# Secure container for executing R scripts with Google Cloud Storage integration
FROM rocker/r-ver:4.3.2

# Set up non-root user for security
RUN groupadd -r ruser && useradd -r -g ruser ruser

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libgit2-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update -y && apt-get install google-cloud-sdk -y && \
    rm -rf /var/lib/apt/lists/*

# Install essential R packages
RUN R -e "install.packages(c( \
    'tidyverse', \
    'bigrquery', \
    'googleCloudStorageR', \
    'DBI', \
    'httr', \
    'jsonlite', \
    'lubridate', \
    'here', \
    'log4r', \
    'config', \
    'rmarkdown', \
    'knitr' \
    ), repos='https://cran.rstudio.com/')"

# Create application directory
WORKDIR /app

# Copy application files
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY data/ ./data/
COPY run_script.R ./
COPY entrypoint.sh ./

# Set permissions
RUN chmod +x entrypoint.sh && \
    chown -R ruser:ruser /app

# Switch to non-root user
USER ruser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD Rscript -e "cat('Container healthy\n')" || exit 1

# Set entrypoint
ENTRYPOINT ["./entrypoint.sh"] 