# Shiny App for GCP Cloud Run
# Production-ready Shiny deployment with security best practices
FROM rocker/shiny-verse:4.3.2

# Create non-root user
RUN groupadd -r shinyuser && useradd -r -g shinyuser shinyuser

# Install system dependencies for Shiny and GCP integration
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    libxml2-dev \
    libssl-dev \
    libsasl2-dev \
    libpq-dev \
    libgit2-dev \
    pandoc \
    pandoc-citeproc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update -y && apt-get install google-cloud-sdk -y && \
    rm -rf /var/lib/apt/lists/*

# Install R packages for Shiny and GCP
RUN R -e "install.packages(c( \
    'shiny', \
    'shinydashboard', \
    'shinyWidgets', \
    'DT', \
    'plotly', \
    'leaflet', \
    'googleCloudStorageR', \
    'bigrquery', \
    'pool', \
    'config', \
    'log4r', \
    'httr', \
    'jsonlite', \
    'lubridate', \
    'scales', \
    'fresh', \
    'bslib', \
    'thematic' \
    ), repos='https://cran.rstudio.com/')"

# Create app directory
WORKDIR /srv/shiny-server

# Copy Shiny configuration
COPY shiny-server.conf /etc/shiny-server/shiny-server.conf

# Copy application files
COPY app/ ./app/
COPY config/ ./config/
COPY www/ ./www/
COPY logs/ ./logs/

# Create necessary directories and set permissions
RUN mkdir -p /var/log/shiny-server /var/lib/shiny-server /tmp/shiny-server && \
    chown -R shinyuser:shinyuser /srv/shiny-server /var/log/shiny-server /var/lib/shiny-server /tmp/shiny-server && \
    chmod -R 755 /srv/shiny-server

# Switch to non-root user
USER shinyuser

# Expose port
EXPOSE 8080

# Health check for Cloud Run
HEALTHCHECK --interval=30s --timeout=30s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

# Start Shiny Server
CMD ["/usr/bin/shiny-server"] 