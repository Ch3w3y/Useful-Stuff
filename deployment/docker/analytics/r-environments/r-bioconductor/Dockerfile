# R Bioconductor Environment for Genomics Analysis
# =================================================
# 
# Comprehensive Docker image for R-based bioinformatics and genomics
# Based on rocker/r-ver with Bioconductor and specialized packages
#
# Usage:
#   docker build -t r-bioconductor .
#   docker run -p 8787:8787 -v $(pwd):/workspace r-bioconductor
#
# Author: Data Science Team
# Date: 2024

# Use official R base image with specific version
FROM rocker/r-ver:4.3.2

# Set labels
LABEL maintainer="Data Science Team"
LABEL description="R environment for bioinformatics and genomics analysis"
LABEL version="1.0.0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV R_VERSION=4.3.2
ENV BIOCONDUCTOR_VERSION=3.18
ENV RSTUDIO_VERSION=2023.12.1+402
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    cmake \
    gfortran \
    # Scientific libraries
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    # Graphics and GUI
    libcairo2-dev \
    libxt-dev \
    libx11-dev \
    libxext-dev \
    libxft-dev \
    libxmu-dev \
    libxss1 \
    # Image processing
    libpng-dev \
    libjpeg-dev \
    libtiff5-dev \
    libwebp-dev \
    libmagick++-dev \
    librsvg2-dev \
    # XML and text processing
    libxml2-dev \
    libxslt1-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libssh2-1-dev \
    # Font support
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    # Spatial analysis
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libudunits2-dev \
    # Database connectivity
    libpq-dev \
    libmariadb-dev \
    libsqlite3-dev \
    unixodbc-dev \
    # Document processing
    libpoppler-cpp-dev \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    # Bioinformatics tools
    bedtools \
    samtools \
    bcftools \
    tabix \
    # Network tools
    wget \
    curl \
    git \
    # Text editors
    vim \
    nano \
    # Process monitoring
    htop \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install additional genomics tools
RUN cd /tmp && \
    # Install BLAST+
    wget https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/ncbi-blast-2.15.0+-x64-linux.tar.gz && \
    tar -xzf ncbi-blast-2.15.0+-x64-linux.tar.gz && \
    cp ncbi-blast-2.15.0+/bin/* /usr/local/bin/ && \
    # Install FastQC
    wget https://www.bioinformatics.babraham.ac.uk/projects/fastqc/fastqc_v0.12.1.zip && \
    unzip fastqc_v0.12.1.zip && \
    chmod +x FastQC/fastqc && \
    cp FastQC/fastqc /usr/local/bin/ && \
    cp -r FastQC /opt/ && \
    # Install MultiQC
    pip3 install multiqc && \
    # Clean up
    rm -rf /tmp/*

# Set up R library directory
RUN mkdir -p /usr/local/lib/R/site-library && \
    chmod -R 755 /usr/local/lib/R/site-library

# Configure R
RUN echo "options(repos = c(CRAN = 'https://cloud.r-project.org/'))" > /usr/local/lib/R/etc/Rprofile.site && \
    echo "options(download.file.method = 'libcurl')" >> /usr/local/lib/R/etc/Rprofile.site && \
    echo "options(Ncpus = parallel::detectCores())" >> /usr/local/lib/R/etc/Rprofile.site && \
    echo ".libPaths('/usr/local/lib/R/site-library')" >> /usr/local/lib/R/etc/Rprofile.site

# Install Bioconductor
RUN R -e "install.packages('BiocManager')" && \
    R -e "BiocManager::install(version = '3.18')"

# Install core R packages
RUN R -e "install.packages(c( \
    'devtools', \
    'remotes', \
    'pak', \
    'renv', \
    'tidyverse', \
    'data.table', \
    'dtplyr', \
    'readxl', \
    'openxlsx', \
    'jsonlite', \
    'yaml', \
    'here', \
    'glue', \
    'stringr', \
    'lubridate', \
    'forcats' \
    ))"

# Install statistical and ML packages
RUN R -e "install.packages(c( \
    'lme4', \
    'nlme', \
    'survival', \
    'survminer', \
    'caret', \
    'randomForest', \
    'xgboost', \
    'e1071', \
    'glmnet', \
    'MASS', \
    'car', \
    'broom', \
    'broom.mixed', \
    'performance', \
    'see', \
    'report' \
    ))"

# Install time series packages
RUN R -e "install.packages(c( \
    'forecast', \
    'prophet', \
    'tseries', \
    'zoo', \
    'xts', \
    'lubridate', \
    'timetk', \
    'modeltime' \
    ))"

# Install visualization packages
RUN R -e "install.packages(c( \
    'ggplot2', \
    'ggthemes', \
    'ggrepel', \
    'ggpubr', \
    'patchwork', \
    'cowplot', \
    'gridExtra', \
    'pheatmap', \
    'corrplot', \
    'VennDiagram', \
    'UpSetR', \
    'plotly', \
    'DT', \
    'htmlwidgets', \
    'leaflet', \
    'circlize' \
    ))"

# Install document generation packages
RUN R -e "install.packages(c( \
    'rmarkdown', \
    'knitr', \
    'bookdown', \
    'blogdown', \
    'xaringan', \
    'flexdashboard', \
    'shiny', \
    'shinydashboard', \
    'shinyWidgets', \
    'DT' \
    ))"

# Install core Bioconductor packages
RUN R -e "BiocManager::install(c( \
    'Biobase', \
    'BiocGenerics', \
    'S4Vectors', \
    'IRanges', \
    'GenomicRanges', \
    'GenomicFeatures', \
    'GenomicAlignments', \
    'Biostrings', \
    'BSgenome', \
    'rtracklayer', \
    'AnnotationDbi', \
    'org.Hs.eg.db', \
    'org.Mm.eg.db', \
    'TxDb.Hsapiens.UCSC.hg38.knownGene', \
    'TxDb.Mmusculus.UCSC.mm10.knownGene', \
    'BSgenome.Hsapiens.UCSC.hg38', \
    'BSgenome.Mmusculus.UCSC.mm38' \
    ))"

# Install RNA-seq analysis packages
RUN R -e "BiocManager::install(c( \
    'DESeq2', \
    'edgeR', \
    'limma', \
    'tximport', \
    'biomaRt', \
    'clusterProfiler', \
    'DOSE', \
    'ReactomePA', \
    'pathview', \
    'enrichplot', \
    'fgsea', \
    'GSVA', \
    'GSEABase', \
    'msigdbr' \
    ))"

# Install single-cell analysis packages
RUN R -e "install.packages(c('Seurat', 'SeuratObject'))" && \
    R -e "BiocManager::install(c( \
    'SingleCellExperiment', \
    'scater', \
    'scran', \
    'scuttle', \
    'DropletUtils', \
    'celda', \
    'slingshot', \
    'monocle3', \
    'destiny', \
    'SCENIC' \
    ))"

# Install specialized bioinformatics packages
RUN R -e "BiocManager::install(c( \
    'VariantAnnotation', \
    'VariantFiltering', \
    'DNAcopy', \
    'CNTools', \
    'Rsamtools', \
    'GenomicFiles', \
    'Gviz', \
    'ggbio', \
    'karyoploteR', \
    'ChIPseeker', \
    'DiffBind', \
    'csaw', \
    'ATACseqQC', \
    'chromVAR' \
    ))"

# Install proteomics packages
RUN R -e "BiocManager::install(c( \
    'MSnbase', \
    'xcms', \
    'CAMERA', \
    'mzR', \
    'MSstats', \
    'MSstatsTMT', \
    'MSstatsLOBD', \
    'limma', \
    'vsn', \
    'preprocessCore' \
    ))"

# Install phylogenetics packages
RUN R -e "install.packages(c( \
    'ape', \
    'phangorn', \
    'ggtree', \
    'treeio', \
    'phytools', \
    'geiger', \
    'picante', \
    'vegan', \
    'phyloseq' \
    ))"

# Install network analysis packages
RUN R -e "install.packages(c( \
    'igraph', \
    'network', \
    'sna', \
    'intergraph', \
    'visNetwork', \
    'networkD3' \
    ))" && \
    R -e "BiocManager::install(c( \
    'WGCNA', \
    'STRINGdb', \
    'RCy3', \
    'graphite', \
    'KEGGgraph', \
    'Rgraphviz' \
    ))"

# Install spatial analysis packages
RUN R -e "install.packages(c( \
    'sf', \
    'sp', \
    'raster', \
    'terra', \
    'rgdal', \
    'rgeos', \
    'maptools', \
    'leaflet', \
    'mapview', \
    'tmap' \
    ))"

# Install additional visualization packages for genomics
RUN R -e "BiocManager::install(c( \
    'ComplexHeatmap', \
    'EnhancedVolcano', \
    'ggbio', \
    'Gviz', \
    'trackViewer', \
    'GenomeInfoDb' \
    ))"

# Install Bayesian analysis packages
RUN R -e "install.packages(c( \
    'rstanarm', \
    'brms', \
    'MCMCglmm', \
    'BayesFactor', \
    'bayesplot', \
    'tidybayes', \
    'loo', \
    'bridgesampling' \
    ))"

# Install RStudio Server
RUN wget https://download2.rstudio.org/server/jammy/amd64/rstudio-server-${RSTUDIO_VERSION}-amd64.deb && \
    dpkg -i rstudio-server-${RSTUDIO_VERSION}-amd64.deb || true && \
    apt-get install -f -y && \
    rm rstudio-server-${RSTUDIO_VERSION}-amd64.deb

# Configure RStudio Server
RUN echo "www-port=8787" >> /etc/rstudio/rserver.conf && \
    echo "www-address=0.0.0.0" >> /etc/rstudio/rserver.conf && \
    echo "auth-required-user-group=rstudio" >> /etc/rstudio/rserver.conf

# Create rstudio user
RUN useradd -m -s /bin/bash rstudio && \
    echo "rstudio:rstudio" | chpasswd && \
    usermod -aG sudo rstudio

# Create workspace directory
RUN mkdir -p /workspace && \
    chown -R rstudio:rstudio /workspace

# Install Jupyter with R kernel
RUN apt-get update && apt-get install -y python3-pip && \
    pip3 install jupyter jupyterlab && \
    R -e "install.packages('IRkernel')" && \
    R -e "IRkernel::installspec(user = FALSE)"

# Set up environment for bioinformatics tools
ENV PATH="/opt/FastQC:${PATH}"
ENV BLASTDB="/data/blast"

# Create directories for common bioinformatics data
RUN mkdir -p /data/{genomes,annotation,blast,fastq,results} && \
    chown -R rstudio:rstudio /data

# Copy configuration files
COPY --chown=rstudio:rstudio configs/.Rprofile /home/rstudio/.Rprofile
COPY --chown=rstudio:rstudio configs/rstudio-prefs.json /home/rstudio/.config/rstudio/rstudio-prefs.json

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Start RStudio Server\n\
/usr/lib/rstudio-server/bin/rserver --server-daemonize 0 --www-port 8787 &\n\
\n\
# Print access information\n\
echo "================================="\n\
echo "R Bioconductor Environment Ready!"\n\
echo "================================="\n\
echo "RStudio Server: http://localhost:8787"\n\
echo "Username: rstudio"\n\
echo "Password: rstudio"\n\
echo ""\n\
echo "Jupyter Lab: jupyter lab --ip=0.0.0.0 --port=8888 --allow-root"\n\
echo ""\n\
echo "Installed packages:"\n\
echo "- R $(R --version | head -n1 | awk '"'"'{print $3}'"'"')"\n\
echo "- Bioconductor $(R -e \"cat(as.character(BiocManager::version()))\" 2>/dev/null)"\n\
echo "- $(R -e \"cat(length(installed.packages()[,1]))\" 2>/dev/null) R packages"\n\
echo "- FastQC, MultiQC, BLAST+, SAMtools, BCFtools"\n\
echo "================================="\n\
\n\
# Keep container running\n\
tail -f /dev/null' > /start.sh && \
    chmod +x /start.sh

# Set working directory
WORKDIR /workspace

# Switch to rstudio user
USER rstudio

# Expose ports
EXPOSE 8787 8888

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8787/ || exit 1

# Set default command
CMD ["/start.sh"] 