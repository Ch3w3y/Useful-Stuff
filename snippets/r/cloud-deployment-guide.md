# R Cloud Deployment Guide

Quick-start guide for deploying R applications across cloud platforms with reproducible environments.

## Quick Setup Templates

### 1. GitHub Actions for R

```yaml
# .github/workflows/deploy-r.yml
name: Deploy R Application
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: r-lib/actions/setup-r@v2
      with:
        r-version: '4.3.2'
    
    - name: Install dependencies
      run: |
        install.packages('renv')
        renv::restore()
      shell: Rscript {0}
    
    - name: Build Docker image
      run: docker build -t my-r-app .
    
    - name: Deploy to Cloud Run
      run: |
        echo ${{ secrets.GCP_SA_KEY }} | base64 -d > gcp-key.json
        gcloud auth activate-service-account --key-file gcp-key.json
        gcloud run deploy r-app --image my-r-app --platform managed
```

### 2. Docker for R Applications

```dockerfile
# Dockerfile
FROM rocker/r-ver:4.3.2
RUN apt-get update && apt-get install -y libcurl4-openssl-dev libssl-dev libxml2-dev
COPY renv.lock renv.lock
RUN R -e "install.packages('renv'); renv::restore()"
COPY . /app
WORKDIR /app
EXPOSE 8000
CMD ["R", "-e", "plumber::plumb('api.R')$run(host='0.0.0.0', port=8000)"]
```

### 3. Simple Plumber API

```r
# api.R
library(plumber)

#* @get /predict
#* @param x Numeric input
function(x) {
  result <- as.numeric(x) * 2 + 1
  list(prediction = result)
}

#* @get /health
function() {
  list(status = "healthy", timestamp = Sys.time())
}
```

### 4. Environment Management

```r
# setup.R - Run this first
if (!require("renv")) install.packages("renv")
renv::init()
renv::install(c("plumber", "dplyr", "ggplot2"))
renv::snapshot()
```

## Cloud Platform Quick Deploys

### GCP Cloud Run
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT_ID/r-app
gcloud run deploy --image gcr.io/PROJECT_ID/r-app --platform managed
```

### AWS Lambda (with Docker)
```bash
# Deploy with SAM
sam build
sam deploy --guided
```

### Azure Container Instances
```bash
# Deploy to ACI
az container create --resource-group myRG --name r-app --image myregistry.azurecr.io/r-app
```

## Production Checklist

- [ ] Environment variables configured
- [ ] Health check endpoint implemented
- [ ] Logging setup
- [ ] Error handling in place
- [ ] Resource limits defined
- [ ] Monitoring configured
- [ ] Security headers added
- [ ] API documentation created

## Common Issues & Solutions

**Issue**: Package installation fails
```r
# Solution: Use binary packages
options(repos = c(CRAN = "https://packagemanager.rstudio.com/cran/__linux__/focal/latest"))
```

**Issue**: Memory errors in production
```r
# Solution: Implement memory management
gc() # Force garbage collection
options(memory.limit = 4000) # Set memory limit
```

**Issue**: Slow API responses
```r
# Solution: Add caching
library(memoise)
cached_function <- memoise(expensive_function)
``` 