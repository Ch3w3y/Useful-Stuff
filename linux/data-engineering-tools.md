# Linux Data Engineering Tools & Workflows

A comprehensive guide to Linux-native tools and workflows for data engineering, focusing on command-line utilities, automation, monitoring, and optimization for high-performance data processing.

## Table of Contents

1. [Essential Command-Line Tools](#essential-command-line-tools)
2. [Data Processing Pipelines](#data-processing-pipelines)
3. [System Monitoring & Performance](#system-monitoring--performance)
4. [Container Orchestration](#container-orchestration)
5. [Database Management](#database-management)
6. [Stream Processing](#stream-processing)
7. [Automation & Scheduling](#automation--scheduling)
8. [Network & Security Tools](#network--security-tools)
9. [File System Optimization](#file-system-optimization)
10. [Backup & Recovery](#backup--recovery)

## Essential Command-Line Tools

### Modern Unix Tools for Data Engineers

#### Installation on Arch/CachyOS
```bash
# Essential modern CLI tools
sudo pacman -S ripgrep fd bat exa fzf htop btop ncdu tldr \
               jq yq-go parallel gnu-parallel gawk miller \
               rsync rclone hyperfine bandwhich dust \
               procs sd tokei gitui lazygit

# AUR packages for additional tools
yay -S csvkit-git visidata pv progress csvtool \
       q-text-as-data dasel xsv choose

# Python data tools
pip install --user pandas-profiling great-expectations \
            dbt-core dbt-postgres prefect \
            apache-airflow kedro
```

#### Installation on Ubuntu/Debian
```bash
# Update package lists
sudo apt update

# Essential tools
sudo apt install -y ripgrep fd-find bat exa fzf htop \
                   ncdu tldr jq parallel gawk \
                   rsync rclone pv progress \
                   silversearcher-ag tree

# Snap packages
sudo snap install yq dasel

# Python tools via pip
pip3 install --user csvkit visidata q miller \
             pandas-profiling great-expectations \
             dbt-core prefect apache-airflow
```

### Data Manipulation Tools

#### CSV/TSV Processing
```bash
# csvkit - Swiss Army knife for CSV files
sudo pip3 install csvkit

# Basic operations
csvstat data.csv                    # Statistics
csvcut -c 1,3,5 data.csv           # Select columns
csvgrep -c 3 -m "pattern" data.csv # Filter rows
csvjoin -c id file1.csv file2.csv  # Join files
csvsql --query "SELECT * FROM stdin WHERE col > 100" data.csv

# xsv - Fast CSV toolkit (install via cargo)
cargo install xsv

# Example operations
xsv headers data.csv               # Show headers
xsv select 1,3-5 data.csv         # Select columns
xsv stats data.csv                 # Statistics
xsv search -s column "pattern" data.csv
xsv join id file1.csv id file2.csv # Join files
```

#### JSON Processing
```bash
# jq - JSON processor
echo '{"name": "John", "age": 30}' | jq '.name'
curl -s api.endpoint.com | jq '.data[] | {id, name}'
jq -r '.items[] | [.id, .name, .value] | @csv' data.json

# Complex transformations
jq 'group_by(.category) | map({category: .[0].category, count: length, total: map(.value) | add})' data.json

# yq - YAML processor
yq eval '.database.host' config.yaml
yq eval -o json config.yaml        # Convert YAML to JSON
```

#### Text Processing
```bash
# Advanced grep with ripgrep
rg "pattern" --type json --stats   # Search with file type
rg -A 5 -B 5 "ERROR" logs/         # Context lines
rg -c "pattern" .                  # Count matches
rg --json "pattern" | jq '.data.lines.text'

# Miller (mlr) - CSV/JSON/YAML processor
mlr --csv stats1 -a mean,stddev -f price data.csv
mlr --json cat then put '$ratio = $sales / $target' data.json
mlr --csv nest --ivar item -f quantity,price data.csv
```

### Performance Monitoring
```bash
# btop - Modern top replacement
btop

# htop with custom config
htop --sort-key=PERCENT_CPU --tree

# iotop for I/O monitoring
sudo iotop -a -o -d 1

# Network monitoring
sudo bandwhich                     # Network utilization by process
ss -tulpn                         # Socket statistics
netstat -i                        # Network interface statistics

# Disk usage analysis
ncdu /data                        # Interactive disk usage
dust /data                        # Modern du replacement
df -h --output=source,size,used,avail,pcent,target
```

## Data Processing Pipelines

### Shell-Based ETL Pipeline
```bash
#!/bin/bash
# data-pipeline.sh - Comprehensive ETL pipeline

set -euo pipefail

# Configuration
DATA_DIR="/data"
LOGS_DIR="/var/log/data-pipeline"
TEMP_DIR="/tmp/pipeline-$$"
DATE=$(date +%Y%m%d_%H%M%S)

# Logging setup
setup_logging() {
    mkdir -p "${LOGS_DIR}"
    exec 1> >(tee -a "${LOGS_DIR}/pipeline_${DATE}.log")
    exec 2> >(tee -a "${LOGS_DIR}/pipeline_${DATE}.err" >&2)
}

# Error handling
trap_exit() {
    local exit_code=$?
    echo "[$(date)] Pipeline exited with code: ${exit_code}"
    cleanup
    exit ${exit_code}
}

trap trap_exit EXIT

# Cleanup function
cleanup() {
    echo "[$(date)] Cleaning up temporary files..."
    rm -rf "${TEMP_DIR}"
}

# Create temp directory
mkdir -p "${TEMP_DIR}"

# Extract phase
extract_data() {
    echo "[$(date)] Starting data extraction..."
    
    # Database extraction
    if command -v psql >/dev/null; then
        psql -h "${DB_HOST}" -U "${DB_USER}" -d "${DB_NAME}" \
             -c "COPY (SELECT * FROM transactions WHERE date >= '$(date -d '1 day ago' +%F)') TO STDOUT WITH CSV HEADER" \
             > "${TEMP_DIR}/transactions.csv"
    fi
    
    # API extraction
    if command -v curl >/dev/null; then
        curl -s -H "Authorization: Bearer ${API_TOKEN}" \
             "${API_ENDPOINT}/data?date=$(date -d '1 day ago' +%F)" \
             | jq -r '.results[] | [.id, .timestamp, .value] | @csv' \
             > "${TEMP_DIR}/api_data.csv"
    fi
    
    # File system extraction
    find "${DATA_DIR}/raw" -name "*.csv" -mtime -1 \
        -exec cp {} "${TEMP_DIR}/" \;
    
    echo "[$(date)] Data extraction completed"
}

# Transform phase
transform_data() {
    echo "[$(date)] Starting data transformation..."
    
    # Combine CSV files
    csvstack "${TEMP_DIR}"/*.csv > "${TEMP_DIR}/combined.csv"
    
    # Data cleaning and transformation
    mlr --csv clean-whitespace then \
        put '$amount = float($amount)' then \
        filter '$amount > 0' then \
        put '$date_processed = strftime(systime(), "%Y-%m-%d %H:%M:%S")' \
        "${TEMP_DIR}/combined.csv" > "${TEMP_DIR}/cleaned.csv"
    
    # Data validation
    csvstat "${TEMP_DIR}/cleaned.csv" > "${TEMP_DIR}/validation_report.txt"
    
    # Check for anomalies
    python3 -c "
import pandas as pd
import numpy as np

df = pd.read_csv('${TEMP_DIR}/cleaned.csv')

# Statistical anomaly detection
Q1 = df['amount'].quantile(0.25)
Q3 = df['amount'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

anomalies = df[(df['amount'] < lower_bound) | (df['amount'] > upper_bound)]
anomalies.to_csv('${TEMP_DIR}/anomalies.csv', index=False)

print(f'Found {len(anomalies)} anomalies')
"
    
    echo "[$(date)] Data transformation completed"
}

# Load phase
load_data() {
    echo "[$(date)] Starting data loading..."
    
    # Load to database
    if command -v psql >/dev/null; then
        psql -h "${DW_HOST}" -U "${DW_USER}" -d "${DW_NAME}" \
             -c "\COPY processed_transactions FROM '${TEMP_DIR}/cleaned.csv' WITH CSV HEADER"
    fi
    
    # Load to data lake
    if command -v aws >/dev/null; then
        aws s3 cp "${TEMP_DIR}/cleaned.csv" \
            "s3://${BUCKET}/processed/$(date +%Y/%m/%d)/transactions_${DATE}.csv"
    fi
    
    # Load to local storage
    cp "${TEMP_DIR}/cleaned.csv" "${DATA_DIR}/processed/transactions_${DATE}.csv"
    
    echo "[$(date)] Data loading completed"
}

# Main pipeline execution
main() {
    setup_logging
    
    echo "[$(date)] Starting ETL pipeline..."
    
    extract_data
    transform_data
    load_data
    
    echo "[$(date)] ETL pipeline completed successfully"
}

# Execute if run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
```

### Parallel Processing with GNU Parallel
```bash
#!/bin/bash
# parallel-processing.sh

# Process multiple files in parallel
process_file() {
    local file="$1"
    local output_dir="$2"
    
    echo "Processing: ${file}"
    
    # Example: CSV processing
    mlr --csv stats1 -a count,mean,stddev -f value "${file}" \
        > "${output_dir}/$(basename "${file}" .csv)_stats.csv"
    
    # Example: Data validation
    csvstat "${file}" > "${output_dir}/$(basename "${file}" .csv)_validation.txt"
}

export -f process_file

# Process files in parallel
find /data/input -name "*.csv" | \
    parallel -j $(nproc) process_file {} /data/output

# Parallel database operations
parallel -j 4 --colsep ',' \
    'psql -h {1} -d {2} -c "SELECT COUNT(*) FROM {3}"' \
    :::: database_connections.csv

# Parallel API calls
parallel -j 10 \
    'curl -s "https://api.example.com/data/{}" | jq "." > "data_{}.json"' \
    ::: {1..100}
```

## System Monitoring & Performance

### Custom Monitoring Script
```bash
#!/bin/bash
# system-monitor.sh - Comprehensive system monitoring

MONITOR_LOG="/var/log/system-monitor.log"
ALERT_THRESHOLD_CPU=80
ALERT_THRESHOLD_MEM=85
ALERT_THRESHOLD_DISK=90

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${MONITOR_LOG}"
}

check_cpu() {
    local cpu_usage
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
    
    if (( $(echo "${cpu_usage} > ${ALERT_THRESHOLD_CPU}" | bc -l) )); then
        log_message "ALERT: High CPU usage: ${cpu_usage}%"
        
        # Get top CPU consuming processes
        ps aux --sort=-%cpu | head -10 >> "${MONITOR_LOG}"
    fi
    
    echo "CPU Usage: ${cpu_usage}%"
}

check_memory() {
    local mem_info
    mem_info=$(free | grep Mem)
    local total=$(echo ${mem_info} | awk '{print $2}')
    local used=$(echo ${mem_info} | awk '{print $3}')
    local mem_usage=$(echo "scale=2; ${used}*100/${total}" | bc)
    
    if (( $(echo "${mem_usage} > ${ALERT_THRESHOLD_MEM}" | bc -l) )); then
        log_message "ALERT: High memory usage: ${mem_usage}%"
        
        # Get top memory consuming processes
        ps aux --sort=-%mem | head -10 >> "${MONITOR_LOG}"
    fi
    
    echo "Memory Usage: ${mem_usage}%"
}

check_disk() {
    local disk_usage
    disk_usage=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
    
    if [ "${disk_usage}" -gt "${ALERT_THRESHOLD_DISK}" ]; then
        log_message "ALERT: High disk usage: ${disk_usage}%"
        
        # Find large files
        find / -type f -size +100M 2>/dev/null | head -20 >> "${MONITOR_LOG}"
    fi
    
    echo "Disk Usage: ${disk_usage}%"
}

check_services() {
    local services=("postgresql" "redis" "nginx" "docker")
    
    for service in "${services[@]}"; do
        if systemctl is-active --quiet "${service}"; then
            echo "Service ${service}: Running"
        else
            log_message "ALERT: Service ${service} is not running"
        fi
    done
}

check_network() {
    # Check network connectivity
    if ping -c 1 google.com &> /dev/null; then
        echo "Network: Connected"
    else
        log_message "ALERT: Network connectivity issues"
    fi
    
    # Check listening ports
    netstat -tlnp | grep LISTEN >> "${MONITOR_LOG}"
}

main() {
    log_message "Starting system monitoring check"
    
    check_cpu
    check_memory
    check_disk
    check_services
    check_network
    
    log_message "System monitoring check completed"
}

main "$@"
```

### Performance Profiling
```bash
#!/bin/bash
# profile-performance.sh

profile_io() {
    echo "=== I/O Performance Profiling ==="
    
    # Test write performance
    echo "Testing write performance..."
    dd if=/dev/zero of=/tmp/testfile bs=1M count=1024 conv=fdatasync 2>&1 | \
        grep -E "(copied|MB/s)"
    
    # Test read performance
    echo "Testing read performance..."
    dd if=/tmp/testfile of=/dev/null bs=1M 2>&1 | \
        grep -E "(copied|MB/s)"
    
    # Random I/O test with fio (if available)
    if command -v fio >/dev/null; then
        fio --name=random-write --ioengine=libaio --iodepth=1 \
            --rw=randwrite --bs=4k --direct=1 --size=1G \
            --numjobs=1 --runtime=30 --group_reporting \
            --filename=/tmp/fio-test
    fi
    
    rm -f /tmp/testfile /tmp/fio-test
}

profile_network() {
    echo "=== Network Performance Profiling ==="
    
    # Network interface statistics
    cat /proc/net/dev
    
    # Bandwidth test (if iperf3 available)
    if command -v iperf3 >/dev/null; then
        echo "Run: iperf3 -s on target machine"
        echo "Then: iperf3 -c <target_ip> -t 30"
    fi
}

profile_cpu() {
    echo "=== CPU Performance Profiling ==="
    
    # CPU info
    lscpu
    
    # CPU benchmark
    echo "Running CPU benchmark..."
    time $(seq 1 100000 | while read i; do echo $i > /dev/null; done)
    
    # Parallel processing test
    echo "Testing parallel processing..."
    time parallel -j $(nproc) echo {} ::: {1..10000} > /dev/null
}

profile_memory() {
    echo "=== Memory Performance Profiling ==="
    
    # Memory info
    free -h
    cat /proc/meminfo | grep -E "(MemTotal|MemFree|Cached|Buffers)"
    
    # Memory bandwidth test (if available)
    if command -v stream >/dev/null; then
        stream
    fi
}

main() {
    echo "Starting comprehensive performance profiling..."
    echo "Date: $(date)"
    echo "System: $(uname -a)"
    echo "=================================="
    
    profile_cpu
    profile_memory
    profile_io
    profile_network
    
    echo "=================================="
    echo "Performance profiling completed"
}

main "$@"
```

## Container Orchestration

### Docker Compose for Data Stack
```yaml
# docker-compose.data-stack.yml
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    container_name: data-postgres
    environment:
      POSTGRES_DB: datawarehouse
      POSTGRES_USER: datauser
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-scripts:/docker-entrypoint-initdb.d
    ports:
      - "5432:5432"
    networks:
      - data-network
    restart: unless-stopped

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: data-redis
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    networks:
      - data-network
    restart: unless-stopped

  # Apache Kafka
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    container_name: data-zookeeper
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    networks:
      - data-network

  kafka:
    image: confluentinc/cp-kafka:latest
    container_name: data-kafka
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
    networks:
      - data-network

  # Apache Airflow
  airflow-webserver:
    image: apache/airflow:2.8.0
    container_name: data-airflow-webserver
    command: webserver
    entrypoint: ./scripts/airflow-entrypoint.sh
    depends_on:
      - postgres
    environment:
      - AIRFLOW__CORE__EXECUTOR=LocalExecutor
      - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://datauser:${POSTGRES_PASSWORD}@postgres/airflow
      - AIRFLOW__CORE__FERNET_KEY=${AIRFLOW_FERNET_KEY}
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
      - ./plugins:/opt/airflow/plugins
      - ./scripts:/opt/airflow/scripts
    ports:
      - "8080:8080"
    networks:
      - data-network
    restart: unless-stopped

  # Jupyter Lab
  jupyter:
    image: jupyter/datascience-notebook:latest
    container_name: data-jupyter
    environment:
      - JUPYTER_ENABLE_LAB=yes
      - JUPYTER_TOKEN=${JUPYTER_TOKEN}
    volumes:
      - ./notebooks:/home/jovyan/work
      - jupyter_data:/home/jovyan
    ports:
      - "8888:8888"
    networks:
      - data-network
    restart: unless-stopped

  # MinIO (S3-compatible storage)
  minio:
    image: minio/minio:latest
    container_name: data-minio
    command: server /data --console-address ":9001"
    environment:
      - MINIO_ACCESS_KEY=${MINIO_ACCESS_KEY}
      - MINIO_SECRET_KEY=${MINIO_SECRET_KEY}
    volumes:
      - minio_data:/data
    ports:
      - "9000:9000"
      - "9001:9001"
    networks:
      - data-network
    restart: unless-stopped

  # Grafana
  grafana:
    image: grafana/grafana:latest
    container_name: data-grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
    ports:
      - "3000:3000"
    networks:
      - data-network
    restart: unless-stopped

  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    container_name: data-prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - data-network
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  jupyter_data:
  minio_data:
  grafana_data:
  prometheus_data:

networks:
  data-network:
    driver: bridge
```

### Container Management Scripts
```bash
#!/bin/bash
# container-manager.sh

COMPOSE_FILE="docker-compose.data-stack.yml"
ENV_FILE=".env"

# Load environment variables
if [[ -f "${ENV_FILE}" ]]; then
    source "${ENV_FILE}"
fi

start_stack() {
    echo "Starting data stack..."
    docker-compose -f "${COMPOSE_FILE}" up -d
    
    echo "Waiting for services to be ready..."
    sleep 30
    
    check_health
}

stop_stack() {
    echo "Stopping data stack..."
    docker-compose -f "${COMPOSE_FILE}" down
}

restart_stack() {
    echo "Restarting data stack..."
    stop_stack
    start_stack
}

check_health() {
    echo "=== Service Health Check ==="
    
    services=("postgres:5432" "redis:6379" "kafka:9092" "jupyter:8888" "minio:9000")
    
    for service in "${services[@]}"; do
        IFS=':' read -r name port <<< "${service}"
        if nc -z localhost "${port}"; then
            echo "✓ ${name} is healthy"
        else
            echo "✗ ${name} is not responding"
        fi
    done
}

backup_data() {
    echo "Creating data backup..."
    
    # Create backup directory
    backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "${backup_dir}"
    
    # Database backup
    docker exec data-postgres pg_dumpall -U datauser > "${backup_dir}/postgres_backup.sql"
    
    # Volume backups
    docker run --rm -v data_postgres_data:/data -v $(pwd)/${backup_dir}:/backup alpine \
        tar czf /backup/postgres_data.tar.gz -C /data .
    
    docker run --rm -v data_minio_data:/data -v $(pwd)/${backup_dir}:/backup alpine \
        tar czf /backup/minio_data.tar.gz -C /data .
    
    echo "Backup completed: ${backup_dir}"
}

show_logs() {
    local service="${1:-}"
    
    if [[ -n "${service}" ]]; then
        docker-compose -f "${COMPOSE_FILE}" logs -f "${service}"
    else
        docker-compose -f "${COMPOSE_FILE}" logs -f
    fi
}

show_stats() {
    echo "=== Container Statistics ==="
    docker stats --no-stream
    
    echo -e "\n=== Resource Usage ==="
    docker system df
}

case "${1:-}" in
    start)
        start_stack
        ;;
    stop)
        stop_stack
        ;;
    restart)
        restart_stack
        ;;
    health)
        check_health
        ;;
    backup)
        backup_data
        ;;
    logs)
        show_logs "${2}"
        ;;
    stats)
        show_stats
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|health|backup|logs [service]|stats}"
        exit 1
        ;;
esac
```

## Database Management

### PostgreSQL Administration Scripts
```bash
#!/bin/bash
# postgres-admin.sh

DB_HOST="${DB_HOST:-localhost}"
DB_USER="${DB_USER:-postgres}"
DB_NAME="${DB_NAME:-datawarehouse}"

# Database maintenance
vacuum_analyze() {
    echo "Running VACUUM ANALYZE on all tables..."
    
    psql -h "${DB_HOST}" -U "${DB_USER}" -d "${DB_NAME}" -c "
    DO \$\$
    DECLARE
        r RECORD;
    BEGIN
        FOR r IN SELECT schemaname, tablename FROM pg_tables 
                 WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
        LOOP
            EXECUTE 'VACUUM ANALYZE ' || quote_ident(r.schemaname) || '.' || quote_ident(r.tablename);
            RAISE NOTICE 'Processed: %.%', r.schemaname, r.tablename;
        END LOOP;
    END
    \$\$;"
}

# Index maintenance
reindex_tables() {
    echo "Reindexing all tables..."
    
    psql -h "${DB_HOST}" -U "${DB_USER}" -d "${DB_NAME}" -c "
    SELECT 'REINDEX TABLE ' || schemaname || '.' || tablename || ';' as reindex_cmd
    FROM pg_tables 
    WHERE schemaname NOT IN ('information_schema', 'pg_catalog');" \
    | grep REINDEX | psql -h "${DB_HOST}" -U "${DB_USER}" -d "${DB_NAME}"
}

# Performance monitoring
check_slow_queries() {
    echo "=== Slow Queries Report ==="
    
    psql -h "${DB_HOST}" -U "${DB_USER}" -d "${DB_NAME}" -c "
    SELECT 
        query,
        calls,
        total_time,
        mean_time,
        rows
    FROM pg_stat_statements
    ORDER BY mean_time DESC
    LIMIT 10;"
}

# Connection monitoring
check_connections() {
    echo "=== Active Connections ==="
    
    psql -h "${DB_HOST}" -U "${DB_USER}" -d "${DB_NAME}" -c "
    SELECT 
        datname,
        usename,
        client_addr,
        state,
        query_start,
        LEFT(query, 50) as query_preview
    FROM pg_stat_activity
    WHERE state != 'idle'
    ORDER BY query_start;"
}

# Backup functions
create_backup() {
    local backup_dir="${1:-/tmp/postgres_backups}"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    
    mkdir -p "${backup_dir}"
    
    echo "Creating backup: ${backup_dir}/backup_${timestamp}.sql"
    pg_dump -h "${DB_HOST}" -U "${DB_USER}" "${DB_NAME}" \
        | gzip > "${backup_dir}/backup_${timestamp}.sql.gz"
    
    # Keep only last 7 days of backups
    find "${backup_dir}" -name "backup_*.sql.gz" -mtime +7 -delete
}

case "${1:-}" in
    vacuum)
        vacuum_analyze
        ;;
    reindex)
        reindex_tables
        ;;
    slow-queries)
        check_slow_queries
        ;;
    connections)
        check_connections
        ;;
    backup)
        create_backup "${2}"
        ;;
    *)
        echo "Usage: $0 {vacuum|reindex|slow-queries|connections|backup [dir]}"
        exit 1
        ;;
esac
```

## Stream Processing

### Kafka Administration
```bash
#!/bin/bash
# kafka-admin.sh

KAFKA_HOME="/opt/kafka"
BOOTSTRAP_SERVERS="localhost:9092"

# Topic management
create_topic() {
    local topic_name="$1"
    local partitions="${2:-3}"
    local replication="${3:-1}"
    
    "${KAFKA_HOME}/bin/kafka-topics.sh" \
        --create \
        --bootstrap-server "${BOOTSTRAP_SERVERS}" \
        --topic "${topic_name}" \
        --partitions "${partitions}" \
        --replication-factor "${replication}"
}

list_topics() {
    "${KAFKA_HOME}/bin/kafka-topics.sh" \
        --list \
        --bootstrap-server "${BOOTSTRAP_SERVERS}"
}

describe_topic() {
    local topic_name="$1"
    
    "${KAFKA_HOME}/bin/kafka-topics.sh" \
        --describe \
        --bootstrap-server "${BOOTSTRAP_SERVERS}" \
        --topic "${topic_name}"
}

# Consumer group management
list_consumer_groups() {
    "${KAFKA_HOME}/bin/kafka-consumer-groups.sh" \
        --bootstrap-server "${BOOTSTRAP_SERVERS}" \
        --list
}

describe_consumer_group() {
    local group_name="$1"
    
    "${KAFKA_HOME}/bin/kafka-consumer-groups.sh" \
        --bootstrap-server "${BOOTSTRAP_SERVERS}" \
        --group "${group_name}" \
        --describe
}

# Data streaming
produce_data() {
    local topic_name="$1"
    local data_file="$2"
    
    if [[ -f "${data_file}" ]]; then
        cat "${data_file}" | "${KAFKA_HOME}/bin/kafka-console-producer.sh" \
            --bootstrap-server "${BOOTSTRAP_SERVERS}" \
            --topic "${topic_name}"
    else
        echo "Data file not found: ${data_file}"
        exit 1
    fi
}

consume_data() {
    local topic_name="$1"
    local group_id="${2:-console-consumer}"
    
    "${KAFKA_HOME}/bin/kafka-console-consumer.sh" \
        --bootstrap-server "${BOOTSTRAP_SERVERS}" \
        --topic "${topic_name}" \
        --group "${group_id}" \
        --from-beginning
}

# Performance monitoring
check_performance() {
    echo "=== Kafka Performance Metrics ==="
    
    # Topic performance
    "${KAFKA_HOME}/bin/kafka-run-class.sh" kafka.tools.JmxTool \
        --object-name kafka.server:type=BrokerTopicMetrics,name=MessagesInPerSec \
        --jmx-url service:jmx:rmi:///jndi/rmi://localhost:9999/jmxrmi \
        --date-format "YYYY-MM-dd HH:mm:ss" \
        --attributes Count,OneMinuteRate | head -20
}

case "${1:-}" in
    create-topic)
        create_topic "${2}" "${3}" "${4}"
        ;;
    list-topics)
        list_topics
        ;;
    describe-topic)
        describe_topic "${2}"
        ;;
    list-groups)
        list_consumer_groups
        ;;
    describe-group)
        describe_consumer_group "${2}"
        ;;
    produce)
        produce_data "${2}" "${3}"
        ;;
    consume)
        consume_data "${2}" "${3}"
        ;;
    performance)
        check_performance
        ;;
    *)
        echo "Usage: $0 {create-topic|list-topics|describe-topic|list-groups|describe-group|produce|consume|performance}"
        exit 1
        ;;
esac
```

### Real-time Stream Processing
```python
#!/usr/bin/env python3
# stream-processor.py

import json
import logging
from typing import Dict, Any
from kafka import KafkaConsumer, KafkaProducer
import pandas as pd
from datetime import datetime
import redis
import psycopg2
from psycopg2.extras import execute_values

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StreamProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.consumer = self._create_consumer()
        self.producer = self._create_producer()
        self.redis_client = redis.Redis(
            host=config.get('redis_host', 'localhost'),
            port=config.get('redis_port', 6379),
            decode_responses=True
        )
        self.db_connection = self._create_db_connection()
    
    def _create_consumer(self):
        return KafkaConsumer(
            self.config['input_topic'],
            bootstrap_servers=self.config.get('kafka_servers', ['localhost:9092']),
            group_id=self.config.get('consumer_group', 'stream-processor'),
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=True
        )
    
    def _create_producer(self):
        return KafkaProducer(
            bootstrap_servers=self.config.get('kafka_servers', ['localhost:9092']),
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
    
    def _create_db_connection(self):
        return psycopg2.connect(
            host=self.config.get('db_host', 'localhost'),
            database=self.config.get('db_name', 'datawarehouse'),
            user=self.config.get('db_user', 'postgres'),
            password=self.config.get('db_password', '')
        )
    
    def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process individual message"""
        try:
            # Add processing timestamp
            message['processed_at'] = datetime.now().isoformat()
            
            # Data enrichment from Redis cache
            if 'user_id' in message:
                user_data = self.redis_client.hgetall(f"user:{message['user_id']}")
                if user_data:
                    message['user_data'] = user_data
            
            # Apply business logic
            if message.get('event_type') == 'transaction':
                message = self._process_transaction(message)
            elif message.get('event_type') == 'user_activity':
                message = self._process_user_activity(message)
            
            return message
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return None
    
    def _process_transaction(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process transaction events"""
        amount = float(message.get('amount', 0))
        
        # Fraud detection
        if amount > 10000:
            message['fraud_flag'] = True
            message['risk_score'] = 1.0
        else:
            message['fraud_flag'] = False
            message['risk_score'] = amount / 10000
        
        # Category enrichment
        merchant = message.get('merchant', '')
        message['category'] = self._get_merchant_category(merchant)
        
        return message
    
    def _process_user_activity(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process user activity events"""
        user_id = message.get('user_id')
        activity_type = message.get('activity_type')
        
        # Update user session in Redis
        session_key = f"session:{user_id}"
        self.redis_client.hset(session_key, 'last_activity', datetime.now().isoformat())
        self.redis_client.hset(session_key, 'activity_type', activity_type)
        self.redis_client.expire(session_key, 3600)  # 1 hour TTL
        
        return message
    
    def _get_merchant_category(self, merchant: str) -> str:
        """Get merchant category from database"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute(
                    "SELECT category FROM merchant_categories WHERE merchant_name = %s",
                    (merchant,)
                )
                result = cursor.fetchone()
                return result[0] if result else 'unknown'
        except Exception as e:
            logger.error(f"Error getting merchant category: {e}")
            return 'unknown'
    
    def batch_process(self, messages: list, batch_size: int = 100):
        """Process messages in batches for database insertion"""
        processed_messages = []
        
        for message in messages:
            processed = self.process_message(message)
            if processed:
                processed_messages.append(processed)
        
        # Batch insert to database
        if processed_messages:
            self._batch_insert_to_db(processed_messages)
        
        return processed_messages
    
    def _batch_insert_to_db(self, messages: list):
        """Batch insert processed messages to database"""
        try:
            with self.db_connection.cursor() as cursor:
                # Prepare data for insertion
                values = [
                    (
                        msg.get('event_id'),
                        msg.get('event_type'),
                        json.dumps(msg),
                        msg.get('processed_at')
                    )
                    for msg in messages
                ]
                
                # Batch insert
                execute_values(
                    cursor,
                    """
                    INSERT INTO processed_events (event_id, event_type, event_data, processed_at)
                    VALUES %s
                    ON CONFLICT (event_id) DO NOTHING
                    """,
                    values
                )
                
                self.db_connection.commit()
                logger.info(f"Inserted {len(messages)} messages to database")
                
        except Exception as e:
            logger.error(f"Error inserting to database: {e}")
            self.db_connection.rollback()
    
    def run(self):
        """Main processing loop"""
        logger.info("Starting stream processor...")
        
        batch = []
        batch_size = self.config.get('batch_size', 100)
        
        try:
            for message in self.consumer:
                try:
                    data = message.value
                    batch.append(data)
                    
                    # Process batch when full
                    if len(batch) >= batch_size:
                        processed = self.batch_process(batch)
                        
                        # Send to output topic
                        for processed_msg in processed:
                            self.producer.send(
                                self.config['output_topic'],
                                processed_msg
                            )
                        
                        batch = []
                        logger.info(f"Processed batch of {len(processed)} messages")
                    
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    continue
                    
        except KeyboardInterrupt:
            logger.info("Shutting down stream processor...")
        finally:
            # Process remaining batch
            if batch:
                processed = self.batch_process(batch)
                for processed_msg in processed:
                    self.producer.send(self.config['output_topic'], processed_msg)
            
            self.consumer.close()
            self.producer.close()
            self.db_connection.close()

if __name__ == "__main__":
    config = {
        'input_topic': 'raw-events',
        'output_topic': 'processed-events',
        'kafka_servers': ['localhost:9092'],
        'consumer_group': 'stream-processor',
        'redis_host': 'localhost',
        'redis_port': 6379,
        'db_host': 'localhost',
        'db_name': 'datawarehouse',
        'db_user': 'postgres',
        'db_password': 'password',
        'batch_size': 100
    }
    
    processor = StreamProcessor(config)
    processor.run()
```

This comprehensive Linux data engineering tools guide provides practical, production-ready scripts and workflows that can be immediately implemented. The focus is on command-line efficiency, automation, and performance optimization specifically tailored for data engineering tasks on Linux systems. 