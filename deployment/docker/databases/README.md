# Docker Database Deployments

## Overview

Comprehensive guide to deploying and managing databases in Docker containers, covering PostgreSQL, MySQL, MongoDB, Redis, and specialized analytics databases with production-ready configurations.

## Table of Contents

- [PostgreSQL Deployment](#postgresql-deployment)
- [MySQL Deployment](#mysql-deployment)
- [MongoDB Deployment](#mongodb-deployment)
- [Redis Deployment](#redis-deployment)
- [Analytics Databases](#analytics-databases)
- [Backup & Recovery](#backup--recovery)
- [Monitoring & Maintenance](#monitoring--maintenance)
- [Security Best Practices](#security-best-practices)

## PostgreSQL Deployment

### Production PostgreSQL Configuration

```yaml
# docker-compose.postgres.yml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    container_name: postgres-prod
    restart: unless-stopped
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-myapp}
      POSTGRES_USER: ${POSTGRES_USER:-postgres}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_INITDB_ARGS: "--encoding=UTF-8 --lc-collate=C --lc-ctype=C"
      PGDATA: /var/lib/postgresql/data/pgdata
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./postgres/init:/docker-entrypoint-initdb.d:ro
      - ./postgres/config/postgresql.conf:/etc/postgresql/postgresql.conf:ro
      - ./postgres/config/pg_hba.conf:/etc/postgresql/pg_hba.conf:ro
      - ./postgres/backups:/backups
    ports:
      - "5432:5432"
    command: >
      postgres
      -c config_file=/etc/postgresql/postgresql.conf
      -c hba_file=/etc/postgresql/pg_hba.conf
    networks:
      - database_network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-postgres} -d ${POSTGRES_DB:-myapp}"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  postgres-exporter:
    image: prometheuscommunity/postgres-exporter:latest
    container_name: postgres-exporter
    restart: unless-stopped
    environment:
      DATA_SOURCE_NAME: "postgresql://${POSTGRES_USER:-postgres}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB:-myapp}?sslmode=disable"
    ports:
      - "9187:9187"
    networks:
      - database_network
      - monitoring_network
    depends_on:
      postgres:
        condition: service_healthy

volumes:
  postgres_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /opt/postgres/data

networks:
  database_network:
    driver: bridge
  monitoring_network:
    external: true
```

### PostgreSQL Configuration Files

```bash
# postgres/config/postgresql.conf
# Performance tuning for production
max_connections = 200
shared_buffers = 1GB
effective_cache_size = 3GB
maintenance_work_mem = 256MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200

# Logging configuration
log_destination = 'stderr'
logging_collector = on
log_directory = 'log'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_truncate_on_rotation = on
log_rotation_age = 1d
log_rotation_size = 10MB
log_min_duration_statement = 1000
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '
log_checkpoints = on
log_connections = on
log_disconnections = on
log_lock_waits = on
log_temp_files = 0

# WAL configuration
wal_level = replica
archive_mode = on
archive_command = 'test ! -f /backups/wal/%f && cp %p /backups/wal/%f'
max_wal_senders = 3
checkpoint_timeout = 5min
min_wal_size = 80MB
max_wal_size = 1GB

# Replication settings
hot_standby = on
max_standby_streaming_delay = 30s
wal_receiver_status_interval = 10s
hot_standby_feedback = on
```

```bash
# postgres/config/pg_hba.conf
# TYPE  DATABASE        USER            ADDRESS                 METHOD

# "local" is for Unix domain socket connections only
local   all             all                                     trust

# IPv4 local connections:
host    all             all             127.0.0.1/32            md5
host    all             all             172.16.0.0/12           md5
host    all             all             10.0.0.0/8              md5

# IPv6 local connections:
host    all             all             ::1/128                 md5

# Replication connections
host    replication     replicator      172.16.0.0/12           md5
```

### PostgreSQL Initialization Scripts

```sql
-- postgres/init/01-create-databases.sql
CREATE DATABASE analytics;
CREATE DATABASE metrics;
CREATE DATABASE sessions;

-- Create additional users
CREATE USER analytics_user WITH PASSWORD 'analytics_password';
CREATE USER readonly_user WITH PASSWORD 'readonly_password';
CREATE USER backup_user WITH PASSWORD 'backup_password';

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE analytics TO analytics_user;
GRANT CONNECT ON DATABASE myapp TO readonly_user;
GRANT USAGE ON SCHEMA public TO readonly_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO readonly_user;

-- Create extensions
\c myapp;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

\c analytics;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "timescaledb";
```

## MySQL Deployment

### Production MySQL Configuration

```yaml
# docker-compose.mysql.yml
version: '3.8'

services:
  mysql:
    image: mysql:8.0
    container_name: mysql-prod
    restart: unless-stopped
    environment:
      MYSQL_ROOT_PASSWORD: ${MYSQL_ROOT_PASSWORD}
      MYSQL_DATABASE: ${MYSQL_DATABASE:-myapp}
      MYSQL_USER: ${MYSQL_USER:-myapp_user}
      MYSQL_PASSWORD: ${MYSQL_PASSWORD}
    volumes:
      - mysql_data:/var/lib/mysql
      - ./mysql/config/my.cnf:/etc/mysql/conf.d/custom.cnf:ro
      - ./mysql/init:/docker-entrypoint-initdb.d:ro
      - ./mysql/backups:/backups
    ports:
      - "3306:3306"
    networks:
      - database_network
    command: >
      --character-set-server=utf8mb4
      --collation-server=utf8mb4_unicode_ci
      --skip-character-set-client-handshake
      --max_connections=200
      --max_allowed_packet=256M
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-h", "localhost", "-u", "root", "-p${MYSQL_ROOT_PASSWORD}"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  mysql-exporter:
    image: prom/mysqld-exporter:latest
    container_name: mysql-exporter
    restart: unless-stopped
    environment:
      DATA_SOURCE_NAME: "${MYSQL_USER:-myapp_user}:${MYSQL_PASSWORD}@(mysql:3306)/"
    ports:
      - "9104:9104"
    networks:
      - database_network
      - monitoring_network
    depends_on:
      mysql:
        condition: service_healthy

volumes:
  mysql_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /opt/mysql/data

networks:
  database_network:
    driver: bridge
  monitoring_network:
    external: true
```

### MySQL Configuration

```ini
# mysql/config/my.cnf
[mysqld]
# General settings
user = mysql
default-storage-engine = InnoDB
socket = /var/lib/mysql/mysql.sock
pid-file = /var/lib/mysql/mysql.pid

# Connection settings
max_connections = 200
max_connect_errors = 100000
max_allowed_packet = 256M
interactive_timeout = 600
wait_timeout = 600

# Buffer settings
innodb_buffer_pool_size = 1G
innodb_buffer_pool_instances = 8
innodb_log_file_size = 256M
innodb_log_buffer_size = 16M
innodb_flush_log_at_trx_commit = 2
innodb_flush_method = O_DIRECT

# Query cache
query_cache_type = 1
query_cache_size = 256M
query_cache_limit = 1M

# Logging
general_log = ON
general_log_file = /var/lib/mysql/general.log
slow_query_log = ON
slow_query_log_file = /var/lib/mysql/slow.log
long_query_time = 2
log_queries_not_using_indexes = ON

# Binary logging
log-bin = /var/lib/mysql/mysql-bin
binlog_format = ROW
expire_logs_days = 7
max_binlog_size = 100M
sync_binlog = 1

# Character set
character-set-server = utf8mb4
collation-server = utf8mb4_unicode_ci

[client]
default-character-set = utf8mb4

[mysql]
default-character-set = utf8mb4
```

## MongoDB Deployment

### Production MongoDB Configuration

```yaml
# docker-compose.mongodb.yml
version: '3.8'

services:
  mongodb:
    image: mongo:7.0
    container_name: mongodb-prod
    restart: unless-stopped
    environment:
      MONGO_INITDB_ROOT_USERNAME: ${MONGO_ROOT_USERNAME:-admin}
      MONGO_INITDB_ROOT_PASSWORD: ${MONGO_ROOT_PASSWORD}
      MONGO_INITDB_DATABASE: ${MONGO_DATABASE:-myapp}
    volumes:
      - mongodb_data:/data/db
      - mongodb_config:/data/configdb
      - ./mongodb/config/mongod.conf:/etc/mongod.conf:ro
      - ./mongodb/init:/docker-entrypoint-initdb.d:ro
      - ./mongodb/backups:/backups
    ports:
      - "27017:27017"
    networks:
      - database_network
    command: mongod --config /etc/mongod.conf
    healthcheck:
      test: echo 'db.runCommand("ping").ok' | mongosh localhost:27017/test --quiet
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  mongodb-exporter:
    image: percona/mongodb_exporter:latest
    container_name: mongodb-exporter
    restart: unless-stopped
    environment:
      MONGODB_URI: "mongodb://${MONGO_ROOT_USERNAME:-admin}:${MONGO_ROOT_PASSWORD}@mongodb:27017"
    ports:
      - "9216:9216"
    networks:
      - database_network
      - monitoring_network
    depends_on:
      mongodb:
        condition: service_healthy

volumes:
  mongodb_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /opt/mongodb/data
  mongodb_config:
    driver: local

networks:
  database_network:
    driver: bridge
  monitoring_network:
    external: true
```

### MongoDB Configuration

```yaml
# mongodb/config/mongod.conf
storage:
  dbPath: /data/db
  journal:
    enabled: true
  wiredTiger:
    engineConfig:
      journalCompressor: snappy
      directoryForIndexes: false
    collectionConfig:
      blockCompressor: snappy
    indexConfig:
      prefixCompression: true

systemLog:
  destination: file
  logAppend: true
  path: /var/log/mongodb/mongod.log
  logRotate: rename
  verbosity: 1

net:
  port: 27017
  bindIp: 0.0.0.0
  maxIncomingConnections: 200

processManagement:
  timeZoneInfo: /usr/share/zoneinfo

security:
  authorization: enabled

operationProfiling:
  slowOpThresholdMs: 100
  mode: slowOp

replication:
  replSetName: "rs0"
```

## Redis Deployment

### Production Redis Configuration

```yaml
# docker-compose.redis.yml
version: '3.8'

services:
  redis-master:
    image: redis:7-alpine
    container_name: redis-master
    restart: unless-stopped
    command: >
      redis-server
      --appendonly yes
      --appendfsync everysec
      --save 900 1
      --save 300 10
      --save 60 10000
      --maxmemory 1gb
      --maxmemory-policy allkeys-lru
      --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_master_data:/data
      - ./redis/config/redis.conf:/usr/local/etc/redis/redis.conf:ro
    ports:
      - "6379:6379"
    networks:
      - database_network
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "${REDIS_PASSWORD}", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  redis-replica:
    image: redis:7-alpine
    container_name: redis-replica
    restart: unless-stopped
    command: >
      redis-server
      --slaveof redis-master 6379
      --masterauth ${REDIS_PASSWORD}
      --requirepass ${REDIS_PASSWORD}
      --slave-read-only yes
    volumes:
      - redis_replica_data:/data
    ports:
      - "6380:6379"
    networks:
      - database_network
    depends_on:
      redis-master:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "${REDIS_PASSWORD}", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis-sentinel:
    image: redis:7-alpine
    container_name: redis-sentinel
    restart: unless-stopped
    command: >
      redis-sentinel /usr/local/etc/redis/sentinel.conf
    volumes:
      - ./redis/config/sentinel.conf:/usr/local/etc/redis/sentinel.conf:ro
    ports:
      - "26379:26379"
    networks:
      - database_network
    depends_on:
      - redis-master
      - redis-replica

  redis-exporter:
    image: oliver006/redis_exporter:latest
    container_name: redis-exporter
    restart: unless-stopped
    environment:
      REDIS_ADDR: "redis://redis-master:6379"
      REDIS_PASSWORD: "${REDIS_PASSWORD}"
    ports:
      - "9121:9121"
    networks:
      - database_network
      - monitoring_network
    depends_on:
      redis-master:
        condition: service_healthy

volumes:
  redis_master_data:
    driver: local
  redis_replica_data:
    driver: local

networks:
  database_network:
    driver: bridge
  monitoring_network:
    external: true
```

## Analytics Databases

### ClickHouse Deployment

```yaml
# docker-compose.clickhouse.yml
version: '3.8'

services:
  clickhouse:
    image: clickhouse/clickhouse-server:latest
    container_name: clickhouse-prod
    restart: unless-stopped
    environment:
      CLICKHOUSE_DB: ${CLICKHOUSE_DB:-analytics}
      CLICKHOUSE_USER: ${CLICKHOUSE_USER:-clickhouse}
      CLICKHOUSE_PASSWORD: ${CLICKHOUSE_PASSWORD}
      CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT: 1
    volumes:
      - clickhouse_data:/var/lib/clickhouse
      - clickhouse_logs:/var/log/clickhouse-server
      - ./clickhouse/config:/etc/clickhouse-server/config.d:ro
      - ./clickhouse/init:/docker-entrypoint-initdb.d:ro
    ports:
      - "8123:8123"  # HTTP
      - "9000:9000"  # Native
    networks:
      - database_network
    ulimits:
      nofile:
        soft: 65536
        hard: 65536
    healthcheck:
      test: ["CMD", "clickhouse", "client", "--query", "SELECT 1"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

  clickhouse-exporter:
    image: vertamedia/clickhouse-exporter:latest
    container_name: clickhouse-exporter
    restart: unless-stopped
    environment:
      CLICKHOUSE_URL: "http://clickhouse:8123"
      CLICKHOUSE_USER: "${CLICKHOUSE_USER:-clickhouse}"
      CLICKHOUSE_PASSWORD: "${CLICKHOUSE_PASSWORD}"
    ports:
      - "9116:9116"
    networks:
      - database_network
      - monitoring_network
    depends_on:
      clickhouse:
        condition: service_healthy

volumes:
  clickhouse_data:
    driver: local
  clickhouse_logs:
    driver: local

networks:
  database_network:
    driver: bridge
  monitoring_network:
    external: true
```

### TimescaleDB Deployment

```yaml
# docker-compose.timescaledb.yml
version: '3.8'

services:
  timescaledb:
    image: timescale/timescaledb:latest-pg15
    container_name: timescaledb-prod
    restart: unless-stopped
    environment:
      POSTGRES_DB: ${TIMESCALE_DB:-timeseries}
      POSTGRES_USER: ${TIMESCALE_USER:-timescale}
      POSTGRES_PASSWORD: ${TIMESCALE_PASSWORD}
      PGDATA: /var/lib/postgresql/data/pgdata
    volumes:
      - timescaledb_data:/var/lib/postgresql/data
      - ./timescaledb/init:/docker-entrypoint-initdb.d:ro
      - ./timescaledb/config/postgresql.conf:/etc/postgresql/postgresql.conf:ro
    ports:
      - "5433:5432"
    networks:
      - database_network
    command: >
      postgres
      -c config_file=/etc/postgresql/postgresql.conf
      -c shared_preload_libraries=timescaledb
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${TIMESCALE_USER:-timescale} -d ${TIMESCALE_DB:-timeseries}"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

volumes:
  timescaledb_data:
    driver: local

networks:
  database_network:
    driver: bridge
```

## Backup & Recovery

### Automated Backup Scripts

```bash
#!/bin/bash
# scripts/backup-databases.sh

set -euo pipefail

# Configuration
BACKUP_DIR="/opt/backups"
RETENTION_DAYS=30
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p "$BACKUP_DIR"/{postgres,mysql,mongodb,redis}

# PostgreSQL Backup
echo "Backing up PostgreSQL..."
docker exec postgres-prod pg_dump -U postgres -d myapp | gzip > "$BACKUP_DIR/postgres/myapp_$TIMESTAMP.sql.gz"

# MySQL Backup
echo "Backing up MySQL..."
docker exec mysql-prod mysqldump -u root -p"$MYSQL_ROOT_PASSWORD" --all-databases | gzip > "$BACKUP_DIR/mysql/all_databases_$TIMESTAMP.sql.gz"

# MongoDB Backup
echo "Backing up MongoDB..."
docker exec mongodb-prod mongodump --username admin --password "$MONGO_ROOT_PASSWORD" --authenticationDatabase admin --gzip --archive | gzip > "$BACKUP_DIR/mongodb/mongodump_$TIMESTAMP.gz"

# Redis Backup
echo "Backing up Redis..."
docker exec redis-master redis-cli -a "$REDIS_PASSWORD" --rdb - > "$BACKUP_DIR/redis/dump_$TIMESTAMP.rdb"

# Cleanup old backups
echo "Cleaning up old backups..."
find "$BACKUP_DIR" -type f -mtime +$RETENTION_DAYS -delete

# Upload to S3 (optional)
if [[ -n "${AWS_S3_BUCKET:-}" ]]; then
    echo "Uploading to S3..."
    aws s3 sync "$BACKUP_DIR" "s3://$AWS_S3_BUCKET/database-backups/"
fi

echo "Backup completed successfully!"
```

### Recovery Scripts

```bash
#!/bin/bash
# scripts/restore-database.sh

set -euo pipefail

DB_TYPE="$1"
BACKUP_FILE="$2"

case "$DB_TYPE" in
    "postgres")
        echo "Restoring PostgreSQL from $BACKUP_FILE..."
        gunzip -c "$BACKUP_FILE" | docker exec -i postgres-prod psql -U postgres -d myapp
        ;;
    "mysql")
        echo "Restoring MySQL from $BACKUP_FILE..."
        gunzip -c "$BACKUP_FILE" | docker exec -i mysql-prod mysql -u root -p"$MYSQL_ROOT_PASSWORD"
        ;;
    "mongodb")
        echo "Restoring MongoDB from $BACKUP_FILE..."
        gunzip -c "$BACKUP_FILE" | docker exec -i mongodb-prod mongorestore --username admin --password "$MONGO_ROOT_PASSWORD" --authenticationDatabase admin --gzip --archive
        ;;
    "redis")
        echo "Restoring Redis from $BACKUP_FILE..."
        docker stop redis-master
        docker cp "$BACKUP_FILE" redis-master:/data/dump.rdb
        docker start redis-master
        ;;
    *)
        echo "Unknown database type: $DB_TYPE"
        exit 1
        ;;
esac

echo "Restore completed successfully!"
```

---

*This comprehensive Docker database deployment guide provides production-ready configurations for all major database systems with monitoring, backup, and security best practices.* 