# Monitoring and Observability

## Overview

Comprehensive monitoring and observability framework for data pipelines, ML systems, and infrastructure. Includes metrics collection, log aggregation, alerting, and dashboard creation for production environments.

## Table of Contents

- [System Monitoring](#system-monitoring)
- [Application Performance Monitoring](#application-performance-monitoring)
- [Log Management](#log-management)
- [Alerting Systems](#alerting-systems)
- [Dashboard Creation](#dashboard-creation)
- [Infrastructure Monitoring](#infrastructure-monitoring)
- [ML Model Monitoring](#ml-model-monitoring)
- [Best Practices](#best-practices)

## System Monitoring

### Metrics Collection Framework

```python
import psutil
import time
import json
from datetime import datetime
from typing import Dict, List, Any
import sqlite3
import requests
from threading import Thread

class SystemMetricsCollector:
    """Comprehensive system metrics collection"""
    
    def __init__(self, db_path: str = "metrics.db"):
        self.db_path = db_path
        self.collecting = False
        self._init_database()
    
    def _init_database(self):
        """Initialize metrics database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                metric_type TEXT,
                metric_name TEXT,
                value REAL,
                unit TEXT,
                tags TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def collect_cpu_metrics(self) -> Dict[str, Any]:
        """Collect CPU metrics"""
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        cpu_count = psutil.cpu_count()
        load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
        
        return {
            'cpu_percent_total': psutil.cpu_percent(interval=1),
            'cpu_percent_per_core': cpu_percent,
            'cpu_count_logical': cpu_count,
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'load_avg_1m': load_avg[0],
            'load_avg_5m': load_avg[1],
            'load_avg_15m': load_avg[2]
        }
    
    def collect_memory_metrics(self) -> Dict[str, Any]:
        """Collect memory metrics"""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'memory_total': memory.total,
            'memory_used': memory.used,
            'memory_available': memory.available,
            'memory_percent': memory.percent,
            'swap_total': swap.total,
            'swap_used': swap.used,
            'swap_percent': swap.percent
        }
    
    def collect_disk_metrics(self) -> Dict[str, Any]:
        """Collect disk metrics"""
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        metrics = {
            'disk_total': disk_usage.total,
            'disk_used': disk_usage.used,
            'disk_free': disk_usage.free,
            'disk_percent': (disk_usage.used / disk_usage.total) * 100
        }
        
        if disk_io:
            metrics.update({
                'disk_read_bytes': disk_io.read_bytes,
                'disk_write_bytes': disk_io.write_bytes,
                'disk_read_count': disk_io.read_count,
                'disk_write_count': disk_io.write_count
            })
        
        return metrics
    
    def collect_network_metrics(self) -> Dict[str, Any]:
        """Collect network metrics"""
        net_io = psutil.net_io_counters()
        net_connections = len(psutil.net_connections())
        
        return {
            'network_bytes_sent': net_io.bytes_sent,
            'network_bytes_recv': net_io.bytes_recv,
            'network_packets_sent': net_io.packets_sent,
            'network_packets_recv': net_io.packets_recv,
            'network_connections': net_connections
        }
    
    def collect_process_metrics(self) -> Dict[str, Any]:
        """Collect process-level metrics"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Top CPU and memory consumers
        top_cpu = sorted(processes, key=lambda x: x['cpu_percent'] or 0, reverse=True)[:5]
        top_memory = sorted(processes, key=lambda x: x['memory_percent'] or 0, reverse=True)[:5]
        
        return {
            'total_processes': len(processes),
            'top_cpu_processes': top_cpu,
            'top_memory_processes': top_memory
        }
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all system metrics"""
        timestamp = datetime.now()
        
        metrics = {
            'timestamp': timestamp,
            'cpu': self.collect_cpu_metrics(),
            'memory': self.collect_memory_metrics(),
            'disk': self.collect_disk_metrics(),
            'network': self.collect_network_metrics(),
            'processes': self.collect_process_metrics()
        }
        
        return metrics
    
    def store_metrics(self, metrics: Dict[str, Any]):
        """Store metrics in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = metrics['timestamp']
        
        # Store CPU metrics
        for key, value in metrics['cpu'].items():
            if isinstance(value, list):
                for i, v in enumerate(value):
                    cursor.execute('''
                        INSERT INTO system_metrics 
                        (timestamp, metric_type, metric_name, value, unit, tags)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (timestamp, 'cpu', key, v, 'percent', json.dumps({'core': i})))
            else:
                cursor.execute('''
                    INSERT INTO system_metrics 
                    (timestamp, metric_type, metric_name, value, unit, tags)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (timestamp, 'cpu', key, value, 'percent' if 'percent' in key else 'count', '{}'))
        
        # Store memory metrics
        for key, value in metrics['memory'].items():
            unit = 'bytes' if 'total' in key or 'used' in key or 'available' in key else 'percent'
            cursor.execute('''
                INSERT INTO system_metrics 
                (timestamp, metric_type, metric_name, value, unit, tags)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (timestamp, 'memory', key, value, unit, '{}'))
        
        # Store disk metrics
        for key, value in metrics['disk'].items():
            unit = 'bytes' if any(x in key for x in ['total', 'used', 'free', 'read_bytes', 'write_bytes']) else 'count'
            if 'percent' in key:
                unit = 'percent'
            cursor.execute('''
                INSERT INTO system_metrics 
                (timestamp, metric_type, metric_name, value, unit, tags)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (timestamp, 'disk', key, value, unit, '{}'))
        
        # Store network metrics
        for key, value in metrics['network'].items():
            unit = 'bytes' if 'bytes' in key else 'count'
            cursor.execute('''
                INSERT INTO system_metrics 
                (timestamp, metric_type, metric_name, value, unit, tags)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (timestamp, 'network', key, value, unit, '{}'))
        
        conn.commit()
        conn.close()
    
    def start_collection(self, interval: int = 60):
        """Start continuous metrics collection"""
        self.collecting = True
        
        def collection_loop():
            while self.collecting:
                try:
                    metrics = self.collect_all_metrics()
                    self.store_metrics(metrics)
                    time.sleep(interval)
                except Exception as e:
                    print(f"Error collecting metrics: {e}")
                    time.sleep(interval)
        
        collection_thread = Thread(target=collection_loop)
        collection_thread.daemon = True
        collection_thread.start()
    
    def stop_collection(self):
        """Stop metrics collection"""
        self.collecting = False
```

### Custom Metrics Framework

```python
class CustomMetricsCollector:
    """Custom application metrics collector"""
    
    def __init__(self, db_path: str = "app_metrics.db"):
        self.db_path = db_path
        self.metrics_buffer = []
        self._init_database()
    
    def _init_database(self):
        """Initialize application metrics database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS app_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                metric_name TEXT,
                metric_value REAL,
                metric_type TEXT,
                tags TEXT,
                service_name TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def counter(self, name: str, value: float = 1, tags: Dict = None, service: str = "default"):
        """Record counter metric"""
        self._record_metric(name, value, "counter", tags, service)
    
    def gauge(self, name: str, value: float, tags: Dict = None, service: str = "default"):
        """Record gauge metric"""
        self._record_metric(name, value, "gauge", tags, service)
    
    def histogram(self, name: str, value: float, tags: Dict = None, service: str = "default"):
        """Record histogram metric"""
        self._record_metric(name, value, "histogram", tags, service)
    
    def timer(self, name: str, duration: float, tags: Dict = None, service: str = "default"):
        """Record timer metric"""
        self._record_metric(name, duration, "timer", tags, service)
    
    def _record_metric(self, name: str, value: float, metric_type: str, 
                      tags: Dict = None, service: str = "default"):
        """Record metric to buffer"""
        metric = {
            'timestamp': datetime.now(),
            'name': name,
            'value': value,
            'type': metric_type,
            'tags': json.dumps(tags or {}),
            'service': service
        }
        
        self.metrics_buffer.append(metric)
        
        # Flush buffer if it gets too large
        if len(self.metrics_buffer) >= 100:
            self.flush_metrics()
    
    def flush_metrics(self):
        """Flush metrics buffer to database"""
        if not self.metrics_buffer:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for metric in self.metrics_buffer:
            cursor.execute('''
                INSERT INTO app_metrics 
                (timestamp, metric_name, metric_value, metric_type, tags, service_name)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                metric['timestamp'],
                metric['name'],
                metric['value'],
                metric['type'],
                metric['tags'],
                metric['service']
            ))
        
        conn.commit()
        conn.close()
        
        self.metrics_buffer.clear()
    
    def get_metrics(self, name: str = None, service: str = None, 
                   hours: int = 24) -> List[Dict]:
        """Retrieve metrics from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT * FROM app_metrics 
            WHERE timestamp > datetime('now', '-{} hours')
        '''.format(hours)
        
        params = []
        if name:
            query += " AND metric_name = ?"
            params.append(name)
        if service:
            query += " AND service_name = ?"
            params.append(service)
        
        query += " ORDER BY timestamp DESC"
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        return [dict(zip([col[0] for col in cursor.description], row)) for row in results]
```

## Application Performance Monitoring

### Performance Monitoring Decorator

```python
import functools
import time
import traceback
from typing import Callable, Any

class PerformanceMonitor:
    """Application performance monitoring with decorators"""
    
    def __init__(self, metrics_collector: CustomMetricsCollector):
        self.metrics = metrics_collector
    
    def monitor_function(self, function_name: str = None, 
                        service_name: str = "default"):
        """Decorator to monitor function performance"""
        def decorator(func: Callable) -> Callable:
            name = function_name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                start_time = time.time()
                
                try:
                    # Record function call
                    self.metrics.counter(f"{name}.calls", tags={'service': service_name})
                    
                    result = func(*args, **kwargs)
                    
                    # Record success
                    self.metrics.counter(f"{name}.success", tags={'service': service_name})
                    
                    return result
                
                except Exception as e:
                    # Record error
                    self.metrics.counter(f"{name}.errors", 
                                       tags={'service': service_name, 'error_type': type(e).__name__})
                    raise
                
                finally:
                    # Record execution time
                    duration = time.time() - start_time
                    self.metrics.timer(f"{name}.duration", duration, 
                                     tags={'service': service_name})
                
            return wrapper
        return decorator
    
    def monitor_class(self, class_name: str = None, service_name: str = "default"):
        """Class decorator to monitor all methods"""
        def decorator(cls):
            name = class_name or cls.__name__
            
            for attr_name in dir(cls):
                attr = getattr(cls, attr_name)
                if callable(attr) and not attr_name.startswith('_'):
                    monitored_method = self.monitor_function(
                        f"{name}.{attr_name}", service_name
                    )(attr)
                    setattr(cls, attr_name, monitored_method)
            
            return cls
        return decorator

# Usage example
metrics_collector = CustomMetricsCollector()
monitor = PerformanceMonitor(metrics_collector)

@monitor.monitor_function("data_processing", "etl_service")
def process_data(data):
    # Simulate data processing
    time.sleep(0.1)
    return len(data) * 2

@monitor.monitor_class("DataProcessor", "etl_service")
class DataProcessor:
    def transform(self, data):
        return data.upper()
    
    def validate(self, data):
        return len(data) > 0
```

### Health Check System

```python
from enum import Enum
from typing import Dict, List, Optional
import requests
import socket

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class HealthCheck:
    """Individual health check"""
    
    def __init__(self, name: str, check_func: Callable, 
                 timeout: int = 30, critical: bool = False):
        self.name = name
        self.check_func = check_func
        self.timeout = timeout
        self.critical = critical
    
    def execute(self) -> Dict[str, Any]:
        """Execute health check"""
        start_time = time.time()
        
        try:
            result = self.check_func()
            duration = time.time() - start_time
            
            return {
                'name': self.name,
                'status': HealthStatus.HEALTHY,
                'duration': duration,
                'details': result,
                'critical': self.critical
            }
        
        except Exception as e:
            duration = time.time() - start_time
            
            return {
                'name': self.name,
                'status': HealthStatus.UNHEALTHY,
                'duration': duration,
                'error': str(e),
                'critical': self.critical
            }

class HealthCheckManager:
    """Manage multiple health checks"""
    
    def __init__(self):
        self.checks = {}
    
    def add_check(self, health_check: HealthCheck):
        """Add health check"""
        self.checks[health_check.name] = health_check
    
    def add_database_check(self, name: str, connection_string: str):
        """Add database connectivity check"""
        def check_database():
            # This is a simplified example
            # In practice, use appropriate database client
            try:
                # Simulate database check
                time.sleep(0.01)
                return {"connection": "ok", "query_time": "10ms"}
            except Exception:
                raise Exception("Database connection failed")
        
        self.add_check(HealthCheck(name, check_database, critical=True))
    
    def add_service_check(self, name: str, url: str):
        """Add external service check"""
        def check_service():
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return {
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds()
            }
        
        self.add_check(HealthCheck(name, check_service))
    
    def add_disk_space_check(self, name: str, threshold: float = 0.8):
        """Add disk space check"""
        def check_disk():
            disk_usage = psutil.disk_usage('/')
            usage_percent = disk_usage.used / disk_usage.total
            
            if usage_percent > threshold:
                raise Exception(f"Disk usage too high: {usage_percent:.1%}")
            
            return {
                "usage_percent": usage_percent,
                "free_bytes": disk_usage.free
            }
        
        self.add_check(HealthCheck(name, check_disk, critical=True))
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = []
        overall_status = HealthStatus.HEALTHY
        
        for check in self.checks.values():
            result = check.execute()
            results.append(result)
            
            # Determine overall status
            if result['status'] == HealthStatus.UNHEALTHY:
                if result['critical']:
                    overall_status = HealthStatus.UNHEALTHY
                elif overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
        
        return {
            'overall_status': overall_status.value,
            'timestamp': datetime.now(),
            'checks': results,
            'summary': {
                'total': len(results),
                'healthy': len([r for r in results if r['status'] == HealthStatus.HEALTHY]),
                'unhealthy': len([r for r in results if r['status'] == HealthStatus.UNHEALTHY])
            }
        }
```

## Log Management

### Structured Logging

```python
import logging
import json
from datetime import datetime
from typing import Dict, Any

class StructuredLogger:
    """Structured logging for better observability"""
    
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Create structured formatter
        formatter = StructuredFormatter()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler('application.log')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message with structured data"""
        self._log(logging.INFO, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with structured data"""
        self._log(logging.ERROR, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with structured data"""
        self._log(logging.WARNING, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with structured data"""
        self._log(logging.DEBUG, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs):
        """Internal logging method"""
        extra = {
            'structured_data': kwargs,
            'timestamp': datetime.now().isoformat()
        }
        self.logger.log(level, message, extra=extra)

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logs"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': getattr(record, 'timestamp', datetime.now().isoformat()),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add structured data if present
        if hasattr(record, 'structured_data'):
            log_data.update(record.structured_data)
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)

# Usage example
logger = StructuredLogger('data_pipeline')

logger.info("Processing started", 
           batch_id="batch_123", 
           record_count=1000,
           source="database")

logger.error("Processing failed", 
            batch_id="batch_123",
            error_code="DB_CONNECTION_ERROR",
            retry_count=3)
```

### Log Aggregation

```python
import gzip
import os
from pathlib import Path
from typing import Generator

class LogAggregator:
    """Aggregate and analyze logs"""
    
    def __init__(self, log_directory: str):
        self.log_directory = Path(log_directory)
    
    def parse_log_file(self, file_path: Path) -> Generator[Dict, None, None]:
        """Parse structured log file"""
        try:
            if file_path.suffix == '.gz':
                file_obj = gzip.open(file_path, 'rt')
            else:
                file_obj = open(file_path, 'r')
            
            with file_obj as f:
                for line in f:
                    try:
                        yield json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
    
    def aggregate_errors(self, hours: int = 24) -> Dict[str, Any]:
        """Aggregate error logs"""
        errors = []
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        for log_file in self.log_directory.glob("*.log*"):
            for log_entry in self.parse_log_file(log_file):
                if log_entry.get('level') == 'ERROR':
                    log_time = datetime.fromisoformat(log_entry['timestamp'])
                    if log_time > cutoff_time:
                        errors.append(log_entry)
        
        # Group by error type
        error_summary = {}
        for error in errors:
            error_type = error.get('error_code', 'UNKNOWN')
            if error_type not in error_summary:
                error_summary[error_type] = {
                    'count': 0,
                    'first_seen': error['timestamp'],
                    'last_seen': error['timestamp'],
                    'examples': []
                }
            
            error_summary[error_type]['count'] += 1
            error_summary[error_type]['last_seen'] = error['timestamp']
            
            if len(error_summary[error_type]['examples']) < 3:
                error_summary[error_type]['examples'].append(error['message'])
        
        return {
            'total_errors': len(errors),
            'error_types': len(error_summary),
            'summary': error_summary
        }
    
    def get_performance_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Extract performance metrics from logs"""
        durations = []
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        for log_file in self.log_directory.glob("*.log*"):
            for log_entry in self.parse_log_file(log_file):
                if 'duration' in log_entry:
                    log_time = datetime.fromisoformat(log_entry['timestamp'])
                    if log_time > cutoff_time:
                        durations.append(log_entry['duration'])
        
        if not durations:
            return {'message': 'No performance data found'}
        
        durations.sort()
        n = len(durations)
        
        return {
            'count': n,
            'min': min(durations),
            'max': max(durations),
            'mean': sum(durations) / n,
            'median': durations[n // 2],
            'p95': durations[int(n * 0.95)],
            'p99': durations[int(n * 0.99)]
        }
```

## Alerting Systems

### Alert Manager

```python
from enum import Enum
from dataclasses import dataclass
from typing import List, Callable, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Alert:
    name: str
    message: str
    severity: AlertSeverity
    timestamp: datetime
    tags: Dict[str, str]
    resolved: bool = False
    acknowledged: bool = False

class AlertManager:
    """Manage alerts and notifications"""
    
    def __init__(self):
        self.alerts = []
        self.notification_channels = []
        self.alert_rules = []
    
    def add_notification_channel(self, channel: 'NotificationChannel'):
        """Add notification channel"""
        self.notification_channels.append(channel)
    
    def add_alert_rule(self, rule: 'AlertRule'):
        """Add alert rule"""
        self.alert_rules.append(rule)
    
    def create_alert(self, name: str, message: str, severity: AlertSeverity,
                    tags: Dict[str, str] = None) -> Alert:
        """Create new alert"""
        alert = Alert(
            name=name,
            message=message,
            severity=severity,
            timestamp=datetime.now(),
            tags=tags or {}
        )
        
        self.alerts.append(alert)
        self._send_notifications(alert)
        
        return alert
    
    def resolve_alert(self, alert_name: str):
        """Resolve alert by name"""
        for alert in self.alerts:
            if alert.name == alert_name and not alert.resolved:
                alert.resolved = True
                self._send_resolution_notifications(alert)
    
    def check_alert_rules(self, metrics: Dict[str, Any]):
        """Check all alert rules against metrics"""
        for rule in self.alert_rules:
            if rule.should_trigger(metrics):
                # Check if alert already exists
                existing_alert = next(
                    (a for a in self.alerts 
                     if a.name == rule.name and not a.resolved), 
                    None
                )
                
                if not existing_alert:
                    self.create_alert(
                        rule.name,
                        rule.get_message(metrics),
                        rule.severity,
                        rule.tags
                    )
    
    def _send_notifications(self, alert: Alert):
        """Send alert notifications"""
        for channel in self.notification_channels:
            if channel.should_notify(alert):
                try:
                    channel.send_alert(alert)
                except Exception as e:
                    print(f"Failed to send notification via {channel}: {e}")
    
    def _send_resolution_notifications(self, alert: Alert):
        """Send alert resolution notifications"""
        for channel in self.notification_channels:
            if channel.should_notify(alert):
                try:
                    channel.send_resolution(alert)
                except Exception as e:
                    print(f"Failed to send resolution via {channel}: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return [a for a in self.alerts if not a.resolved]

class AlertRule:
    """Alert rule definition"""
    
    def __init__(self, name: str, condition: Callable[[Dict], bool],
                 severity: AlertSeverity, message_template: str,
                 tags: Dict[str, str] = None):
        self.name = name
        self.condition = condition
        self.severity = severity
        self.message_template = message_template
        self.tags = tags or {}
    
    def should_trigger(self, metrics: Dict[str, Any]) -> bool:
        """Check if alert should trigger"""
        try:
            return self.condition(metrics)
        except Exception:
            return False
    
    def get_message(self, metrics: Dict[str, Any]) -> str:
        """Generate alert message"""
        try:
            return self.message_template.format(**metrics)
        except Exception:
            return self.message_template

class NotificationChannel:
    """Base notification channel"""
    
    def should_notify(self, alert: Alert) -> bool:
        """Check if this channel should notify for the alert"""
        return True
    
    def send_alert(self, alert: Alert):
        """Send alert notification"""
        raise NotImplementedError
    
    def send_resolution(self, alert: Alert):
        """Send resolution notification"""
        raise NotImplementedError

class EmailNotificationChannel(NotificationChannel):
    """Email notification channel"""
    
    def __init__(self, smtp_server: str, smtp_port: int,
                 username: str, password: str, recipients: List[str],
                 min_severity: AlertSeverity = AlertSeverity.MEDIUM):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.recipients = recipients
        self.min_severity = min_severity
    
    def should_notify(self, alert: Alert) -> bool:
        """Check if should notify based on severity"""
        severity_order = [AlertSeverity.LOW, AlertSeverity.MEDIUM, 
                         AlertSeverity.HIGH, AlertSeverity.CRITICAL]
        return severity_order.index(alert.severity) >= severity_order.index(self.min_severity)
    
    def send_alert(self, alert: Alert):
        """Send alert email"""
        subject = f"[{alert.severity.value.upper()}] {alert.name}"
        body = f"""
Alert: {alert.name}
Severity: {alert.severity.value}
Time: {alert.timestamp}
Message: {alert.message}

Tags: {alert.tags}
        """
        
        self._send_email(subject, body)
    
    def send_resolution(self, alert: Alert):
        """Send resolution email"""
        subject = f"[RESOLVED] {alert.name}"
        body = f"""
Alert Resolved: {alert.name}
Original Time: {alert.timestamp}
Resolution Time: {datetime.now()}
        """
        
        self._send_email(subject, body)
    
    def _send_email(self, subject: str, body: str):
        """Send email using SMTP"""
        msg = MIMEMultipart()
        msg['From'] = self.username
        msg['To'] = ', '.join(self.recipients)
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()
        except Exception as e:
            print(f"Failed to send email: {e}")
```

## Usage Examples

### Complete Monitoring Setup

```python
def setup_complete_monitoring():
    """Setup complete monitoring system"""
    
    # Initialize components
    system_metrics = SystemMetricsCollector()
    app_metrics = CustomMetricsCollector()
    performance_monitor = PerformanceMonitor(app_metrics)
    health_checks = HealthCheckManager()
    alert_manager = AlertManager()
    
    # Setup health checks
    health_checks.add_database_check("main_db", "postgresql://localhost:5432/app")
    health_checks.add_service_check("api_service", "http://localhost:8080/health")
    health_checks.add_disk_space_check("disk_space", threshold=0.85)
    
    # Setup alert rules
    cpu_alert = AlertRule(
        name="high_cpu_usage",
        condition=lambda m: m.get('cpu_percent_total', 0) > 80,
        severity=AlertSeverity.HIGH,
        message_template="CPU usage is {cpu_percent_total:.1f}%"
    )
    alert_manager.add_alert_rule(cpu_alert)
    
    memory_alert = AlertRule(
        name="high_memory_usage",
        condition=lambda m: m.get('memory_percent', 0) > 85,
        severity=AlertSeverity.CRITICAL,
        message_template="Memory usage is {memory_percent:.1f}%"
    )
    alert_manager.add_alert_rule(memory_alert)
    
    # Setup notification channels
    email_channel = EmailNotificationChannel(
        smtp_server="smtp.example.com",
        smtp_port=587,
        username="alerts@example.com",
        password="password",
        recipients=["admin@example.com"],
        min_severity=AlertSeverity.MEDIUM
    )
    alert_manager.add_notification_channel(email_channel)
    
    # Start monitoring
    system_metrics.start_collection(interval=30)
    
    # Monitoring loop
    def monitoring_loop():
        while True:
            try:
                # Collect current metrics
                current_metrics = system_metrics.collect_all_metrics()
                
                # Check health
                health_status = health_checks.run_all_checks()
                
                # Check alert rules
                flat_metrics = {}
                for category, metrics in current_metrics.items():
                    if isinstance(metrics, dict):
                        for key, value in metrics.items():
                            if isinstance(value, (int, float)):
                                flat_metrics[key] = value
                
                alert_manager.check_alert_rules(flat_metrics)
                
                # Log status
                logger = StructuredLogger('monitoring')
                logger.info("Monitoring check completed",
                           health_status=health_status['overall_status'],
                           active_alerts=len(alert_manager.get_active_alerts()),
                           cpu_usage=flat_metrics.get('cpu_percent_total', 0))
                
                time.sleep(60)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(60)
    
    # Start monitoring in background
    monitor_thread = Thread(target=monitoring_loop)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    return {
        'system_metrics': system_metrics,
        'app_metrics': app_metrics,
        'health_checks': health_checks,
        'alert_manager': alert_manager,
        'performance_monitor': performance_monitor
    }

if __name__ == "__main__":
    monitoring_system = setup_complete_monitoring()
    
    # Example of using monitored function
    @monitoring_system['performance_monitor'].monitor_function("example_task")
    def example_task():
        time.sleep(1)
        return "completed"
    
    # Run example
    result = example_task()
    
    # Keep monitoring running
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("Stopping monitoring...")
        monitoring_system['system_metrics'].stop_collection()
```

---

*This comprehensive monitoring framework provides production-ready observability for your data infrastructure and applications.* 