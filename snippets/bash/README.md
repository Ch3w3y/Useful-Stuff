# Bash Code Snippets Collection

## Overview

Comprehensive collection of Bash code snippets for system administration, automation, file operations, text processing, and development workflows.

## Table of Contents

- [File Operations](#file-operations)
- [Text Processing](#text-processing)
- [System Administration](#system-administration)
- [Network Operations](#network-operations)
- [Process Management](#process-management)
- [Data Processing](#data-processing)
- [Automation Scripts](#automation-scripts)
- [Development Utilities](#development-utilities)

## File Operations

### File Management and Navigation

```bash
#!/bin/bash

# Find files by extension
find_by_extension() {
    local extension="$1"
    local directory="${2:-.}"
    find "$directory" -type f -name "*.$extension" 2>/dev/null
}

# Find large files
find_large_files() {
    local size="${1:-100M}"
    local directory="${2:-.}"
    find "$directory" -type f -size +"$size" -exec ls -lh {} \; | awk '{ print $9 ": " $5 }'
}

# Bulk rename files
bulk_rename() {
    local pattern="$1"
    local replacement="$2"
    local directory="${3:-.}"
    
    for file in "$directory"/*"$pattern"*; do
        if [[ -f "$file" ]]; then
            new_name="${file/$pattern/$replacement}"
            mv "$file" "$new_name"
            echo "Renamed: $(basename "$file") -> $(basename "$new_name")"
        fi
    done
}

# Create directory structure
create_project_structure() {
    local project_name="$1"
    
    mkdir -p "$project_name"/{src,tests,docs,config,scripts,data/{raw,processed}}
    touch "$project_name"/{README.md,.gitignore}
    
    echo "Project structure created for: $project_name"
    tree "$project_name" 2>/dev/null || ls -la "$project_name"
}

# Safe file deletion with confirmation
safe_delete() {
    local file="$1"
    
    if [[ ! -e "$file" ]]; then
        echo "Error: File '$file' does not exist"
        return 1
    fi
    
    echo "Are you sure you want to delete '$file'? (y/N)"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        rm -rf "$file"
        echo "Deleted: $file"
    else
        echo "Deletion cancelled"
    fi
}

# Backup files with timestamp
backup_file() {
    local file="$1"
    local backup_dir="${2:-./backups}"
    
    if [[ ! -f "$file" ]]; then
        echo "Error: File '$file' does not exist"
        return 1
    fi
    
    mkdir -p "$backup_dir"
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local backup_name="${backup_dir}/$(basename "$file")_${timestamp}"
    
    cp "$file" "$backup_name"
    echo "Backup created: $backup_name"
}

# Synchronize directories
sync_directories() {
    local source="$1"
    local destination="$2"
    local dry_run="${3:-false}"
    
    if [[ "$dry_run" == "true" ]]; then
        rsync -avhn --delete "$source/" "$destination/"
    else
        rsync -avh --delete "$source/" "$destination/"
        echo "Synchronization complete: $source -> $destination"
    fi
}

# File permissions management
fix_permissions() {
    local directory="$1"
    
    # Set standard permissions
    find "$directory" -type d -exec chmod 755 {} \;
    find "$directory" -type f -exec chmod 644 {} \;
    find "$directory" -name "*.sh" -exec chmod +x {} \;
    
    echo "Permissions fixed for: $directory"
}
```

### Archive and Compression

```bash
# Create compressed archives
create_archive() {
    local source="$1"
    local archive_name="$2"
    local compression="${3:-gz}"
    
    case "$compression" in
        "gz"|"gzip")
            tar -czf "${archive_name}.tar.gz" "$source"
            echo "Created: ${archive_name}.tar.gz"
            ;;
        "bz2"|"bzip2")
            tar -cjf "${archive_name}.tar.bz2" "$source"
            echo "Created: ${archive_name}.tar.bz2"
            ;;
        "xz")
            tar -cJf "${archive_name}.tar.xz" "$source"
            echo "Created: ${archive_name}.tar.xz"
            ;;
        "zip")
            zip -r "${archive_name}.zip" "$source"
            echo "Created: ${archive_name}.zip"
            ;;
        *)
            echo "Unsupported compression format: $compression"
            return 1
            ;;
    esac
}

# Extract archives automatically
extract_archive() {
    local archive="$1"
    local destination="${2:-.}"
    
    if [[ ! -f "$archive" ]]; then
        echo "Error: Archive '$archive' not found"
        return 1
    fi
    
    case "$archive" in
        *.tar.gz|*.tgz)
            tar -xzf "$archive" -C "$destination"
            ;;
        *.tar.bz2|*.tbz2)
            tar -xjf "$archive" -C "$destination"
            ;;
        *.tar.xz|*.txz)
            tar -xJf "$archive" -C "$destination"
            ;;
        *.zip)
            unzip "$archive" -d "$destination"
            ;;
        *.rar)
            unrar x "$archive" "$destination"
            ;;
        *.7z)
            7z x "$archive" -o"$destination"
            ;;
        *)
            echo "Unsupported archive format: $archive"
            return 1
            ;;
    esac
    
    echo "Extracted: $archive to $destination"
}

# Compress files by age
compress_old_files() {
    local directory="$1"
    local days="${2:-30}"
    local archive_dir="${3:-./archives}"
    
    mkdir -p "$archive_dir"
    
    find "$directory" -type f -mtime +"$days" -print0 | while IFS= read -r -d '' file; do
        relative_path="${file#$directory/}"
        archive_path="$archive_dir/$(dirname "$relative_path")"
        mkdir -p "$archive_path"
        
        gzip -c "$file" > "$archive_path/$(basename "$file").gz"
        rm "$file"
        echo "Compressed and removed: $file"
    done
}
```

## Text Processing

### Advanced Text Manipulation

```bash
# CSV processing utilities
csv_column() {
    local file="$1"
    local column="$2"
    local delimiter="${3:-,}"
    
    awk -F"$delimiter" -v col="$column" '{print $col}' "$file"
}

csv_filter() {
    local file="$1"
    local column="$2"
    local value="$3"
    local delimiter="${4:-,}"
    
    awk -F"$delimiter" -v col="$column" -v val="$value" '$col == val' "$file"
}

csv_stats() {
    local file="$1"
    local column="$2"
    local delimiter="${3:-,}"
    
    awk -F"$delimiter" -v col="$column" '
    NR > 1 {
        sum += $col
        count++
        if (NR == 2 || $col < min) min = $col
        if (NR == 2 || $col > max) max = $col
    }
    END {
        if (count > 0) {
            avg = sum / count
            print "Count: " count
            print "Sum: " sum
            print "Average: " avg
            print "Min: " min
            print "Max: " max
        }
    }' "$file"
}

# Log analysis
analyze_logs() {
    local log_file="$1"
    local pattern="${2:-ERROR}"
    
    echo "=== Log Analysis for: $log_file ==="
    echo "Total lines: $(wc -l < "$log_file")"
    echo "Pattern '$pattern' occurrences: $(grep -c "$pattern" "$log_file")"
    echo
    
    echo "=== Top 10 IP addresses ==="
    awk '{print $1}' "$log_file" | sort | uniq -c | sort -nr | head -10
    echo
    
    echo "=== Top 10 requested URLs ==="
    awk '{print $7}' "$log_file" | sort | uniq -c | sort -nr | head -10
    echo
    
    echo "=== Status code distribution ==="
    awk '{print $9}' "$log_file" | sort | uniq -c | sort -nr
}

# Text formatting and cleanup
clean_text() {
    local file="$1"
    local output="${2:-${file}.clean}"
    
    # Remove extra whitespace, empty lines, and normalize line endings
    sed -e 's/[[:space:]]*$//' \
        -e 's/^[[:space:]]*//' \
        -e '/^$/d' \
        "$file" | tr -s ' ' > "$output"
    
    echo "Cleaned text saved to: $output"
}

# Extract emails from text
extract_emails() {
    local file="$1"
    
    grep -oE '\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b' "$file" | sort -u
}

# Extract URLs from text
extract_urls() {
    local file="$1"
    
    grep -oE 'https?://[^[:space:]]+' "$file" | sort -u
}

# Word frequency analysis
word_frequency() {
    local file="$1"
    local top_n="${2:-10}"
    
    tr '[:upper:]' '[:lower:]' < "$file" | \
    tr -d '[:punct:]' | \
    tr ' ' '\n' | \
    grep -v '^$' | \
    sort | \
    uniq -c | \
    sort -nr | \
    head -"$top_n"
}
```

## System Administration

### System Monitoring and Maintenance

```bash
# System health check
system_health_check() {
    echo "=== System Health Check - $(date) ==="
    echo
    
    # CPU usage
    echo "=== CPU Usage ==="
    top -bn1 | grep "Cpu(s)" | awk '{print "CPU Usage: " $2}' | sed 's/%us,//'
    echo
    
    # Memory usage
    echo "=== Memory Usage ==="
    free -h | awk 'NR==2{printf "Memory Usage: %s/%s (%.2f%%)\n", $3,$2,$3*100/$2 }'
    echo
    
    # Disk usage
    echo "=== Disk Usage ==="
    df -h | awk '$NF=="/"{printf "Disk Usage: %d/%dGB (%s)\n", $3,$2,$5}'
    echo
    
    # Load average
    echo "=== Load Average ==="
    uptime | awk -F'load average:' '{print "Load Average:" $2}'
    echo
    
    # Top processes by CPU
    echo "=== Top 5 CPU Processes ==="
    ps aux --sort=-%cpu | head -6
    echo
    
    # Top processes by Memory
    echo "=== Top 5 Memory Processes ==="
    ps aux --sort=-%mem | head -6
}

# Disk cleanup
disk_cleanup() {
    local threshold="${1:-80}"
    
    echo "=== Disk Cleanup - $(date) ==="
    
    # Check disk usage
    disk_usage=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
    
    if [[ $disk_usage -gt $threshold ]]; then
        echo "Disk usage is ${disk_usage}%, starting cleanup..."
        
        # Clean package cache
        if command -v apt-get &> /dev/null; then
            sudo apt-get autoremove -y
            sudo apt-get autoclean
        elif command -v yum &> /dev/null; then
            sudo yum clean all
        fi
        
        # Clean temporary files
        sudo find /tmp -type f -atime +7 -delete
        sudo find /var/tmp -type f -atime +7 -delete
        
        # Clean log files older than 30 days
        sudo find /var/log -name "*.log" -type f -mtime +30 -delete
        
        # Clean user cache
        rm -rf ~/.cache/*
        
        echo "Cleanup completed"
    else
        echo "Disk usage is ${disk_usage}%, no cleanup needed"
    fi
}

# Service management
manage_service() {
    local action="$1"
    local service="$2"
    
    case "$action" in
        "status")
            systemctl status "$service"
            ;;
        "start")
            sudo systemctl start "$service"
            echo "Started service: $service"
            ;;
        "stop")
            sudo systemctl stop "$service"
            echo "Stopped service: $service"
            ;;
        "restart")
            sudo systemctl restart "$service"
            echo "Restarted service: $service"
            ;;
        "enable")
            sudo systemctl enable "$service"
            echo "Enabled service: $service"
            ;;
        "disable")
            sudo systemctl disable "$service"
            echo "Disabled service: $service"
            ;;
        *)
            echo "Usage: manage_service {status|start|stop|restart|enable|disable} <service_name>"
            return 1
            ;;
    esac
}

# User management
create_user_account() {
    local username="$1"
    local groups="${2:-users}"
    
    if id "$username" &>/dev/null; then
        echo "User $username already exists"
        return 1
    fi
    
    sudo useradd -m -s /bin/bash -G "$groups" "$username"
    sudo passwd "$username"
    
    echo "User account created: $username"
    echo "Groups: $groups"
}

# Network diagnostics
network_diagnostics() {
    local host="${1:-google.com}"
    
    echo "=== Network Diagnostics ==="
    echo
    
    # Interface information
    echo "=== Network Interfaces ==="
    ip addr show
    echo
    
    # Routing table
    echo "=== Routing Table ==="
    ip route show
    echo
    
    # DNS resolution
    echo "=== DNS Resolution Test ==="
    nslookup "$host"
    echo
    
    # Connectivity test
    echo "=== Connectivity Test ==="
    ping -c 4 "$host"
    echo
    
    # Port scan
    echo "=== Common Port Check ==="
    for port in 22 80 443 53; do
        if timeout 3 bash -c "</dev/tcp/$host/$port" 2>/dev/null; then
            echo "Port $port: Open"
        else
            echo "Port $port: Closed/Filtered"
        fi
    done
}
```

## Network Operations

### Network Utilities and Monitoring

```bash
# Port scanner
scan_ports() {
    local host="$1"
    local start_port="${2:-1}"
    local end_port="${3:-1000}"
    
    echo "Scanning ports $start_port-$end_port on $host..."
    
    for port in $(seq "$start_port" "$end_port"); do
        if timeout 1 bash -c "</dev/tcp/$host/$port" 2>/dev/null; then
            echo "Port $port: Open"
        fi
    done
}

# Download with retry
download_with_retry() {
    local url="$1"
    local output="$2"
    local max_attempts="${3:-3}"
    local delay="${4:-5}"
    
    for attempt in $(seq 1 "$max_attempts"); do
        echo "Download attempt $attempt/$max_attempts..."
        
        if curl -L -o "$output" "$url"; then
            echo "Download successful: $output"
            return 0
        else
            echo "Download failed, retrying in $delay seconds..."
            sleep "$delay"
        fi
    done
    
    echo "Download failed after $max_attempts attempts"
    return 1
}

# Bandwidth monitor
monitor_bandwidth() {
    local interface="${1:-eth0}"
    local interval="${2:-1}"
    
    echo "Monitoring bandwidth on $interface (Ctrl+C to stop)..."
    echo "Time,RX_bytes,TX_bytes,RX_rate,TX_rate"
    
    local prev_rx=0
    local prev_tx=0
    
    while true; do
        local rx_bytes=$(cat "/sys/class/net/$interface/statistics/rx_bytes" 2>/dev/null || echo 0)
        local tx_bytes=$(cat "/sys/class/net/$interface/statistics/tx_bytes" 2>/dev/null || echo 0)
        
        local rx_rate=$((rx_bytes - prev_rx))
        local tx_rate=$((tx_bytes - prev_tx))
        
        printf "%s,%d,%d,%d,%d\n" "$(date '+%H:%M:%S')" "$rx_bytes" "$tx_bytes" "$rx_rate" "$tx_rate"
        
        prev_rx=$rx_bytes
        prev_tx=$tx_bytes
        
        sleep "$interval"
    done
}

# Check website status
check_website() {
    local url="$1"
    local timeout="${2:-10}"
    
    local response=$(curl -s -o /dev/null -w "%{http_code},%{time_total},%{size_download}" \
                    --max-time "$timeout" "$url")
    
    IFS=',' read -r status_code response_time size <<< "$response"
    
    echo "URL: $url"
    echo "Status Code: $status_code"
    echo "Response Time: ${response_time}s"
    echo "Size: ${size} bytes"
    
    if [[ "$status_code" -eq 200 ]]; then
        echo "Status: OK"
        return 0
    else
        echo "Status: ERROR"
        return 1
    fi
}
```

---

*This comprehensive Bash snippets collection provides practical, reusable code for system administration, automation, and development workflows.* 