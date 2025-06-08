# Linux Performance Monitoring and Optimization for Data Science

## Table of Contents
- [System Monitoring Essentials](#system-monitoring-essentials)
- [CPU Performance Optimization](#cpu-performance-optimization)
- [Memory Management](#memory-management)
- [Storage I/O Optimization](#storage-io-optimization)
- [Network Performance](#network-performance)
- [GPU Monitoring](#gpu-monitoring)
- [Container Performance](#container-performance)
- [Data Science Specific Optimizations](#data-science-specific-optimizations)
- [Automated Monitoring Setup](#automated-monitoring-setup)

## System Monitoring Essentials

### Core System Monitoring Tools

```bash
# Install essential monitoring tools
sudo apt update && sudo apt install -y \
  htop btop glances iotop \
  sysstat nethogs iftop \
  ncdu tree pv \
  lm-sensors fancontrol \
  stress-ng benchmark \
  smartmontools hdparm

# For RHEL/CentOS/Fedora
sudo dnf install -y htop glances iotop sysstat nethogs iftop ncdu tree pv lm-sensors stress-ng smartmontools hdparm

# For Arch Linux
sudo pacman -S htop glances iotop sysstat nethogs iftop ncdu tree pv lm_sensors stress smartmontools hdparm
```

### Real-time System Overview

```bash
# Enhanced htop with custom configuration
cat > ~/.config/htop/htoprc << 'EOF'
fields=0 48 17 18 38 39 40 2 46 47 49 1
sort_key=46
sort_direction=1
hide_threads=0
hide_kernel_threads=1
hide_userland_threads=0
shadow_other_users=0
show_thread_names=0
show_program_path=1
highlight_base_name=0
highlight_megabytes=1
highlight_threads=1
tree_view=1
header_margin=1
detailed_cpu_time=0
cpu_count_from_zero=0
update_process_names=0
account_guest_in_cpu_meter=0
color_scheme=0
delay=15
left_meters=LeftCPUs Memory Swap
left_meter_modes=1 1 1
right_meters=RightCPUs Tasks LoadAverage Uptime
right_meter_modes=1 2 2 2
EOF

# Modern system monitor with better visualization
btop

# Comprehensive system information
glances -t 1 --enable-plugin cpu,mem,load,network,diskio,fs,sensors,docker
```

### Advanced Performance Monitoring

```bash
# Comprehensive performance monitoring script
cat > ~/bin/perf_monitor.sh << 'EOF'
#!/bin/bash

# Performance monitoring script for data science workloads
LOG_DIR="$HOME/performance_logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="$LOG_DIR/perf_${TIMESTAMP}.log"

echo "=== System Performance Report - $(date) ===" | tee "$LOG_FILE"

# System Information
echo -e "\n--- System Information ---" | tee -a "$LOG_FILE"
hostnamectl | tee -a "$LOG_FILE"
cat /proc/version | tee -a "$LOG_FILE"

# CPU Information
echo -e "\n--- CPU Information ---" | tee -a "$LOG_FILE"
lscpu | tee -a "$LOG_FILE"
cat /proc/cpuinfo | grep "model name" | head -1 | tee -a "$LOG_FILE"
echo "CPU Frequency:" | tee -a "$LOG_FILE"
cpufreq-info | grep "current CPU frequency" | tee -a "$LOG_FILE"

# Memory Information
echo -e "\n--- Memory Information ---" | tee -a "$LOG_FILE"
free -h | tee -a "$LOG_FILE"
echo -e "\nMemory details:" | tee -a "$LOG_FILE"
cat /proc/meminfo | grep -E "(MemTotal|MemFree|MemAvailable|Buffers|Cached|SwapTotal|SwapFree)" | tee -a "$LOG_FILE"

# Storage Information
echo -e "\n--- Storage Information ---" | tee -a "$LOG_FILE"
df -h | tee -a "$LOG_FILE"
echo -e "\nDisk usage by directory:" | tee -a "$LOG_FILE"
du -sh /* 2>/dev/null | sort -hr | head -10 | tee -a "$LOG_FILE"

# Network Information
echo -e "\n--- Network Information ---" | tee -a "$LOG_FILE"
ip addr show | grep -E "(inet |state UP)" | tee -a "$LOG_FILE"

# GPU Information (if available)
echo -e "\n--- GPU Information ---" | tee -a "$LOG_FILE"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi | tee -a "$LOG_FILE"
elif command -v rocm-smi &> /dev/null; then
    rocm-smi | tee -a "$LOG_FILE"
else
    echo "No GPU monitoring tools found" | tee -a "$LOG_FILE"
fi

# Current System Load
echo -e "\n--- Current System Load ---" | tee -a "$LOG_FILE"
uptime | tee -a "$LOG_FILE"
top -b -n1 | head -20 | tee -a "$LOG_FILE"

# I/O Statistics
echo -e "\n--- I/O Statistics ---" | tee -a "$LOG_FILE"
iostat -x 1 1 | tee -a "$LOG_FILE"

# Network Statistics
echo -e "\n--- Network Statistics ---" | tee -a "$LOG_FILE"
ss -tuln | head -20 | tee -a "$LOG_FILE"

echo -e "\nPerformance report saved to: $LOG_FILE"
EOF

chmod +x ~/bin/perf_monitor.sh
```

## CPU Performance Optimization

### CPU Frequency and Governor Settings

```bash
# Check current CPU governor
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Set performance governor for maximum CPU performance
echo "Setting CPU governor to performance..."
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance | sudo tee "$cpu"
done

# Create systemd service to set performance governor at boot
sudo tee /etc/systemd/system/cpu-performance.service << 'EOF'
[Unit]
Description=Set CPU Governor to Performance
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/bin/bash -c 'for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$cpu"; done'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable cpu-performance.service
sudo systemctl start cpu-performance.service

# Monitor CPU frequency in real-time
watch -n 1 'grep "MHz" /proc/cpuinfo | head -$(nproc)'
```

### CPU Affinity and Process Optimization

```bash
# CPU affinity optimization script
cat > ~/bin/optimize_cpu.sh << 'EOF'
#!/bin/bash

# Function to set CPU affinity for data science processes
optimize_process() {
    local process_name="$1"
    local cpu_list="$2"
    
    echo "Optimizing $process_name on CPUs: $cpu_list"
    
    # Find process IDs
    pids=$(pgrep -f "$process_name")
    
    for pid in $pids; do
        if [ -n "$pid" ]; then
            taskset -cp "$cpu_list" "$pid"
            echo "Set CPU affinity for PID $pid ($process_name) to CPUs: $cpu_list"
        fi
    done
}

# Optimize common data science applications
# Adjust CPU lists based on your system (0-based indexing)
TOTAL_CPUS=$(nproc)
HIGH_PERF_CPUS="0-$((TOTAL_CPUS/2-1))"
BACKGROUND_CPUS="$((TOTAL_CPUS/2))-$((TOTAL_CPUS-1))"

echo "Total CPUs: $TOTAL_CPUS"
echo "High performance CPUs: $HIGH_PERF_CPUS"
echo "Background CPUs: $BACKGROUND_CPUS"

# Optimize specific applications
optimize_process "python" "$HIGH_PERF_CPUS"
optimize_process "R" "$HIGH_PERF_CPUS"
optimize_process "jupyter" "$HIGH_PERF_CPUS"
optimize_process "rstudio" "$HIGH_PERF_CPUS"

# Move system processes to background CPUs
optimize_process "systemd" "$BACKGROUND_CPUS"
optimize_process "NetworkManager" "$BACKGROUND_CPUS"

echo "CPU optimization completed"
EOF

chmod +x ~/bin/optimize_cpu.sh
```

### CPU Stress Testing and Benchmarking

```bash
# Comprehensive CPU testing
cpu_stress_test() {
    echo "Starting CPU stress test..."
    
    # Basic stress test
    stress-ng --cpu $(nproc) --timeout 60s --metrics-brief
    
    # Matrix multiplication benchmark
    echo "Running matrix multiplication benchmark..."
    stress-ng --matrix $(nproc) --timeout 30s --metrics-brief
    
    # Memory and CPU combined test
    echo "Running combined CPU and memory test..."
    stress-ng --cpu $(nproc) --vm 2 --vm-bytes 1G --timeout 30s --metrics-brief
}

# Run CPU benchmark
cpu_stress_test
```

## Memory Management

### Memory Optimization Settings

```bash
# Optimize memory settings for data science workloads
sudo tee /etc/sysctl.d/99-datasci-memory.conf << 'EOF'
# Memory optimization for data science workloads

# Reduce swappiness (prefer RAM over swap)
vm.swappiness = 10

# Increase dirty ratio for better I/O performance
vm.dirty_ratio = 20
vm.dirty_background_ratio = 5

# Optimize memory overcommit
vm.overcommit_memory = 1
vm.overcommit_ratio = 80

# Increase max map count for large datasets
vm.max_map_count = 262144

# Optimize page cache
vm.vfs_cache_pressure = 50

# Increase shared memory limits
kernel.shmmax = 68719476736
kernel.shmall = 4294967296
EOF

# Apply settings
sudo sysctl -p /etc/sysctl.d/99-datasci-memory.conf

# Monitor memory usage
watch -n 1 'free -h && echo -e "\n--- Memory Details ---" && cat /proc/meminfo | grep -E "(MemFree|MemAvailable|Buffers|Cached|Dirty)"'
```

### Memory Profiling and Analysis

```bash
# Memory analysis script
cat > ~/bin/memory_analysis.sh << 'EOF'
#!/bin/bash

echo "=== Memory Analysis Report ==="
echo "Generated at: $(date)"
echo

# Overall memory usage
echo "--- System Memory Overview ---"
free -h
echo

# Memory usage by process
echo "--- Top Memory Consumers ---"
ps aux --sort=-%mem | head -20
echo

# Memory map for specific process (if provided)
if [ "$1" ]; then
    echo "--- Memory map for process: $1 ---"
    pmap -d $(pgrep -f "$1" | head -1) 2>/dev/null || echo "Process not found"
    echo
fi

# System memory information
echo "--- Detailed Memory Information ---"
cat /proc/meminfo | grep -E "(MemTotal|MemFree|MemAvailable|Buffers|Cached|SwapTotal|SwapFree|Dirty|Writeback)"
echo

# Hugepages information
echo "--- Hugepages Information ---"
grep -E "(HugePages|Hugepagesize)" /proc/meminfo
echo

# Memory pressure indicators
echo "--- Memory Pressure Indicators ---"
echo "Page faults:"
grep -E "pgfault|pgmajfault" /proc/vmstat
echo
echo "Memory reclaim activity:"
grep -E "pgsteal|pgscan" /proc/vmstat

EOF

chmod +x ~/bin/memory_analysis.sh

# Run memory analysis
~/bin/memory_analysis.sh python
```

### Swap Optimization

```bash
# Optimize swap configuration
optimize_swap() {
    echo "Current swap configuration:"
    swapon --show
    cat /proc/swaps
    
    # Create optimized swap file if needed
    if [ ! -f /swapfile ] && [ $(free | grep Swap | awk '{print $2}') -lt 2097152 ]; then
        echo "Creating optimized swap file..."
        sudo fallocate -l 4G /swapfile
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
        sudo swapon /swapfile
        
        # Add to fstab
        echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    fi
    
    echo "Swap optimization completed"
}

# Run swap optimization
optimize_swap
```

## Storage I/O Optimization

### Disk I/O Monitoring and Optimization

```bash
# I/O monitoring and optimization
cat > ~/bin/io_optimization.sh << 'EOF'
#!/bin/bash

echo "=== Storage I/O Optimization ==="

# Identify storage devices
echo "--- Storage Devices ---"
lsblk -f
echo

# Check current I/O scheduler
echo "--- Current I/O Schedulers ---"
for device in /sys/block/sd*; do
    if [ -d "$device" ]; then
        device_name=$(basename "$device")
        scheduler=$(cat "$device/queue/scheduler")
        echo "$device_name: $scheduler"
    fi
done
echo

# Set optimal I/O scheduler for SSDs and HDDs
for device in /sys/block/sd*; do
    if [ -d "$device" ]; then
        device_name=$(basename "$device")
        
        # Check if it's an SSD
        rotational=$(cat "$device/queue/rotational")
        
        if [ "$rotational" = "0" ]; then
            # SSD - use none (noop) or mq-deadline
            echo noop | sudo tee "$device/queue/scheduler" 2>/dev/null || echo mq-deadline | sudo tee "$device/queue/scheduler"
            echo "Set SSD $device_name scheduler to noop/mq-deadline"
        else
            # HDD - use deadline or bfq
            echo deadline | sudo tee "$device/queue/scheduler" 2>/dev/null || echo bfq | sudo tee "$device/queue/scheduler"
            echo "Set HDD $device_name scheduler to deadline/bfq"
        fi
    fi
done

# Optimize read-ahead settings
echo
echo "--- Optimizing Read-ahead Settings ---"
for device in /sys/block/sd*; do
    if [ -d "$device" ]; then
        device_name=$(basename "$device")
        # Set read-ahead to 4MB for better sequential performance
        echo 8192 | sudo tee "$device/queue/read_ahead_kb"
        echo "Set read-ahead for $device_name to 4MB"
    fi
done

# Check and optimize filesystem mount options
echo
echo "--- Current Mount Options ---"
mount | grep -E "(ext4|xfs|btrfs)" | while read line; do
    echo "$line"
done

EOF

chmod +x ~/bin/io_optimization.sh

# Run I/O optimization
~/bin/io_optimization.sh

# Monitor I/O in real-time
iotop -ao

# Detailed I/O statistics
iostat -x 1 5
```

### Filesystem Optimization

```bash
# Filesystem optimization for data science workloads
optimize_filesystem() {
    echo "=== Filesystem Optimization ==="
    
    # Check current filesystems
    df -T
    
    # Optimize ext4 filesystems
    echo "Optimizing ext4 filesystems..."
    for fs in $(findmnt -t ext4 -o TARGET -n); do
        echo "Optimizing $fs"
        # These would require remounting, document the optimal options instead
        cat << EOF
Optimal ext4 mount options for $fs:
- noatime: Reduce access time updates
- commit=60: Increase commit interval
- barrier=0: Disable barriers if on UPS/battery backup
- data=writeback: Fastest journaling mode (less safe)

Add to /etc/fstab:
UUID=... $fs ext4 defaults,noatime,commit=60 0 2
EOF
    done
    
    # Optimize XFS filesystems
    echo "XFS optimization recommendations:"
    cat << EOF
Optimal XFS mount options:
- noatime: Reduce access time updates
- largeio: Optimize for large I/O operations
- swalloc: Stripe width allocation

Example /etc/fstab entry:
UUID=... /data xfs defaults,noatime,largeio,swalloc 0 2
EOF
}

optimize_filesystem
```

## Network Performance

### Network Optimization Settings

```bash
# Network optimization for data science workloads
sudo tee /etc/sysctl.d/99-datasci-network.conf << 'EOF'
# Network optimization for data science workloads

# TCP buffer sizes
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 65536 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728

# Increase maximum number of connections
net.core.somaxconn = 65535
net.core.netdev_max_backlog = 5000

# TCP optimization
net.ipv4.tcp_congestion_control = bbr
net.ipv4.tcp_slow_start_after_idle = 0
net.ipv4.tcp_keepalive_time = 600
net.ipv4.tcp_keepalive_intvl = 60
net.ipv4.tcp_keepalive_probes = 3

# Reduce TCP time wait
net.ipv4.tcp_fin_timeout = 30
net.ipv4.tcp_tw_reuse = 1

# Enable TCP window scaling
net.ipv4.tcp_window_scaling = 1
net.ipv4.tcp_timestamps = 1
net.ipv4.tcp_sack = 1

# UDP buffer sizes
net.core.rmem_default = 262144
net.core.wmem_default = 262144
EOF

# Apply network optimizations
sudo sysctl -p /etc/sysctl.d/99-datasci-network.conf

# Monitor network performance
cat > ~/bin/network_monitor.sh << 'EOF'
#!/bin/bash

echo "=== Network Performance Monitor ==="
echo "Generated at: $(date)"
echo

# Interface statistics
echo "--- Network Interfaces ---"
ip -s link show
echo

# Network connections
echo "--- Active Connections ---"
ss -tuln | head -20
echo

# Bandwidth usage
echo "--- Bandwidth Usage (real-time) ---"
echo "Press Ctrl+C to stop"
iftop -t -s 10 2>/dev/null || echo "iftop not available, install with: sudo apt install iftop"

EOF

chmod +x ~/bin/network_monitor.sh
```

## GPU Monitoring

### NVIDIA GPU Optimization

```bash
# NVIDIA GPU monitoring and optimization
setup_nvidia_monitoring() {
    if command -v nvidia-smi &> /dev/null; then
        echo "Setting up NVIDIA GPU monitoring..."
        
        # Create comprehensive GPU monitoring script
        cat > ~/bin/gpu_monitor.sh << 'EOF'
#!/bin/bash

echo "=== NVIDIA GPU Performance Monitor ==="
echo "Generated at: $(date)"
echo

# GPU information
echo "--- GPU Information ---"
nvidia-smi -q -d MEMORY,UTILIZATION,TEMPERATURE,POWER,CLOCK
echo

# Real-time monitoring
echo "--- Real-time GPU Stats ---"
watch -n 1 'nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw,power.limit --format=csv'

EOF
        
        chmod +x ~/bin/gpu_monitor.sh
        
        # Set up optimal GPU settings
        echo "Configuring optimal GPU settings..."
        
        # Set persistence mode
        sudo nvidia-smi -pm 1
        
        # Set application clock speeds to maximum
        sudo nvidia-smi -ac $(nvidia-smi --query-supported-clocks=memory,graphics --format=csv,noheader,nounits | tail -1 | tr ',' ' ')
        
        # Enable ECC if available
        sudo nvidia-smi -e 1 2>/dev/null || echo "ECC not available on this GPU"
        
        echo "NVIDIA GPU optimization completed"
    else
        echo "NVIDIA drivers not found"
    fi
}

# AMD GPU optimization
setup_amd_monitoring() {
    if command -v rocm-smi &> /dev/null; then
        echo "Setting up AMD GPU monitoring..."
        
        cat > ~/bin/amd_gpu_monitor.sh << 'EOF'
#!/bin/bash

echo "=== AMD GPU Performance Monitor ==="
echo "Generated at: $(date)"
echo

# GPU information
echo "--- GPU Information ---"
rocm-smi --showproductname --showtemp --showclocks --showmeminfo --showuse
echo

# Real-time monitoring
echo "--- Real-time GPU Stats ---"
watch -n 1 rocm-smi

EOF
        
        chmod +x ~/bin/amd_gpu_monitor.sh
        echo "AMD GPU monitoring setup completed"
    else
        echo "AMD ROCm drivers not found"
    fi
}

# Run GPU setup
setup_nvidia_monitoring
setup_amd_monitoring
```

## Container Performance

### Docker Optimization for Data Science

```bash
# Docker optimization for data science workloads
cat > ~/bin/docker_optimization.sh << 'EOF'
#!/bin/bash

echo "=== Docker Optimization for Data Science ==="

# Create optimized Docker daemon configuration
sudo mkdir -p /etc/docker

sudo tee /etc/docker/daemon.json << 'DOCKER_EOF'
{
  "storage-driver": "overlay2",
  "storage-opts": [
    "overlay2.override_kernel_check=true"
  ],
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "default-runtime": "runc",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-ulimits": {
    "memlock": {
      "Name": "memlock",
      "Hard": -1,
      "Soft": -1
    },
    "stack": {
      "Name": "stack",
      "Hard": 67108864,
      "Soft": 67108864
    }
  },
  "default-shm-size": "2G",
  "max-concurrent-downloads": 10,
  "max-concurrent-uploads": 10
}
DOCKER_EOF

# Restart Docker service
sudo systemctl restart docker

# Container monitoring script
cat > ~/bin/container_monitor.sh << 'MONITOR_EOF'
#!/bin/bash

echo "=== Container Performance Monitor ==="
echo "Generated at: $(date)"
echo

# Container resource usage
echo "--- Container Resource Usage ---"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}"
echo

# System resources used by Docker
echo "--- Docker System Resources ---"
docker system df
echo

# Running containers
echo "--- Running Containers ---"
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"

MONITOR_EOF

chmod +x ~/bin/container_monitor.sh

echo "Docker optimization completed"
echo "Run ~/bin/container_monitor.sh to monitor container performance"

EOF

chmod +x ~/bin/docker_optimization.sh

# Run Docker optimization
~/bin/docker_optimization.sh
```

## Data Science Specific Optimizations

### Python/Conda Environment Optimization

```bash
# Python environment optimization
cat > ~/bin/python_optimization.sh << 'EOF'
#!/bin/bash

echo "=== Python/Conda Environment Optimization ==="

# Set optimal Python environment variables
cat > ~/.python_env << 'PYTHON_EOF'
# Python optimization for data science
export PYTHONOPTIMIZE=2
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# NumPy/SciPy optimizations
export OPENBLAS_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export NUMEXPR_NUM_THREADS=$(nproc)
export OMP_NUM_THREADS=$(nproc)

# Pandas optimizations
export PANDAS_COMPUTE_N_WORKERS=$(nproc)

# Jupyter optimizations
export JUPYTER_CONFIG_DIR=$HOME/.jupyter
export JUPYTER_DATA_DIR=$HOME/.local/share/jupyter
PYTHON_EOF

# Source Python environment in bashrc
if ! grep -q "source ~/.python_env" ~/.bashrc; then
    echo "source ~/.python_env" >> ~/.bashrc
fi

# Create optimized Jupyter configuration
mkdir -p ~/.jupyter

cat > ~/.jupyter/jupyter_notebook_config.py << 'JUPYTER_EOF'
# Jupyter optimization configuration
c.NotebookApp.max_buffer_size = 2147483648  # 2GB
c.NotebookApp.iopub_data_rate_limit = 1000000000  # 1GB/s
c.NotebookApp.rate_limit_window = 1.0
c.NotebookApp.disable_check_xsrf = True
c.NotebookApp.tornado_settings = {
    'headers': {
        'Content-Security-Policy': "frame-ancestors 'self' *"
    }
}

# Performance settings
c.NotebookApp.kernel_spec_manager_class = 'jupyter_client.kernelspec.KernelSpecManager'
c.MultiKernelManager.default_kernel_name = 'python3'

# Memory and CPU settings
import os
c.NotebookApp.max_body_size = 2 * 1024 * 1024 * 1024  # 2GB
c.NotebookApp.max_buffer_size = 2 * 1024 * 1024 * 1024  # 2GB

JUPYTER_EOF

echo "Python optimization completed"
echo "Restart your shell or run: source ~/.bashrc"

EOF

chmod +x ~/bin/python_optimization.sh

# Run Python optimization
~/bin/python_optimization.sh
```

### R Environment Optimization

```bash
# R environment optimization
cat > ~/bin/r_optimization.sh << 'EOF'
#!/bin/bash

echo "=== R Environment Optimization ==="

# Create R environment configuration
cat > ~/.Renviron << 'R_EOF'
# R optimization for data science
R_MAX_NUM_DLLS=500
R_LIBS_USER=~/R/library
R_HISTSIZE=10000

# Parallel processing
MC_CORES=8
NCPUS=8

# Memory settings
R_MAX_VSIZE=100Gb

# BLAS/LAPACK optimization
OPENBLAS_NUM_THREADS=8
MKL_NUM_THREADS=8

# RStudio settings
RSTUDIO_PANDOC=/usr/lib/rstudio/bin/pandoc
R_EOF

# Create optimized Rprofile
cat > ~/.Rprofile << 'RPROFILE_EOF'
# R startup optimization
options(
  # Performance options
  max.print = 1000,
  scipen = 999,
  digits = 4,
  warn = 1,
  
  # Memory options
  expressions = 500000,
  
  # Parallel options
  mc.cores = parallel::detectCores(),
  Ncpus = parallel::detectCores(),
  
  # Repository options
  repos = c(CRAN = "https://cran.rstudio.com/"),
  
  # Browser options
  browser = "firefox",
  
  # Package library
  lib.loc = "~/R/library"
)

# Load commonly used packages silently
suppressPackageStartupMessages({
  if (interactive()) {
    # Only load in interactive sessions
    tryCatch({
      library(data.table)
      library(dplyr)
      library(ggplot2)
    }, error = function(e) {
      # Packages not installed, skip
    })
  }
})

# Useful functions
.rs.api.clear <- function() cat("\014")  # Clear console
.First <- function() {
  if (interactive()) {
    cat("R", R.version.string, "optimized for data science\n")
    cat("Available cores:", parallel::detectCores(), "\n")
    cat("Memory limit:", memory.limit(), "MB\n")
  }
}

# Set number of threads for data.table
if (requireNamespace("data.table", quietly = TRUE)) {
  data.table::setDTthreads(parallel::detectCores())
}

RPROFILE_EOF

# Create R library directory
mkdir -p ~/R/library

echo "R optimization completed"
echo "Restart R/RStudio to apply changes"

EOF

chmod +x ~/bin/r_optimization.sh

# Run R optimization
~/bin/r_optimization.sh
```

## Automated Monitoring Setup

### System Performance Dashboard

```bash
# Create comprehensive monitoring dashboard
cat > ~/bin/performance_dashboard.sh << 'EOF'
#!/bin/bash

# Performance dashboard script
DASHBOARD_DIR="$HOME/performance_dashboard"
mkdir -p "$DASHBOARD_DIR"

# Generate HTML dashboard
cat > "$DASHBOARD_DIR/dashboard.html" << 'HTML_EOF'
<!DOCTYPE html>
<html>
<head>
    <title>System Performance Dashboard</title>
    <meta http-equiv="refresh" content="30">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .metric-box { border: 1px solid #ccc; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .critical { background-color: #ffebee; }
        .warning { background-color: #fff3e0; }
        .good { background-color: #e8f5e8; }
        .header { font-size: 18px; font-weight: bold; margin-bottom: 10px; }
        pre { background-color: #f5f5f5; padding: 10px; overflow-x: auto; }
        .timestamp { color: #666; font-size: 12px; }
    </style>
</head>
<body>
    <h1>System Performance Dashboard</h1>
    <div class="timestamp">Last updated: $(date)</div>
    
    <div class="metric-box">
        <div class="header">System Overview</div>
        <pre>$(uptime)</pre>
        <pre>$(free -h)</pre>
    </div>
    
    <div class="metric-box">
        <div class="header">CPU Usage</div>
        <pre>$(top -b -n1 | head -5 | tail -1)</pre>
        <pre>$(mpstat 1 1 | tail -1 2>/dev/null || echo "sysstat not installed")</pre>
    </div>
    
    <div class="metric-box">
        <div class="header">Memory Usage</div>
        <pre>$(ps aux --sort=-%mem | head -10)</pre>
    </div>
    
    <div class="metric-box">
        <div class="header">Disk Usage</div>
        <pre>$(df -h | grep -vE '^Filesystem|tmpfs|cdrom')</pre>
        <pre>$(iostat -x 1 1 | tail -5 2>/dev/null || echo "sysstat not installed")</pre>
    </div>
    
    <div class="metric-box">
        <div class="header">Network</div>
        <pre>$(ss -tuln | head -10)</pre>
    </div>
    
    <div class="metric-box">
        <div class="header">GPU Status</div>
        <pre>$(nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null || echo "No NVIDIA GPU detected")</pre>
    </div>
    
    <div class="metric-box">
        <div class="header">Running Processes</div>
        <pre>$(ps aux --sort=-%cpu | head -15)</pre>
    </div>
</body>
</html>
HTML_EOF

echo "Dashboard generated: $DASHBOARD_DIR/dashboard.html"
echo "Open in browser: file://$DASHBOARD_DIR/dashboard.html"

# Create auto-refresh script
cat > "$DASHBOARD_DIR/auto_refresh.sh" << 'REFRESH_EOF'
#!/bin/bash
while true; do
    bash ~/bin/performance_dashboard.sh
    sleep 30
done
REFRESH_EOF

chmod +x "$DASHBOARD_DIR/auto_refresh.sh"

EOF

chmod +x ~/bin/performance_dashboard.sh

# Generate initial dashboard
~/bin/performance_dashboard.sh
```

### Automated Performance Alerts

```bash
# Create performance alerting system
cat > ~/bin/performance_alerts.sh << 'EOF'
#!/bin/bash

# Performance alerting thresholds
CPU_THRESHOLD=80
MEMORY_THRESHOLD=85
DISK_THRESHOLD=90
LOAD_THRESHOLD=10

LOG_FILE="$HOME/performance_alerts.log"

# Function to send alert
send_alert() {
    local metric="$1"
    local value="$2"
    local threshold="$3"
    local timestamp=$(date)
    
    echo "[$timestamp] ALERT: $metric is at $value% (threshold: $threshold%)" | tee -a "$LOG_FILE"
    
    # Add notification here (email, Slack, etc.)
    # notify-send "Performance Alert" "$metric is at $value% (threshold: $threshold%)"
}

# Check CPU usage
cpu_usage=$(top -b -n1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1 | cut -d',' -f1)
if (( $(echo "$cpu_usage > $CPU_THRESHOLD" | bc -l) )); then
    send_alert "CPU Usage" "$cpu_usage" "$CPU_THRESHOLD"
fi

# Check memory usage
memory_usage=$(free | grep Mem | awk '{printf("%.0f", $3/$2 * 100.0)}')
if [ "$memory_usage" -gt "$MEMORY_THRESHOLD" ]; then
    send_alert "Memory Usage" "$memory_usage" "$MEMORY_THRESHOLD"
fi

# Check disk usage
disk_usage=$(df / | tail -1 | awk '{print $5}' | cut -d'%' -f1)
if [ "$disk_usage" -gt "$DISK_THRESHOLD" ]; then
    send_alert "Disk Usage" "$disk_usage" "$DISK_THRESHOLD"
fi

# Check load average
load_avg=$(uptime | awk -F'load average:' '{print $2}' | cut -d',' -f1 | xargs)
if (( $(echo "$load_avg > $LOAD_THRESHOLD" | bc -l) )); then
    send_alert "Load Average" "$load_avg" "$LOAD_THRESHOLD"
fi

EOF

chmod +x ~/bin/performance_alerts.sh

# Create cron job for performance monitoring
(crontab -l 2>/dev/null; echo "*/5 * * * * $HOME/bin/performance_alerts.sh") | crontab -

echo "Performance monitoring setup completed!"
echo "Available scripts:"
echo "  ~/bin/perf_monitor.sh       - Comprehensive performance report"
echo "  ~/bin/memory_analysis.sh    - Memory usage analysis"
echo "  ~/bin/io_optimization.sh    - Storage I/O optimization"
echo "  ~/bin/gpu_monitor.sh        - GPU monitoring (if available)"
echo "  ~/bin/performance_dashboard.sh - HTML dashboard generator"
echo "  ~/bin/performance_alerts.sh - Performance alerting"
echo ""
echo "Performance alerts will run every 5 minutes via cron"
echo "Dashboard: file://$HOME/performance_dashboard/dashboard.html"
```

This comprehensive Linux performance monitoring and optimization guide provides:

1. **System Monitoring Tools**: Installation and configuration of essential monitoring utilities
2. **CPU Optimization**: Governor settings, affinity optimization, and stress testing
3. **Memory Management**: Swap optimization, memory analysis, and system tuning
4. **Storage I/O**: Disk scheduler optimization, filesystem tuning, and I/O monitoring
5. **Network Performance**: TCP/UDP optimizations and bandwidth monitoring
6. **GPU Monitoring**: NVIDIA and AMD GPU optimization and monitoring
7. **Container Performance**: Docker optimization for data science workloads
8. **Data Science Specific**: Python/R environment optimizations
9. **Automated Monitoring**: Performance dashboards and alerting systems

The guide includes practical scripts and configurations that can be immediately applied to optimize Linux systems for data science workloads. 