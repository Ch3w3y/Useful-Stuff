# Bash Scripting Guide

## Overview

Comprehensive guide to Bash scripting for automation, system administration, and command-line mastery. Covers fundamentals through advanced techniques with practical examples and best practices.

## Table of Contents

- [Bash Fundamentals](#bash-fundamentals)
- [Variables and Data Types](#variables-and-data-types)
- [Control Structures](#control-structures)
- [Functions](#functions)
- [File Operations](#file-operations)
- [Text Processing](#text-processing)
- [System Administration](#system-administration)
- [Advanced Scripting](#advanced-scripting)
- [Best Practices](#best-practices)

## Bash Fundamentals

### Basic Syntax and Structure

```bash
#!/bin/bash
# This is a comment
# Script: basic_example.sh
# Purpose: Demonstrate basic Bash syntax

# Variables
name="World"
count=42

# Basic output
echo "Hello, $name!"
echo "Count: $count"

# Command substitution
current_date=$(date)
echo "Current date: $current_date"

# Arithmetic
result=$((count + 8))
echo "Result: $result"

# Conditional
if [ $count -gt 40 ]; then
    echo "Count is greater than 40"
fi

# Loop
for i in {1..5}; do
    echo "Iteration: $i"
done
```

### Command Line Arguments

```bash
#!/bin/bash
# Script: arguments.sh
# Usage: ./arguments.sh arg1 arg2 arg3

echo "Script name: $0"
echo "First argument: $1"
echo "Second argument: $2"
echo "All arguments: $@"
echo "Number of arguments: $#"
echo "Process ID: $$"
echo "Exit status of last command: $?"

# Check if arguments provided
if [ $# -eq 0 ]; then
    echo "No arguments provided"
    echo "Usage: $0 <arg1> <arg2> ..."
    exit 1
fi

# Process each argument
echo "Processing arguments:"
for arg in "$@"; do
    echo "  - $arg"
done
```

### Exit Codes and Error Handling

```bash
#!/bin/bash
# Script: error_handling.sh

# Set strict mode
set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Function to handle errors
error_handler() {
    echo "Error occurred in script at line: ${1:-"Unknown"}"
    echo "Command: ${2:-"Unknown"}"
    exit 1
}

# Set error trap
trap 'error_handler ${LINENO} "$BASH_COMMAND"' ERR

# Success function
success_msg() {
    echo "✓ $1"
}

# Error function
error_msg() {
    echo "✗ $1" >&2
}

# Warning function
warn_msg() {
    echo "⚠ $1" >&2
}

# Example usage
check_file() {
    local file="$1"
    
    if [ -f "$file" ]; then
        success_msg "File $file exists"
        return 0
    else
        error_msg "File $file not found"
        return 1
    fi
}

# Test the function
check_file "/etc/passwd" || warn_msg "System file check failed"
```

## Variables and Data Types

### Variable Declaration and Scope

```bash
#!/bin/bash
# Script: variables.sh

# Global variables
GLOBAL_VAR="I'm global"
readonly READONLY_VAR="Cannot change me"

# Local variables in functions
demonstrate_scope() {
    local local_var="I'm local"
    local_var="Modified locally"
    
    echo "Inside function:"
    echo "  Global: $GLOBAL_VAR"
    echo "  Local: $local_var"
    echo "  Readonly: $READONLY_VAR"
}

# Arrays
declare -a indexed_array=("apple" "banana" "cherry")
declare -A associative_array=(
    ["name"]="John"
    ["age"]="30"
    ["city"]="New York"
)

# Environment variables
export MY_ENV_VAR="Available to child processes"

echo "Before function call:"
echo "Global: $GLOBAL_VAR"

demonstrate_scope

echo -e "\nArray examples:"
echo "Indexed array: ${indexed_array[@]}"
echo "Array length: ${#indexed_array[@]}"
echo "First element: ${indexed_array[0]}"

echo -e "\nAssociative array:"
for key in "${!associative_array[@]}"; do
    echo "$key: ${associative_array[$key]}"
done
```

### String Manipulation

```bash
#!/bin/bash
# Script: string_manipulation.sh

text="Hello, World! Welcome to Bash scripting."
filename="document.txt.backup"

echo "Original text: $text"
echo "String length: ${#text}"

# Substring extraction
echo "Substring (7-12): ${text:7:5}"
echo "From position 7: ${text:7}"

# Case conversion
echo "Uppercase: ${text^^}"
echo "Lowercase: ${text,,}"
echo "Title case first char: ${text^}"

# Pattern matching and replacement
echo "Replace 'World': ${text/World/Universe}"
echo "Replace all 'e': ${text//e/E}"

# Filename manipulation
echo "Filename: $filename"
echo "Extension: ${filename##*.}"
echo "Base name: ${filename%.*}"
echo "Directory: ${filename%/*}"

# Default values
unset undefined_var
echo "Default value: ${undefined_var:-"default"}"
echo "Assign default: ${undefined_var:="assigned_default"}"

# String validation
validate_string() {
    local input="$1"
    
    # Check if empty
    [ -z "$input" ] && echo "String is empty" && return 1
    
    # Check if contains only digits
    if [[ "$input" =~ ^[0-9]+$ ]]; then
        echo "String contains only digits"
    fi
    
    # Check if valid email format
    if [[ "$input" =~ ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$ ]]; then
        echo "Valid email format"
    fi
    
    return 0
}

validate_string "123"
validate_string "user@example.com"
```

## Control Structures

### Conditional Statements

```bash
#!/bin/bash
# Script: conditionals.sh

check_file_permissions() {
    local file="$1"
    
    if [ ! -e "$file" ]; then
        echo "File does not exist: $file"
        return 1
    fi
    
    echo "Checking permissions for: $file"
    
    # Multiple conditions with elif
    if [ -r "$file" ] && [ -w "$file" ] && [ -x "$file" ]; then
        echo "✓ Full permissions (read/write/execute)"
    elif [ -r "$file" ] && [ -w "$file" ]; then
        echo "✓ Read and write permissions"
    elif [ -r "$file" ]; then
        echo "✓ Read permission only"
    else
        echo "✗ No read permission"
    fi
    
    # Case statement
    case "$file" in
        *.txt)
            echo "Text file detected"
            ;;
        *.sh)
            echo "Shell script detected"
            ;;
        *.log)
            echo "Log file detected"
            ;;
        *)
            echo "Unknown file type"
            ;;
    esac
}

# Numeric comparisons
compare_numbers() {
    local num1="$1"
    local num2="$2"
    
    if [ "$num1" -eq "$num2" ]; then
        echo "$num1 equals $num2"
    elif [ "$num1" -gt "$num2" ]; then
        echo "$num1 is greater than $num2"
    else
        echo "$num1 is less than $num2"
    fi
    
    # Arithmetic evaluation
    if (( num1 > 0 && num2 > 0 )); then
        echo "Both numbers are positive"
    fi
}

# Test functions
check_file_permissions "$0"  # Check this script
compare_numbers 10 5
```

### Loops

```bash
#!/bin/bash
# Script: loops.sh

# For loops
echo "=== For Loop Examples ==="

# Range loop
echo "Counting 1 to 5:"
for i in {1..5}; do
    echo "  Count: $i"
done

# Step loop
echo "Even numbers 2 to 10:"
for i in {2..10..2}; do
    echo "  Even: $i"
done

# Array iteration
fruits=("apple" "banana" "cherry" "date")
echo "Fruits:"
for fruit in "${fruits[@]}"; do
    echo "  - $fruit"
done

# File iteration
echo "Files in current directory:"
for file in *; do
    if [ -f "$file" ]; then
        echo "  File: $file"
    fi
done

# While loops
echo -e "\n=== While Loop Examples ==="

counter=1
echo "While loop counting to 3:"
while [ $counter -le 3 ]; do
    echo "  Counter: $counter"
    ((counter++))
done

# Reading file line by line
echo "Reading /etc/passwd (first 3 lines):"
line_count=0
while IFS=: read -r username password uid gid info home shell; do
    echo "  User: $username, UID: $uid, Shell: $shell"
    ((line_count++))
    [ $line_count -ge 3 ] && break
done < /etc/passwd

# Until loop
echo -e "\n=== Until Loop Example ==="
countdown=3
echo "Countdown:"
until [ $countdown -eq 0 ]; do
    echo "  $countdown..."
    ((countdown--))
    sleep 1
done
echo "  Blast off!"
```

## Functions

### Function Definition and Usage

```bash
#!/bin/bash
# Script: functions.sh

# Basic function
greet() {
    echo "Hello, $1!"
}

# Function with return value
add_numbers() {
    local num1="$1"
    local num2="$2"
    local result=$((num1 + num2))
    echo "$result"  # Return via echo
}

# Function with multiple return types
divide_numbers() {
    local dividend="$1"
    local divisor="$2"
    
    # Check for division by zero
    if [ "$divisor" -eq 0 ]; then
        echo "Error: Division by zero" >&2
        return 1
    fi
    
    # Perform division
    local result=$((dividend / divisor))
    local remainder=$((dividend % divisor))
    
    echo "Result: $result, Remainder: $remainder"
    return 0
}

# Advanced function with options
process_files() {
    local OPTIND=1
    local verbose=false
    local backup=false
    local extension=""
    
    # Parse options
    while getopts "vbe:" opt; do
        case $opt in
            v) verbose=true ;;
            b) backup=true ;;
            e) extension="$OPTARG" ;;
            \?) echo "Invalid option: -$OPTARG" >&2; return 1 ;;
        esac
    done
    
    shift $((OPTIND-1))
    
    # Process remaining arguments (files)
    for file in "$@"; do
        [ "$verbose" = true ] && echo "Processing: $file"
        
        if [ "$backup" = true ]; then
            cp "$file" "${file}.backup"
            [ "$verbose" = true ] && echo "  Backup created: ${file}.backup"
        fi
        
        if [ -n "$extension" ]; then
            mv "$file" "${file}.$extension"
            [ "$verbose" = true ] && echo "  Renamed to: ${file}.$extension"
        fi
    done
}

# Recursive function
factorial() {
    local n="$1"
    
    if [ "$n" -le 1 ]; then
        echo 1
    else
        local prev_result
        prev_result=$(factorial $((n - 1)))
        echo $((n * prev_result))
    fi
}

# Function with error handling
safe_file_operation() {
    local operation="$1"
    local file="$2"
    
    case "$operation" in
        "read")
            if [ -r "$file" ]; then
                cat "$file"
                return 0
            else
                echo "Error: Cannot read file $file" >&2
                return 1
            fi
            ;;
        "delete")
            if [ -f "$file" ]; then
                rm "$file" && echo "File $file deleted"
                return 0
            else
                echo "Error: File $file not found" >&2
                return 1
            fi
            ;;
        *)
            echo "Error: Unknown operation $operation" >&2
            return 1
            ;;
    esac
}

# Test functions
echo "=== Function Examples ==="
greet "World"

sum=$(add_numbers 15 25)
echo "15 + 25 = $sum"

divide_numbers 17 5
divide_numbers 10 0

echo "Factorial of 5: $(factorial 5)"
```

## File Operations

### File System Navigation and Manipulation

```bash
#!/bin/bash
# Script: file_operations.sh

# File and directory operations
manage_files() {
    local base_dir="temp_workspace"
    
    echo "=== File System Operations ==="
    
    # Create directory structure
    echo "Creating directory structure..."
    mkdir -p "$base_dir"/{docs,scripts,logs}/{2023,2024}
    
    # Create sample files
    echo "Creating sample files..."
    for year in 2023 2024; do
        echo "Log entry for $year" > "$base_dir/logs/$year/app.log"
        echo "#!/bin/bash" > "$base_dir/scripts/$year/script.sh"
        echo "Document for $year" > "$base_dir/docs/$year/readme.txt"
    done
    
    # List directory structure
    echo "Directory structure:"
    tree "$base_dir" 2>/dev/null || find "$base_dir" -type d | sort
    
    # File information
    echo -e "\nFile information:"
    for file in "$base_dir"/*/*/*.{log,sh,txt}; do
        if [ -f "$file" ]; then
            echo "File: $file"
            echo "  Size: $(stat -f%z "$file" 2>/dev/null || stat -c%s "$file") bytes"
            echo "  Modified: $(stat -f%Sm "$file" 2>/dev/null || stat -c%y "$file")"
            echo "  Permissions: $(stat -f%Mp%Lp "$file" 2>/dev/null || stat -c%a "$file")"
        fi
    done
    
    # File operations
    echo -e "\nFile operations:"
    
    # Copy files
    cp "$base_dir/docs/2023/readme.txt" "$base_dir/docs/2023/readme_backup.txt"
    echo "✓ Created backup"
    
    # Move files
    mv "$base_dir/docs/2024/readme.txt" "$base_dir/docs/2024/document.txt"
    echo "✓ Renamed file"
    
    # Set permissions
    chmod +x "$base_dir/scripts"/*/*.sh
    echo "✓ Made scripts executable"
    
    # Find files by criteria
    echo -e "\nFinding files:"
    echo "Log files:"
    find "$base_dir" -name "*.log" -type f
    
    echo "Files modified in last minute:"
    find "$base_dir" -mmin -1 -type f
    
    echo "Executable files:"
    find "$base_dir" -executable -type f
    
    # Cleanup
    echo -e "\nCleaning up..."
    rm -rf "$base_dir"
    echo "✓ Workspace cleaned"
}

# File content operations
process_file_content() {
    local input_file="$1"
    local output_file="$2"
    
    if [ ! -f "$input_file" ]; then
        echo "Input file not found: $input_file"
        return 1
    fi
    
    echo "Processing file content..."
    
    # Count lines, words, characters
    echo "File statistics:"
    wc -l "$input_file" | awk '{print "  Lines: " $1}'
    wc -w "$input_file" | awk '{print "  Words: " $1}'
    wc -c "$input_file" | awk '{print "  Characters: " $1}'
    
    # Extract specific content
    echo "First 5 lines:"
    head -5 "$input_file"
    
    echo "Last 5 lines:"
    tail -5 "$input_file"
    
    # Process and save to output file
    {
        echo "=== Processed Content ==="
        echo "Original file: $input_file"
        echo "Processed on: $(date)"
        echo "=== Content ==="
        cat "$input_file" | tr '[:lower:]' '[:upper:]'  # Convert to uppercase
    } > "$output_file"
    
    echo "Processed content saved to: $output_file"
}

# Archive operations
create_archive() {
    local source_dir="$1"
    local archive_name="$2"
    
    if [ ! -d "$source_dir" ]; then
        echo "Source directory not found: $source_dir"
        return 1
    fi
    
    echo "Creating archive..."
    
    # Create different archive formats
    tar -czf "${archive_name}.tar.gz" "$source_dir"
    echo "✓ Created gzipped tar: ${archive_name}.tar.gz"
    
    zip -r "${archive_name}.zip" "$source_dir" >/dev/null
    echo "✓ Created zip: ${archive_name}.zip"
    
    # Display archive information
    echo "Archive information:"
    ls -lh "${archive_name}".{tar.gz,zip}
}

# Test file operations
manage_files

# Create test file for content processing
echo -e "line one\nline two\nline three\nline four\nline five\nline six" > test_input.txt
process_file_content test_input.txt processed_output.txt

# Clean up test files
rm -f test_input.txt processed_output.txt
```

## Text Processing

### Advanced Text Manipulation

```bash
#!/bin/bash
# Script: text_processing.sh

# Text processing with awk, sed, grep
process_log_file() {
    # Create sample log file
    cat << 'EOF' > sample.log
2024-01-15 10:30:15 INFO User john logged in from 192.168.1.10
2024-01-15 10:31:22 WARN Failed login attempt for user admin from 192.168.1.20
2024-01-15 10:32:45 ERROR Database connection failed
2024-01-15 10:33:10 INFO User jane logged in from 192.168.1.15
2024-01-15 10:34:33 WARN Disk space low on /var partition
2024-01-15 10:35:18 INFO User john logged out
2024-01-15 10:36:25 ERROR Network timeout occurred
EOF

    echo "=== Log File Analysis ==="
    echo "Sample log file created with $(wc -l < sample.log) lines"
    
    # Extract specific information using grep
    echo -e "\nERROR entries:"
    grep "ERROR" sample.log
    
    echo -e "\nLogin/logout activities:"
    grep -E "(logged in|logged out)" sample.log
    
    # Process with awk
    echo -e "\nUser activity summary:"
    awk '/logged in/ {users[$4]++} END {for (user in users) print user ": " users[user] " logins"}' sample.log
    
    echo -e "\nHourly log distribution:"
    awk '{split($2, time, ":"); print time[1] ":00"}' sample.log | sort | uniq -c
    
    # Transform with sed
    echo -e "\nFormatted log (date only):"
    sed 's/^\([0-9-]*\) \([0-9:]*\) \([A-Z]*\) \(.*\)/[\1] \3: \4/' sample.log
    
    # Complex processing pipeline
    echo -e "\nIP addresses with login attempts:"
    grep "logged in" sample.log | \
    awk '{print $NF}' | \
    sort | uniq -c | \
    sort -rn
    
    # Clean up
    rm -f sample.log
}

# CSV processing
process_csv_data() {
    # Create sample CSV
    cat << 'EOF' > sample.csv
Name,Age,Department,Salary
John Smith,28,Engineering,75000
Jane Doe,32,Marketing,65000
Bob Johnson,45,Engineering,95000
Alice Brown,29,Sales,55000
Charlie Wilson,38,Marketing,70000
EOF

    echo -e "\n=== CSV Data Processing ==="
    
    # Display original data
    echo "Original CSV data:"
    cat sample.csv
    
    # Extract specific columns
    echo -e "\nNames and Departments:"
    awk -F',' 'NR>1 {print $1 " - " $3}' sample.csv
    
    # Calculate statistics
    echo -e "\nSalary statistics:"
    awk -F',' 'NR>1 {sum+=$4; count++; if($4>max) max=$4; if(min=="" || $4<min) min=$4} 
                END {print "Average: $" int(sum/count); print "Maximum: $" max; print "Minimum: $" min}' sample.csv
    
    # Filter and sort
    echo -e "\nEngineering department (sorted by salary):"
    awk -F',' 'NR==1 {print} $3=="Engineering" {print}' sample.csv | \
    sort -t',' -k4 -rn
    
    # Transform data
    echo -e "\nTransformed data (last name first):"
    awk -F',' 'NR==1 {print "LastName,FirstName,Age,Department,Salary"} 
               NR>1 {split($1, name, " "); print name[2] "," name[1] "," $2 "," $3 "," $4}' sample.csv
    
    # Clean up
    rm -f sample.csv
}

# String manipulation functions
text_utilities() {
    echo -e "\n=== Text Utilities ==="
    
    local sample_text="The Quick Brown Fox Jumps Over The Lazy Dog"
    echo "Original text: $sample_text"
    
    # Word count and analysis
    echo "Word count: $(echo "$sample_text" | wc -w)"
    echo "Character count: $(echo "$sample_text" | wc -c)"
    echo "Unique words: $(echo "$sample_text" | tr ' ' '\n' | sort -u | wc -l)"
    
    # Text transformations
    echo "Lowercase: $(echo "$sample_text" | tr '[:upper:]' '[:lower:]')"
    echo "Uppercase: $(echo "$sample_text" | tr '[:lower:]' '[:upper:]')"
    echo "Reversed: $(echo "$sample_text" | rev)"
    
    # Pattern matching
    echo "Words starting with 'T': $(echo "$sample_text" | grep -oE '\bT[a-zA-Z]*' | tr '\n' ' ')"
    echo "Words with 3 letters: $(echo "$sample_text" | grep -oE '\b[a-zA-Z]{3}\b' | tr '\n' ' ')"
    
    # Text replacement
    echo "Replace 'The' with 'A': $(echo "$sample_text" | sed 's/The/A/g')"
    echo "Remove vowels: $(echo "$sample_text" | tr -d 'aeiouAEIOU')"
}

# Run text processing examples
process_log_file
process_csv_data
text_utilities
```

## System Administration

### System Monitoring and Management

```bash
#!/bin/bash
# Script: system_admin.sh

# System information gathering
get_system_info() {
    echo "=== System Information ==="
    
    # Basic system info
    echo "Hostname: $(hostname)"
    echo "OS: $(uname -s)"
    echo "Kernel: $(uname -r)"
    echo "Architecture: $(uname -m)"
    echo "Uptime: $(uptime | awk -F'up ' '{print $2}' | awk -F',' '{print $1}')"
    
    # CPU information
    echo -e "\n--- CPU Information ---"
    if command -v lscpu >/dev/null; then
        lscpu | grep -E "(Model name|CPU\(s\)|Thread|Core)"
    else
        sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "CPU info not available"
    fi
    
    # Memory information
    echo -e "\n--- Memory Information ---"
    if [ -f /proc/meminfo ]; then
        awk '/MemTotal|MemFree|MemAvailable/ {print $1 " " $2 " " $3}' /proc/meminfo
    else
        # macOS alternative
        echo "Total Memory: $(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024) "GB"}' 2>/dev/null || echo "Unknown")"
    fi
    
    # Disk usage
    echo -e "\n--- Disk Usage ---"
    df -h | grep -E "(Filesystem|/dev/)"
    
    # Network interfaces
    echo -e "\n--- Network Interfaces ---"
    if command -v ip >/dev/null; then
        ip addr show | grep -E "(inet |^[0-9]+:)" | awk '{print $1 $2}'
    else
        ifconfig | grep -E "(^[a-z]|inet )" | awk '{print $1 $2}'
    fi
}

# Process monitoring
monitor_processes() {
    echo -e "\n=== Process Monitoring ==="
    
    # Top CPU consuming processes
    echo "Top 5 CPU consuming processes:"
    if command -v ps >/dev/null; then
        ps aux --sort=-%cpu | head -6 | awk '{printf "%-8s %-6s %-6s %s\n", $1, $2, $3, $11}'
    fi
    
    # Top memory consuming processes
    echo -e "\nTop 5 Memory consuming processes:"
    if command -v ps >/dev/null; then
        ps aux --sort=-%mem | head -6 | awk '{printf "%-8s %-6s %-6s %s\n", $1, $2, $4, $11}'
    fi
    
    # Process count by user
    echo -e "\nProcess count by user:"
    ps aux | awk 'NR>1 {count[$1]++} END {for (user in count) print user ": " count[user]}' | sort -k2 -rn
}

# Service management
manage_services() {
    echo -e "\n=== Service Management ==="
    
    # Function to check if systemctl is available
    if command -v systemctl >/dev/null; then
        echo "System services status:"
        
        # Check common services
        local services=("ssh" "nginx" "apache2" "mysql" "postgresql")
        
        for service in "${services[@]}"; do
            if systemctl list-unit-files | grep -q "^$service"; then
                status=$(systemctl is-active "$service" 2>/dev/null || echo "inactive")
                enabled=$(systemctl is-enabled "$service" 2>/dev/null || echo "disabled")
                echo "  $service: $status ($enabled)"
            fi
        done
        
        # Failed services
        echo -e "\nFailed services:"
        systemctl --failed --no-pager | grep -v "0 loaded units"
        
    else
        echo "systemctl not available (not a systemd system)"
        
        # Alternative for other systems
        echo "Running processes:"
        ps aux | grep -E "(nginx|apache|mysql|ssh)" | grep -v grep
    fi
}

# Log analysis
analyze_logs() {
    echo -e "\n=== Log Analysis ==="
    
    # System log analysis
    local log_files=("/var/log/syslog" "/var/log/messages" "/var/log/system.log")
    local available_log=""
    
    for log_file in "${log_files[@]}"; do
        if [ -r "$log_file" ]; then
            available_log="$log_file"
            break
        fi
    done
    
    if [ -n "$available_log" ]; then
        echo "Analyzing log file: $available_log"
        
        # Recent errors
        echo "Recent errors (last 10):"
        grep -i error "$available_log" | tail -10
        
        # Log activity by hour
        echo -e "\nLog activity by hour (today):"
        grep "$(date '+%b %d')" "$available_log" | \
        awk '{print $3}' | cut -d: -f1 | sort | uniq -c | sort -rn
        
    else
        echo "No readable system log files found"
    fi
    
    # Authentication log analysis
    local auth_logs=("/var/log/auth.log" "/var/log/secure")
    
    for auth_log in "${auth_logs[@]}"; do
        if [ -r "$auth_log" ]; then
            echo -e "\nAuthentication log analysis:"
            echo "Failed login attempts (last 5):"
            grep "Failed password" "$auth_log" | tail -5
            break
        fi
    done
}

# Backup function
create_system_backup() {
    local backup_dir="/tmp/system_backup_$(date +%Y%m%d_%H%M%S)"
    
    echo -e "\n=== System Backup ==="
    echo "Creating backup in: $backup_dir"
    
    mkdir -p "$backup_dir"
    
    # Backup important configuration files
    local config_files=(
        "/etc/passwd"
        "/etc/group"
        "/etc/hosts"
        "/etc/crontab"
    )
    
    echo "Backing up configuration files:"
    for file in "${config_files[@]}"; do
        if [ -r "$file" ]; then
            cp "$file" "$backup_dir/"
            echo "  ✓ Backed up: $file"
        fi
    done
    
    # Create system information snapshot
    {
        echo "=== System Backup Report ==="
        echo "Date: $(date)"
        echo "Hostname: $(hostname)"
        echo "Kernel: $(uname -a)"
        echo ""
        echo "=== Installed Packages ==="
        if command -v dpkg >/dev/null; then
            dpkg -l
        elif command -v rpm >/dev/null; then
            rpm -qa
        elif command -v brew >/dev/null; then
            brew list
        fi
    } > "$backup_dir/system_info.txt"
    
    # Create archive
    tar -czf "${backup_dir}.tar.gz" -C "$(dirname "$backup_dir")" "$(basename "$backup_dir")"
    rm -rf "$backup_dir"
    
    echo "Backup created: ${backup_dir}.tar.gz"
    echo "Backup size: $(ls -lh "${backup_dir}.tar.gz" | awk '{print $5}')"
}

# Run system administration tasks
get_system_info
monitor_processes
manage_services
analyze_logs
create_system_backup
```

---

*This comprehensive Bash scripting guide provides complete coverage from basic syntax to advanced system administration automation.* 