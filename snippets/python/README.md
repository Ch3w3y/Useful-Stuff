# Python Code Snippets Collection

## Overview

Comprehensive collection of Python code snippets for rapid development, covering common tasks, utilities, design patterns, and best practices. Ready-to-use code for everyday programming challenges.

## Table of Contents

- [Data Structures & Algorithms](#data-structures--algorithms)
- [File & I/O Operations](#file--io-operations)
- [String Manipulation](#string-manipulation)
- [Date & Time Utilities](#date--time-utilities)
- [Web & API Utilities](#web--api-utilities)
- [Database Operations](#database-operations)
- [System & OS Utilities](#system--os-utilities)
- [Design Patterns](#design-patterns)

## Data Structures & Algorithms

### List Operations

```python
# List comprehensions and filtering
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Basic list comprehension
squares = [x**2 for x in numbers]
evens = [x for x in numbers if x % 2 == 0]

# Nested list comprehension
matrix = [[i*j for j in range(1, 4)] for i in range(1, 4)]

# Flatten nested lists
nested = [[1, 2], [3, 4], [5, 6]]
flattened = [item for sublist in nested for item in sublist]

# Remove duplicates while preserving order
def remove_duplicates(lst):
    seen = set()
    return [x for x in lst if not (x in seen or seen.add(x))]

# Chunk list into smaller lists
def chunk_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

# Find most common elements
from collections import Counter
def most_common(lst, n=3):
    return Counter(lst).most_common(n)

# Example usage
data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
print(f"Most common: {most_common(data)}")
print(f"Chunked: {chunk_list(numbers, 3)}")
```

### Dictionary Operations

```python
# Dictionary comprehensions and operations
data = {'a': 1, 'b': 2, 'c': 3, 'd': 4}

# Dictionary comprehension
squared_dict = {k: v**2 for k, v in data.items()}
filtered_dict = {k: v for k, v in data.items() if v > 2}

# Merge dictionaries (Python 3.9+)
dict1 = {'a': 1, 'b': 2}
dict2 = {'c': 3, 'd': 4}
merged = dict1 | dict2

# Safe dictionary access
def safe_get(dictionary, key, default=None):
    return dictionary.get(key, default)

# Invert dictionary
def invert_dict(d):
    return {v: k for k, v in d.items()}

# Group by key function
from collections import defaultdict
def group_by(items, key_func):
    groups = defaultdict(list)
    for item in items:
        groups[key_func(item)].append(item)
    return dict(groups)

# Example: Group words by length
words = ['apple', 'banana', 'cherry', 'date', 'elderberry']
grouped = group_by(words, len)
print(f"Grouped by length: {grouped}")

# Nested dictionary access
def deep_get(dictionary, keys, default=None):
    """Safely access nested dictionary values"""
    for key in keys:
        if isinstance(dictionary, dict) and key in dictionary:
            dictionary = dictionary[key]
        else:
            return default
    return dictionary

# Example usage
nested_dict = {'a': {'b': {'c': 'value'}}}
result = deep_get(nested_dict, ['a', 'b', 'c'])  # Returns 'value'
```

### Set Operations

```python
# Set operations and utilities
set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7, 8}

# Basic set operations
union = set1 | set2  # or set1.union(set2)
intersection = set1 & set2  # or set1.intersection(set2)
difference = set1 - set2  # or set1.difference(set2)
symmetric_diff = set1 ^ set2  # or set1.symmetric_difference(set2)

# Check relationships
is_subset = set1.issubset(set2)
is_superset = set1.issuperset(set2)
is_disjoint = set1.isdisjoint(set2)

# Find unique elements across multiple lists
def unique_across_lists(*lists):
    all_items = set()
    for lst in lists:
        all_items.update(lst)
    return list(all_items)

# Example
list1 = [1, 2, 3]
list2 = [3, 4, 5]
list3 = [5, 6, 7]
unique_items = unique_across_lists(list1, list2, list3)
```

### Sorting and Searching

```python
# Advanced sorting techniques
from operator import itemgetter, attrgetter

# Sort list of dictionaries
students = [
    {'name': 'Alice', 'grade': 85, 'age': 20},
    {'name': 'Bob', 'grade': 90, 'age': 19},
    {'name': 'Charlie', 'grade': 85, 'age': 21}
]

# Sort by single key
sorted_by_grade = sorted(students, key=itemgetter('grade'), reverse=True)

# Sort by multiple keys
sorted_multi = sorted(students, key=itemgetter('grade', 'age'), reverse=True)

# Custom sorting function
def custom_sort_key(student):
    return (student['grade'], -student['age'])  # Grade desc, age asc

sorted_custom = sorted(students, key=custom_sort_key, reverse=True)

# Binary search implementation
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1  # Not found

# Quick sort implementation
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quicksort(left) + middle + quicksort(right)

# Example usage
numbers = [64, 34, 25, 12, 22, 11, 90]
sorted_numbers = quicksort(numbers)
index = binary_search(sorted_numbers, 25)
```

## File & I/O Operations

### File Reading and Writing

```python
import os
import json
import csv
from pathlib import Path
import pickle

# Safe file reading with context manager
def read_file_safe(filepath, encoding='utf-8'):
    """Safely read file with error handling"""
    try:
        with open(filepath, 'r', encoding=encoding) as file:
            return file.read()
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

# Write file with backup
def write_file_with_backup(filepath, content, backup=True):
    """Write file with optional backup of existing file"""
    if backup and os.path.exists(filepath):
        backup_path = f"{filepath}.backup"
        os.rename(filepath, backup_path)
    
    with open(filepath, 'w', encoding='utf-8') as file:
        file.write(content)

# Read file line by line (memory efficient)
def read_large_file(filepath):
    """Generator for reading large files line by line"""
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            yield line.strip()

# JSON operations
def read_json(filepath):
    """Read JSON file with error handling"""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
        return None

def write_json(filepath, data, indent=2):
    """Write data to JSON file"""
    with open(filepath, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=indent, ensure_ascii=False)

# CSV operations
def read_csv_as_dict(filepath):
    """Read CSV file as list of dictionaries"""
    with open(filepath, 'r', encoding='utf-8') as file:
        return list(csv.DictReader(file))

def write_csv_from_dict(filepath, data, fieldnames=None):
    """Write list of dictionaries to CSV"""
    if not data:
        return
    
    if fieldnames is None:
        fieldnames = data[0].keys()
    
    with open(filepath, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

# Binary file operations
def save_pickle(obj, filepath):
    """Save object to pickle file"""
    with open(filepath, 'wb') as file:
        pickle.dump(obj, file)

def load_pickle(filepath):
    """Load object from pickle file"""
    with open(filepath, 'rb') as file:
        return pickle.load(file)
```

### Directory Operations

```python
import shutil
import glob
from datetime import datetime

# Directory utilities
def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def list_files(directory, pattern="*", recursive=False):
    """List files matching pattern"""
    path = Path(directory)
    if recursive:
        return list(path.rglob(pattern))
    else:
        return list(path.glob(pattern))

def get_file_info(filepath):
    """Get comprehensive file information"""
    path = Path(filepath)
    if not path.exists():
        return None
    
    stat = path.stat()
    return {
        'name': path.name,
        'size': stat.st_size,
        'modified': datetime.fromtimestamp(stat.st_mtime),
        'created': datetime.fromtimestamp(stat.st_ctime),
        'is_file': path.is_file(),
        'is_dir': path.is_dir(),
        'extension': path.suffix,
        'parent': str(path.parent)
    }

def copy_files(src_pattern, dest_dir, preserve_structure=False):
    """Copy files matching pattern to destination"""
    ensure_dir(dest_dir)
    
    for src_file in glob.glob(src_pattern, recursive=True):
        if preserve_structure:
            # Preserve directory structure
            rel_path = os.path.relpath(src_file)
            dest_path = os.path.join(dest_dir, rel_path)
            ensure_dir(os.path.dirname(dest_path))
        else:
            # Flat copy
            dest_path = os.path.join(dest_dir, os.path.basename(src_file))
        
        shutil.copy2(src_file, dest_path)

def clean_directory(directory, older_than_days=30):
    """Remove files older than specified days"""
    cutoff_time = datetime.now().timestamp() - (older_than_days * 24 * 60 * 60)
    
    for file_path in Path(directory).rglob('*'):
        if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
            file_path.unlink()
            print(f"Removed: {file_path}")

# Archive operations
def create_archive(source_dir, archive_path, format='zip'):
    """Create archive from directory"""
    shutil.make_archive(archive_path.replace(f'.{format}', ''), format, source_dir)

def extract_archive(archive_path, extract_to):
    """Extract archive to directory"""
    shutil.unpack_archive(archive_path, extract_to)
```

## String Manipulation

### Text Processing

```python
import re
import string
from collections import Counter

# String cleaning and validation
def clean_text(text):
    """Clean and normalize text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters (keep alphanumeric and basic punctuation)
    text = re.sub(r'[^\w\s\.\,\!\?\-]', '', text)
    
    return text

def is_valid_email(email):
    """Validate email address"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def is_valid_phone(phone):
    """Validate phone number (US format)"""
    pattern = r'^\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})$'
    return re.match(pattern, phone) is not None

def extract_urls(text):
    """Extract URLs from text"""
    pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(pattern, text)

def extract_hashtags(text):
    """Extract hashtags from text"""
    return re.findall(r'#\w+', text)

def extract_mentions(text):
    """Extract @mentions from text"""
    return re.findall(r'@\w+', text)

# Text analysis
def word_frequency(text, top_n=10):
    """Get word frequency analysis"""
    # Clean and split text
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    words = [word for word in words if word not in stop_words]
    
    return Counter(words).most_common(top_n)

def text_statistics(text):
    """Get comprehensive text statistics"""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    paragraphs = text.split('\n\n')
    
    return {
        'characters': len(text),
        'characters_no_spaces': len(text.replace(' ', '')),
        'words': len(words),
        'sentences': len([s for s in sentences if s.strip()]),
        'paragraphs': len([p for p in paragraphs if p.strip()]),
        'avg_words_per_sentence': len(words) / max(len(sentences), 1),
        'avg_chars_per_word': sum(len(word) for word in words) / max(len(words), 1)
    }

# String formatting and templates
def format_currency(amount, currency='USD'):
    """Format number as currency"""
    symbols = {'USD': '$', 'EUR': '€', 'GBP': '£', 'JPY': '¥'}
    symbol = symbols.get(currency, currency)
    return f"{symbol}{amount:,.2f}"

def format_file_size(bytes_size):
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"

def slugify(text):
    """Convert text to URL-friendly slug"""
    # Convert to lowercase and replace spaces with hyphens
    slug = re.sub(r'[^\w\s-]', '', text.lower())
    slug = re.sub(r'[-\s]+', '-', slug)
    return slug.strip('-')

# Template string replacement
def safe_format(template, **kwargs):
    """Safe string formatting that doesn't fail on missing keys"""
    try:
        return template.format(**kwargs)
    except KeyError as e:
        print(f"Missing template key: {e}")
        return template

# Example usage
sample_text = "Hello World! This is a sample text with #hashtag and @mention. Visit https://example.com"
print(f"URLs: {extract_urls(sample_text)}")
print(f"Hashtags: {extract_hashtags(sample_text)}")
print(f"Statistics: {text_statistics(sample_text)}")
```

## Date & Time Utilities

### Date and Time Operations

```python
from datetime import datetime, timedelta, timezone
import calendar
import time

# Date parsing and formatting
def parse_date(date_string, formats=None):
    """Parse date string with multiple format attempts"""
    if formats is None:
        formats = [
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y-%m-%d %H:%M:%S',
            '%m/%d/%Y %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ'
        ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue
    
    raise ValueError(f"Unable to parse date: {date_string}")

def format_date(dt, format_type='iso'):
    """Format datetime with predefined formats"""
    formats = {
        'iso': '%Y-%m-%d',
        'us': '%m/%d/%Y',
        'eu': '%d/%m/%Y',
        'long': '%B %d, %Y',
        'short': '%b %d, %Y',
        'datetime': '%Y-%m-%d %H:%M:%S',
        'timestamp': '%Y%m%d_%H%M%S'
    }
    
    return dt.strftime(formats.get(format_type, format_type))

# Date calculations
def add_business_days(start_date, days):
    """Add business days (excluding weekends)"""
    current_date = start_date
    days_added = 0
    
    while days_added < days:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5:  # Monday = 0, Sunday = 6
            days_added += 1
    
    return current_date

def get_date_range(start_date, end_date, step_days=1):
    """Generate date range between two dates"""
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=step_days)

def get_month_boundaries(year, month):
    """Get first and last day of month"""
    first_day = datetime(year, month, 1)
    last_day = datetime(year, month, calendar.monthrange(year, month)[1])
    return first_day, last_day

def get_quarter_boundaries(year, quarter):
    """Get first and last day of quarter"""
    if quarter == 1:
        start_month, end_month = 1, 3
    elif quarter == 2:
        start_month, end_month = 4, 6
    elif quarter == 3:
        start_month, end_month = 7, 9
    else:
        start_month, end_month = 10, 12
    
    first_day = datetime(year, start_month, 1)
    last_day = datetime(year, end_month, calendar.monthrange(year, end_month)[1])
    return first_day, last_day

# Time utilities
def time_it(func):
    """Decorator to measure function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def human_readable_duration(seconds):
    """Convert seconds to human readable duration"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    elif seconds < 86400:
        return f"{seconds/3600:.1f} hours"
    else:
        return f"{seconds/86400:.1f} days"

def time_ago(dt):
    """Get human readable time ago string"""
    now = datetime.now()
    diff = now - dt
    
    if diff.days > 0:
        return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif diff.seconds > 60:
        minutes = diff.seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    else:
        return "Just now"

# Timezone utilities
def convert_timezone(dt, from_tz, to_tz):
    """Convert datetime between timezones"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=from_tz)
    return dt.astimezone(to_tz)

def get_utc_timestamp():
    """Get current UTC timestamp"""
    return datetime.now(timezone.utc).timestamp()

# Example usage
now = datetime.now()
print(f"Formatted date: {format_date(now, 'long')}")
print(f"Time ago: {time_ago(now - timedelta(hours=2))}")
print(f"Business days: {add_business_days(now, 5)}")
```

## Web & API Utilities

### HTTP Requests and API Handling

```python
import requests
import json
import time
from urllib.parse import urljoin, urlparse
from functools import wraps

# HTTP request utilities
class APIClient:
    """Reusable API client with common functionality"""
    
    def __init__(self, base_url, headers=None, timeout=30):
        self.base_url = base_url
        self.session = requests.Session()
        self.timeout = timeout
        
        if headers:
            self.session.headers.update(headers)
    
    def request(self, method, endpoint, **kwargs):
        """Make HTTP request with error handling"""
        url = urljoin(self.base_url, endpoint)
        
        try:
            response = self.session.request(
                method, url, timeout=self.timeout, **kwargs
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None
    
    def get(self, endpoint, params=None):
        """GET request"""
        return self.request('GET', endpoint, params=params)
    
    def post(self, endpoint, data=None, json_data=None):
        """POST request"""
        if json_data:
            return self.request('POST', endpoint, json=json_data)
        return self.request('POST', endpoint, data=data)
    
    def put(self, endpoint, data=None, json_data=None):
        """PUT request"""
        if json_data:
            return self.request('PUT', endpoint, json=json_data)
        return self.request('PUT', endpoint, data=data)
    
    def delete(self, endpoint):
        """DELETE request"""
        return self.request('DELETE', endpoint)

# Rate limiting decorator
def rate_limit(calls_per_second=1):
    """Rate limiting decorator"""
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

# Retry decorator
def retry(max_attempts=3, delay=1, backoff=2):
    """Retry decorator with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay
            
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        raise e
                    
                    print(f"Attempt {attempts} failed: {e}. Retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            return None
        return wrapper
    return decorator

# URL utilities
def validate_url(url):
    """Validate URL format"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def extract_domain(url):
    """Extract domain from URL"""
    return urlparse(url).netloc

def build_query_string(params):
    """Build query string from parameters"""
    from urllib.parse import urlencode
    return urlencode(params)

# JSON utilities for APIs
def safe_json_loads(text, default=None):
    """Safely parse JSON with fallback"""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default

def flatten_json(data, separator='_'):
    """Flatten nested JSON structure"""
    def _flatten(obj, parent_key=''):
        items = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{parent_key}{separator}{k}" if parent_key else k
                items.extend(_flatten(v, new_key).items())
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                new_key = f"{parent_key}{separator}{i}" if parent_key else str(i)
                items.extend(_flatten(v, new_key).items())
        else:
            return {parent_key: obj}
        return dict(items)
    
    return _flatten(data)

# Example usage
@rate_limit(calls_per_second=2)
@retry(max_attempts=3)
def fetch_data(url):
    """Example API call with rate limiting and retry"""
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

# API client example
api = APIClient('https://api.example.com', headers={'Authorization': 'Bearer token'})
response = api.get('/users')
if response:
    users = response.json()
```

---

*This comprehensive Python snippets collection provides ready-to-use code for common programming tasks and patterns.* 