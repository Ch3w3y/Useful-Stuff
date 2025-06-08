# Configuration Templates & Project Scaffolding

## Overview

Comprehensive collection of configuration templates, project boilerplates, and scaffolding tools for rapidly setting up development projects across different technologies and architectural patterns. Includes automated project generation and best practice configurations.

## Table of Contents

- [Project Templates](#project-templates)
- [Infrastructure Templates](#infrastructure-templates)
- [Configuration Templates](#configuration-templates)
- [Development Templates](#development-templates)
- [CI/CD Templates](#cicd-templates)
- [Documentation Templates](#documentation-templates)
- [Scaffolding Tools](#scaffolding-tools)
- [Template Management](#template-management)

## Project Templates

### Python Project Template

```bash
#!/bin/bash
# templates/python-project/generate.sh - Python project generator

set -euo pipefail

PROJECT_NAME="${1:-my-python-project}"
PROJECT_TYPE="${2:-library}"
PYTHON_VERSION="${3:-3.10}"

echo "Generating Python $PROJECT_TYPE project: $PROJECT_NAME"

# Create project structure
mkdir -p "$PROJECT_NAME"/{src,tests,docs,scripts}
cd "$PROJECT_NAME"

# Create main package directory
mkdir -p "src/$(echo $PROJECT_NAME | tr '-' '_')"

# Generate pyproject.toml
cat > pyproject.toml << EOF
[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm[toml]>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "$PROJECT_NAME"
dynamic = ["version"]
description = "A Python $PROJECT_TYPE"
readme = "README.md"
license = {file = "LICENSE"}
authors = [
    {name = "Your Name", email = "your.email@example.com"},
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]
requires-python = ">=$PYTHON_VERSION"
dependencies = [
    "click>=8.0.0",
    "pydantic>=1.10.0",
    "requests>=2.28.0",
]

[project.optional-dependencies]
dev = [
    "black>=22.0.0",
    "flake8>=4.0.0",
    "mypy>=0.950",
    "pytest>=7.0.0",
    "pytest-cov>=3.0.0",
    "pre-commit>=2.20.0",
]
docs = [
    "sphinx>=5.0.0",
    "sphinx-rtd-theme>=1.0.0",
    "myst-parser>=0.18.0",
]

[project.urls]
Homepage = "https://github.com/yourusername/$PROJECT_NAME"
Repository = "https://github.com/yourusername/$PROJECT_NAME"
Documentation = "https://$PROJECT_NAME.readthedocs.io/"
"Bug Tracker" = "https://github.com/yourusername/$PROJECT_NAME/issues"

[project.scripts]
$PROJECT_NAME = "$(echo $PROJECT_NAME | tr '-' '_').cli:main"

[tool.setuptools]
packages = ["src"]

[tool.setuptools_scm]
write_to = "src/$(echo $PROJECT_NAME | tr '-' '_')/_version.py"

[tool.black]
line-length = 88
target-version = ["py310"]
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

[tool.mypy]
python_version = "$PYTHON_VERSION"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--cov=src",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-report=xml",
    "--strict-markers",
    "--strict-config",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
]

[tool.coverage.run]
source = ["src"]
omit = ["*/tests/*", "*/test_*.py"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "if settings.DEBUG",
    "raise AssertionError",
    "raise NotImplementedError",
    "if 0:",
    "if __name__ == .__main__.:",
    "class .*\bProtocol\):",
    "@(abc\.)?abstractmethod",
]
EOF

# Create main module
cat > "src/$(echo $PROJECT_NAME | tr '-' '_')/__init__.py" << 'EOF'
"""A Python project template."""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

__all__ = ["__version__"]
EOF

# Create CLI module
cat > "src/$(echo $PROJECT_NAME | tr '-' '_')/cli.py" << 'EOF'
"""Command line interface."""

import click
from . import __version__


@click.group()
@click.version_option(version=__version__)
def main():
    """Main CLI entry point."""
    pass


@main.command()
@click.option("--name", default="World", help="Name to greet")
def hello(name: str):
    """Say hello."""
    click.echo(f"Hello {name}!")


if __name__ == "__main__":
    main()
EOF

# Create core module
cat > "src/$(echo $PROJECT_NAME | tr '-' '_')/core.py" << 'EOF'
"""Core functionality."""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class CoreManager:
    """Core functionality manager."""
    
    def __init__(self):
        self.config: Dict[str, Any] = {}
        logger.info("CoreManager initialized")
    
    def process(self, data: str) -> str:
        """Process input data."""
        logger.debug(f"Processing data: {data}")
        return f"Processed: {data}"
EOF

# Create test files
cat > tests/__init__.py << 'EOF'
"""Test package."""
EOF

cat > tests/conftest.py << 'EOF'
"""Test configuration."""

import pytest
from click.testing import CliRunner


@pytest.fixture
def runner():
    """CLI test runner."""
    return CliRunner()
EOF

cat > "tests/test_$(echo $PROJECT_NAME | tr '-' '_').py" << 'EOF'
"""Test main module."""

import pytest
from click.testing import CliRunner

from src.$(echo $PROJECT_NAME | tr '-' '_') import __version__
from src.$(echo $PROJECT_NAME | tr '-' '_').cli import main
from src.$(echo $PROJECT_NAME | tr '-' '_').core import CoreManager


def test_version():
    """Test version is available."""
    assert __version__ is not None


def test_cli_hello(runner: CliRunner):
    """Test CLI hello command."""
    result = runner.invoke(main, ["hello"])
    assert result.exit_code == 0
    assert "Hello World!" in result.output


def test_cli_hello_with_name(runner: CliRunner):
    """Test CLI hello command with name."""
    result = runner.invoke(main, ["hello", "--name", "Test"])
    assert result.exit_code == 0
    assert "Hello Test!" in result.output


def test_core_manager():
    """Test CoreManager."""
    manager = CoreManager()
    result = manager.process("test data")
    assert result == "Processed: test data"
EOF

# Create README
cat > README.md << EOF
# $PROJECT_NAME

A Python $PROJECT_TYPE for [brief description].

## Features

- Feature 1
- Feature 2
- Feature 3

## Installation

\`\`\`bash
pip install $PROJECT_NAME
\`\`\`

## Quick Start

\`\`\`python
from $(echo $PROJECT_NAME | tr '-' '_') import CoreManager

manager = CoreManager()
result = manager.process("your data")
print(result)
\`\`\`

## CLI Usage

\`\`\`bash
$PROJECT_NAME hello --name "World"
\`\`\`

## Development

### Setup

\`\`\`bash
git clone https://github.com/yourusername/$PROJECT_NAME.git
cd $PROJECT_NAME
pip install -e ".[dev]"
pre-commit install
\`\`\`

### Testing

\`\`\`bash
pytest
\`\`\`

### Linting

\`\`\`bash
black .
flake8 .
mypy .
\`\`\`

## License

MIT License - see LICENSE file for details.
EOF

# Create LICENSE
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2023 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

# Create development files
cat > .gitignore << 'EOF'
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
EOF

cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: debug-statements

  - repo: https://github.com/psf/black
    rev: 22.10.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/flake8
    rev: 5.0.4
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.991
    hooks:
      - id: mypy
        additional_dependencies: [types-requests]
EOF

# Make scripts executable
chmod +x "scripts/$(echo $PROJECT_NAME | tr '-' '_').sh" 2>/dev/null || true

echo "Python project '$PROJECT_NAME' generated successfully!"
echo "Next steps:"
echo "1. cd $PROJECT_NAME"
echo "2. pip install -e \".[dev]\""
echo "3. pre-commit install"
echo "4. pytest"
```

### Node.js Project Template

```bash
#!/bin/bash
# templates/nodejs-project/generate.sh - Node.js project generator

set -euo pipefail

PROJECT_NAME="${1:-my-node-project}"
PROJECT_TYPE="${2:-api}"
FRAMEWORK="${3:-express}"

echo "Generating Node.js $PROJECT_TYPE project: $PROJECT_NAME"

mkdir -p "$PROJECT_NAME"
cd "$PROJECT_NAME"

# Initialize package.json
cat > package.json << EOF
{
  "name": "$PROJECT_NAME",
  "version": "1.0.0",
  "description": "A Node.js $PROJECT_TYPE project",
  "main": "dist/index.js",
  "scripts": {
    "build": "tsc",
    "start": "node dist/index.js",
    "dev": "nodemon src/index.ts",
    "test": "jest",
    "test:watch": "jest --watch",
    "test:coverage": "jest --coverage",
    "lint": "eslint src/**/*.ts",
    "lint:fix": "eslint src/**/*.ts --fix",
    "format": "prettier --write src/**/*.ts",
    "prepare": "husky install",
    "clean": "rimraf dist"
  },
  "keywords": ["nodejs", "$PROJECT_TYPE", "$FRAMEWORK"],
  "author": "Your Name <your.email@example.com>",
  "license": "MIT",
  "dependencies": {
    "express": "^4.18.2",
    "cors": "^2.8.5",
    "helmet": "^6.0.1",
    "morgan": "^1.10.0",
    "dotenv": "^16.0.3"
  },
  "devDependencies": {
    "@types/node": "^18.14.2",
    "@types/express": "^4.17.17",
    "@types/cors": "^2.8.13",
    "@types/morgan": "^1.9.4",
    "@types/jest": "^29.4.0",
    "typescript": "^4.9.5",
    "ts-node": "^10.9.1",
    "nodemon": "^2.0.20",
    "jest": "^29.4.3",
    "ts-jest": "^29.0.5",
    "eslint": "^8.35.0",
    "@typescript-eslint/parser": "^5.54.0",
    "@typescript-eslint/eslint-plugin": "^5.54.0",
    "prettier": "^2.8.4",
    "husky": "^8.0.3",
    "lint-staged": "^13.1.2",
    "rimraf": "^4.1.2"
  },
  "lint-staged": {
    "*.{ts,js}": [
      "eslint --fix",
      "prettier --write"
    ]
  }
}
EOF

# Create TypeScript configuration
cat > tsconfig.json << 'EOF'
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "lib": ["ES2020"],
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "moduleResolution": "node",
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "experimentalDecorators": true,
    "emitDecoratorMetadata": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "tests"]
}
EOF

# Create project structure
mkdir -p src/{controllers,middleware,models,routes,services,utils,types}
mkdir -p tests
mkdir -p docs

# Create main application file
cat > src/index.ts << 'EOF'
import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import dotenv from 'dotenv';

import { errorHandler } from './middleware/errorHandler';
import { notFoundHandler } from './middleware/notFoundHandler';
import { apiRoutes } from './routes';

// Load environment variables
dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(helmet());
app.use(cors());
app.use(morgan('combined'));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Health check endpoint
app.get('/health', (req, res) => {
  res.status(200).json({
    status: 'OK',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
  });
});

// API routes
app.use('/api', apiRoutes);

// Error handling
app.use(notFoundHandler);
app.use(errorHandler);

// Start server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

export default app;
EOF

# Create middleware
cat > src/middleware/errorHandler.ts << 'EOF'
import { Request, Response, NextFunction } from 'express';

export interface AppError extends Error {
  statusCode?: number;
  isOperational?: boolean;
}

export const errorHandler = (
  err: AppError,
  req: Request,
  res: Response,
  next: NextFunction
) => {
  const { statusCode = 500, message } = err;

  console.error(err.stack);

  res.status(statusCode).json({
    status: 'error',
    statusCode,
    message: statusCode === 500 ? 'Internal Server Error' : message,
    ...(process.env.NODE_ENV === 'development' && { stack: err.stack }),
  });
};
EOF

cat > src/middleware/notFoundHandler.ts << 'EOF'
import { Request, Response } from 'express';

export const notFoundHandler = (req: Request, res: Response) => {
  res.status(404).json({
    status: 'error',
    statusCode: 404,
    message: 'Resource not found',
    path: req.originalUrl,
  });
};
EOF

# Create routes
cat > src/routes/index.ts << 'EOF'
import { Router } from 'express';
import { userRoutes } from './userRoutes';

const router = Router();

router.use('/users', userRoutes);

export { router as apiRoutes };
EOF

cat > src/routes/userRoutes.ts << 'EOF'
import { Router } from 'express';
import { UserController } from '../controllers/UserController';

const router = Router();
const userController = new UserController();

router.get('/', userController.getUsers);
router.get('/:id', userController.getUserById);
router.post('/', userController.createUser);
router.put('/:id', userController.updateUser);
router.delete('/:id', userController.deleteUser);

export { router as userRoutes };
EOF

# Create controller
cat > src/controllers/UserController.ts << 'EOF'
import { Request, Response, NextFunction } from 'express';
import { UserService } from '../services/UserService';
import { CreateUserDto, UpdateUserDto } from '../types/user';

export class UserController {
  private userService = new UserService();

  getUsers = async (req: Request, res: Response, next: NextFunction) => {
    try {
      const users = await this.userService.getAllUsers();
      res.json(users);
    } catch (error) {
      next(error);
    }
  };

  getUserById = async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { id } = req.params;
      const user = await this.userService.getUserById(id);
      
      if (!user) {
        return res.status(404).json({ message: 'User not found' });
      }
      
      res.json(user);
    } catch (error) {
      next(error);
    }
  };

  createUser = async (req: Request, res: Response, next: NextFunction) => {
    try {
      const userData: CreateUserDto = req.body;
      const user = await this.userService.createUser(userData);
      res.status(201).json(user);
    } catch (error) {
      next(error);
    }
  };

  updateUser = async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { id } = req.params;
      const userData: UpdateUserDto = req.body;
      const user = await this.userService.updateUser(id, userData);
      
      if (!user) {
        return res.status(404).json({ message: 'User not found' });
      }
      
      res.json(user);
    } catch (error) {
      next(error);
    }
  };

  deleteUser = async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { id } = req.params;
      await this.userService.deleteUser(id);
      res.status(204).send();
    } catch (error) {
      next(error);
    }
  };
}
EOF

# Create service
cat > src/services/UserService.ts << 'EOF'
import { User, CreateUserDto, UpdateUserDto } from '../types/user';

export class UserService {
  private users: User[] = [
    { id: '1', name: 'John Doe', email: 'john@example.com', createdAt: new Date() },
    { id: '2', name: 'Jane Smith', email: 'jane@example.com', createdAt: new Date() },
  ];

  async getAllUsers(): Promise<User[]> {
    return this.users;
  }

  async getUserById(id: string): Promise<User | undefined> {
    return this.users.find(user => user.id === id);
  }

  async createUser(userData: CreateUserDto): Promise<User> {
    const newUser: User = {
      id: Date.now().toString(),
      ...userData,
      createdAt: new Date(),
    };
    
    this.users.push(newUser);
    return newUser;
  }

  async updateUser(id: string, userData: UpdateUserDto): Promise<User | undefined> {
    const userIndex = this.users.findIndex(user => user.id === id);
    
    if (userIndex === -1) {
      return undefined;
    }
    
    this.users[userIndex] = { ...this.users[userIndex], ...userData };
    return this.users[userIndex];
  }

  async deleteUser(id: string): Promise<void> {
    this.users = this.users.filter(user => user.id !== id);
  }
}
EOF

# Create types
cat > src/types/user.ts << 'EOF'
export interface User {
  id: string;
  name: string;
  email: string;
  createdAt: Date;
}

export interface CreateUserDto {
  name: string;
  email: string;
}

export interface UpdateUserDto {
  name?: string;
  email?: string;
}
EOF

# Create environment file
cat > .env.example << 'EOF'
NODE_ENV=development
PORT=3000
DATABASE_URL=postgresql://user:password@localhost:5432/database
JWT_SECRET=your-secret-key
EOF

# Create test file
cat > tests/app.test.ts << 'EOF'
import request from 'supertest';
import app from '../src/index';

describe('API Tests', () => {
  it('should return health status', async () => {
    const response = await request(app)
      .get('/health')
      .expect(200);

    expect(response.body).toHaveProperty('status', 'OK');
    expect(response.body).toHaveProperty('timestamp');
  });

  it('should get all users', async () => {
    const response = await request(app)
      .get('/api/users')
      .expect(200);

    expect(Array.isArray(response.body)).toBe(true);
  });

  it('should create a new user', async () => {
    const newUser = {
      name: 'Test User',
      email: 'test@example.com',
    };

    const response = await request(app)
      .post('/api/users')
      .send(newUser)
      .expect(201);

    expect(response.body).toHaveProperty('id');
    expect(response.body.name).toBe(newUser.name);
    expect(response.body.email).toBe(newUser.email);
  });
});
EOF

echo "Node.js project '$PROJECT_NAME' generated successfully!"
echo "Next steps:"
echo "1. cd $PROJECT_NAME"
echo "2. npm install"
echo "3. npm run dev"
```

## Infrastructure Templates

### Terraform AWS Infrastructure Template

```hcl
# templates/terraform-aws/main.tf - AWS infrastructure template

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
  
  backend "s3" {
    bucket = "your-terraform-state-bucket"
    key    = "infrastructure/terraform.tfstate"
    region = "us-west-2"
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Environment = var.environment
      Project     = var.project_name
      ManagedBy   = "terraform"
    }
  }
}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "development"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "my-project"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# VPC
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name = "${var.project_name}-vpc"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  
  tags = {
    Name = "${var.project_name}-igw"
  }
}

# Public Subnets
resource "aws_subnet" "public" {
  count = 2
  
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  
  tags = {
    Name = "${var.project_name}-public-subnet-${count.index + 1}"
    Type = "public"
  }
}

# Private Subnets
resource "aws_subnet" "private" {
  count = 2
  
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + 10)
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  tags = {
    Name = "${var.project_name}-private-subnet-${count.index + 1}"
    Type = "private"
  }
}

# NAT Gateways
resource "aws_eip" "nat" {
  count = 2
  
  domain = "vpc"
  
  tags = {
    Name = "${var.project_name}-nat-eip-${count.index + 1}"
  }
  
  depends_on = [aws_internet_gateway.main]
}

resource "aws_nat_gateway" "main" {
  count = 2
  
  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id
  
  tags = {
    Name = "${var.project_name}-nat-gateway-${count.index + 1}"
  }
  
  depends_on = [aws_internet_gateway.main]
}

# Route Tables
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
  
  tags = {
    Name = "${var.project_name}-public-rt"
  }
}

resource "aws_route_table" "private" {
  count = 2
  
  vpc_id = aws_vpc.main.id
  
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main[count.index].id
  }
  
  tags = {
    Name = "${var.project_name}-private-rt-${count.index + 1}"
  }
}

# Route Table Associations
resource "aws_route_table_association" "public" {
  count = 2
  
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count = 2
  
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}

# Security Groups
resource "aws_security_group" "web" {
  name_prefix = "${var.project_name}-web-"
  vpc_id      = aws_vpc.main.id
  
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "${var.project_name}-web-sg"
  }
}

resource "aws_security_group" "database" {
  name_prefix = "${var.project_name}-db-"
  vpc_id      = aws_vpc.main.id
  
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.web.id]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "${var.project_name}-db-sg"
  }
}

# Outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = aws_subnet.private[*].id
}

output "security_group_web_id" {
  description = "ID of the web security group"
  value       = aws_security_group.web.id
}

output "security_group_database_id" {
  description = "ID of the database security group"
  value       = aws_security_group.database.id
}
```

### Kubernetes Deployment Template

```yaml
# templates/kubernetes/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: my-app
  labels:
    name: my-app
    environment: development

---
# templates/kubernetes/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: my-app
data:
  NODE_ENV: "production"
  LOG_LEVEL: "info"
  DATABASE_HOST: "postgres-service"
  DATABASE_PORT: "5432"
  REDIS_HOST: "redis-service"
  REDIS_PORT: "6379"

---
# templates/kubernetes/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
  namespace: my-app
type: Opaque
data:
  DATABASE_PASSWORD: cGFzc3dvcmQ=  # base64 encoded 'password'
  JWT_SECRET: c2VjcmV0LWtleQ==      # base64 encoded 'secret-key'

---
# templates/kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-deployment
  namespace: my-app
  labels:
    app: my-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: app
        image: my-app:latest
        ports:
        - containerPort: 3000
        envFrom:
        - configMapRef:
            name: app-config
        - secretRef:
            name: app-secrets
        resources:
          limits:
            cpu: 500m
            memory: 512Mi
          requests:
            cpu: 250m
            memory: 256Mi
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 5
      imagePullSecrets:
      - name: regcred

---
# templates/kubernetes/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: app-service
  namespace: my-app
  labels:
    app: my-app
spec:
  selector:
    app: my-app
  ports:
  - protocol: TCP
    port: 80
    targetPort: 3000
  type: ClusterIP

---
# templates/kubernetes/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: app-ingress
  namespace: my-app
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  tls:
  - hosts:
    - my-app.example.com
    secretName: app-tls
  rules:
  - host: my-app.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: app-service
            port:
              number: 80

---
# templates/kubernetes/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: app-hpa
  namespace: my-app
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: app-deployment
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Scaffolding Tools

### Universal Project Generator

```bash
#!/bin/bash
# scripts/project-generator.sh - Universal project scaffolding tool

set -euo pipefail

TEMPLATES_DIR="${TEMPLATES_DIR:-$HOME/.project-templates}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to list available templates
list_templates() {
    echo "Available project templates:"
    if [[ -d "$TEMPLATES_DIR" ]]; then
        find "$TEMPLATES_DIR" -type d -maxdepth 1 -mindepth 1 -exec basename {} \; | sort
    else
        echo "No templates directory found at $TEMPLATES_DIR"
        echo "Initialize with: $0 init"
    fi
}

# Function to initialize templates directory
initialize_templates() {
    echo "Initializing project templates..."
    
    mkdir -p "$TEMPLATES_DIR"
    
    # Create example templates
    create_python_template
    create_nodejs_template
    create_react_template
    create_docker_template
    
    echo "Templates initialized at $TEMPLATES_DIR"
}

# Function to create a new project
create_project() {
    local template_name="$1"
    local project_name="${2:-my-project}"
    local template_dir="$TEMPLATES_DIR/$template_name"
    
    if [[ ! -d "$template_dir" ]]; then
        echo "Template '$template_name' not found!"
        list_templates
        exit 1
    fi
    
    echo "Creating project '$project_name' from template '$template_name'..."
    
    # Check if generator script exists
    if [[ -f "$template_dir/generate.sh" ]]; then
        cd "$(dirname "$template_dir")"
        bash "$template_dir/generate.sh" "$project_name" "${@:3}"
    else
        # Simple copy if no generator script
        cp -r "$template_dir" "$project_name"
        
        # Replace template variables
        replace_template_variables "$project_name" "$project_name"
    fi
    
    echo "Project '$project_name' created successfully!"
}

# Function to replace template variables
replace_template_variables() {
    local project_dir="$1"
    local project_name="$2"
    
    # Replace {{PROJECT_NAME}} with actual project name
    find "$project_dir" -type f \( -name "*.md" -o -name "*.json" -o -name "*.yaml" -o -name "*.yml" -o -name "*.txt" \) -exec sed -i "s/{{PROJECT_NAME}}/$project_name/g" {} \;
    
    # Replace {{PROJECT_NAME_UNDERSCORE}} with underscored version
    local project_underscore=$(echo "$project_name" | tr '-' '_')
    find "$project_dir" -type f \( -name "*.md" -o -name "*.json" -o -name "*.yaml" -o -name "*.yml" -o -name "*.txt" \) -exec sed -i "s/{{PROJECT_NAME_UNDERSCORE}}/$project_underscore/g" {} \;
    
    # Replace {{CURRENT_YEAR}} with current year
    local current_year=$(date +%Y)
    find "$project_dir" -type f \( -name "*.md" -o -name "*.json" -o -name "*.yaml" -o -name "*.yml" -o -name "*.txt" \) -exec sed -i "s/{{CURRENT_YEAR}}/$current_year/g" {} \;
}

# Function to add a new template
add_template() {
    local template_name="$1"
    local source_dir="${2:-.}"
    local template_dir="$TEMPLATES_DIR/$template_name"
    
    if [[ -d "$template_dir" ]]; then
        read -p "Template '$template_name' already exists. Overwrite? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 0
        fi
        rm -rf "$template_dir"
    fi
    
    echo "Adding template '$template_name' from '$source_dir'..."
    
    cp -r "$source_dir" "$template_dir"
    
    # Remove common files that shouldn't be in templates
    rm -rf "$template_dir"/{node_modules,.git,dist,build,__pycache__,.pytest_cache}
    
    echo "Template '$template_name' added successfully!"
}

# Function to remove a template
remove_template() {
    local template_name="$1"
    local template_dir="$TEMPLATES_DIR/$template_name"
    
    if [[ ! -d "$template_dir" ]]; then
        echo "Template '$template_name' not found!"
        exit 1
    fi
    
    read -p "Are you sure you want to remove template '$template_name'? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$template_dir"
        echo "Template '$template_name' removed."
    fi
}

# Function to update templates from repository
update_templates() {
    local repo_url="${1:-https://github.com/your-org/project-templates.git}"
    local temp_dir=$(mktemp -d)
    
    echo "Updating templates from $repo_url..."
    
    git clone "$repo_url" "$temp_dir"
    
    # Copy templates
    if [[ -d "$temp_dir/templates" ]]; then
        cp -r "$temp_dir/templates/"* "$TEMPLATES_DIR/"
    else
        cp -r "$temp_dir/"* "$TEMPLATES_DIR/"
    fi
    
    rm -rf "$temp_dir"
    
    echo "Templates updated successfully!"
}

# Main command handling
case "${1:-}" in
    "list"|"ls")
        list_templates
        ;;
    "init")
        initialize_templates
        ;;
    "create"|"new")
        if [[ $# -lt 2 ]]; then
            echo "Usage: $0 create <template> <project_name> [additional_args...]"
            exit 1
        fi
        create_project "${@:2}"
        ;;
    "add")
        if [[ $# -lt 2 ]]; then
            echo "Usage: $0 add <template_name> [source_directory]"
            exit 1
        fi
        add_template "${@:2}"
        ;;
    "remove"|"rm")
        if [[ $# -lt 2 ]]; then
            echo "Usage: $0 remove <template_name>"
            exit 1
        fi
        remove_template "$2"
        ;;
    "update")
        update_templates "${2:-}"
        ;;
    *)
        echo "Project Generator - Universal project scaffolding tool"
        echo ""
        echo "Usage: $0 <command> [arguments...]"
        echo ""
        echo "Commands:"
        echo "  list, ls                     - List available templates"
        echo "  init                         - Initialize templates directory"
        echo "  create, new <template> <name> - Create new project from template"
        echo "  add <name> [source]          - Add new template"
        echo "  remove, rm <name>            - Remove template"
        echo "  update [repo_url]            - Update templates from repository"
        echo ""
        echo "Examples:"
        echo "  $0 list"
        echo "  $0 create python-api my-api"
        echo "  $0 create react-app my-frontend"
        echo "  $0 add my-template ./my-project"
        exit 1
        ;;
esac
```

---

*This comprehensive configuration templates guide provides automated project scaffolding, infrastructure provisioning, and development workflow templates for rapid project initialization and consistent development practices across different technologies and platforms.* 