# Development Environment Configurations

## Overview

Comprehensive guide to setting up and managing development environments across different programming languages, platforms, and project types. Includes virtual environments, containerized development, and automated environment provisioning.

## Table of Contents

- [Python Environments](#python-environments)
- [Node.js Environments](#nodejs-environments)
- [Containerized Development](#containerized-development)
- [Multi-Language Environments](#multi-language-environments)
- [Cloud Development Environments](#cloud-development-environments)
- [Environment Automation](#environment-automation)
- [Security & Best Practices](#security--best-practices)
- [Troubleshooting](#troubleshooting)

## Python Environments

### Virtual Environment Management

```bash
#!/bin/bash
# scripts/python-env-setup.sh - Python environment management

set -euo pipefail

# Function to create Python virtual environment
create_python_env() {
    local env_name="$1"
    local python_version="${2:-3.9}"
    local requirements_file="${3:-requirements.txt}"
    
    echo "Creating Python $python_version environment: $env_name"
    
    # Create virtual environment
    python$python_version -m venv "$env_name"
    source "$env_name/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install requirements if file exists
    if [[ -f "$requirements_file" ]]; then
        pip install -r "$requirements_file"
    fi
    
    # Install development tools
    pip install black flake8 mypy pytest pytest-cov ipython jupyter
    
    echo "Environment $env_name created successfully!"
    echo "Activate with: source $env_name/bin/activate"
}

# Function to create conda environment
create_conda_env() {
    local env_name="$1"
    local python_version="${2:-3.9}"
    local environment_file="${3:-environment.yml}"
    
    echo "Creating Conda environment: $env_name"
    
    if [[ -f "$environment_file" ]]; then
        conda env create -n "$env_name" -f "$environment_file"
    else
        conda create -n "$env_name" python="$python_version" -y
        conda activate "$env_name"
        
        # Install common packages
        conda install numpy pandas matplotlib seaborn scikit-learn jupyter -y
        conda install -c conda-forge black flake8 mypy pytest -y
    fi
    
    echo "Conda environment $env_name created successfully!"
    echo "Activate with: conda activate $env_name"
}

# Function to create pipenv environment
create_pipenv_env() {
    local project_dir="$1"
    local python_version="${2:-3.9}"
    
    echo "Creating Pipenv environment in: $project_dir"
    
    cd "$project_dir"
    
    # Initialize Pipenv
    pipenv --python "$python_version"
    
    # Install development dependencies
    pipenv install --dev black flake8 mypy pytest pytest-cov
    
    # Install common packages
    pipenv install requests pandas numpy matplotlib seaborn
    
    echo "Pipenv environment created successfully!"
    echo "Activate with: pipenv shell"
}

# Usage examples
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    case "${1:-}" in
        "venv")
            create_python_env "${2:-myenv}" "${3:-3.9}" "${4:-requirements.txt}"
            ;;
        "conda")
            create_conda_env "${2:-myenv}" "${3:-3.9}" "${4:-environment.yml}"
            ;;
        "pipenv")
            create_pipenv_env "${2:-.}" "${3:-3.9}"
            ;;
        *)
            echo "Usage: $0 {venv|conda|pipenv} [env_name] [python_version] [config_file]"
            exit 1
            ;;
    esac
fi
```

### Python Environment Configuration Files

```yaml
# environment.yml - Conda environment specification
name: data-science-env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - numpy=1.21.0
  - pandas=1.3.0
  - matplotlib=3.4.2
  - seaborn=0.11.1
  - scikit-learn=0.24.2
  - jupyter=1.0.0
  - ipython=7.25.0
  - requests=2.25.1
  - pip=21.1.3
  - pip:
    - black==21.6b0
    - flake8==3.9.2
    - mypy==0.910
    - pytest==6.2.4
    - pytest-cov==2.12.1
    - streamlit==0.84.0
    - fastapi==0.68.0
    - uvicorn==0.14.0
variables:
  - PYTHONPATH: $CONDA_PREFIX/lib/python3.9/site-packages
  - JUPYTER_CONFIG_DIR: $CONDA_PREFIX/etc/jupyter
```

```ini
# Pipfile - Pipenv configuration
[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"

[packages]
numpy = "~=1.21.0"
pandas = "~=1.3.0"
matplotlib = "~=3.4.0"
seaborn = "~=0.11.0"
scikit-learn = "~=0.24.0"
requests = "~=2.25.0"
jupyter = "*"
ipython = "*"
streamlit = "*"
fastapi = "*"
uvicorn = "*"

[dev-packages]
black = "*"
flake8 = "*"
mypy = "*"
pytest = "*"
pytest-cov = "*"
ipdb = "*"
pre-commit = "*"

[requires]
python_version = "3.9"

[scripts]
test = "pytest tests/"
format = "black ."
lint = "flake8 ."
typecheck = "mypy ."
serve = "uvicorn main:app --reload"
```

## Node.js Environments

### Node.js Environment Setup

```bash
#!/bin/bash
# scripts/nodejs-env-setup.sh - Node.js environment management

set -euo pipefail

# Function to install and configure Node.js with NVM
setup_nodejs_nvm() {
    local node_version="${1:-16.14.0}"
    
    echo "Setting up Node.js $node_version with NVM"
    
    # Install NVM if not already installed
    if ! command -v nvm &> /dev/null; then
        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
        export NVM_DIR="$HOME/.nvm"
        [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
    fi
    
    # Install and use specified Node.js version
    nvm install "$node_version"
    nvm use "$node_version"
    nvm alias default "$node_version"
    
    # Update npm to latest version
    npm install -g npm@latest
    
    # Install global development tools
    npm install -g \
        typescript \
        ts-node \
        eslint \
        prettier \
        nodemon \
        pm2 \
        create-react-app \
        @angular/cli \
        @vue/cli \
        next \
        express-generator
    
    echo "Node.js $node_version environment setup complete!"
}

# Function to create project with specific configuration
create_nodejs_project() {
    local project_name="$1"
    local project_type="${2:-express}"
    
    echo "Creating Node.js project: $project_name (type: $project_type)"
    
    mkdir -p "$project_name"
    cd "$project_name"
    
    # Initialize package.json
    npm init -y
    
    case "$project_type" in
        "express")
            npm install express
            npm install --save-dev nodemon @types/node @types/express typescript ts-node
            create_express_structure
            ;;
        "react")
            npx create-react-app . --template typescript
            ;;
        "vue")
            npx @vue/cli create . --default
            ;;
        "next")
            npx create-next-app . --typescript
            ;;
        *)
            echo "Unknown project type: $project_type"
            exit 1
            ;;
    esac
    
    # Add common development dependencies
    npm install --save-dev eslint prettier husky lint-staged
    
    # Setup development scripts
    setup_nodejs_scripts
    
    echo "Project $project_name created successfully!"
}

# Function to create Express.js project structure
create_express_structure() {
    # Create directory structure
    mkdir -p src/{controllers,middleware,models,routes,services,utils}
    mkdir -p tests
    mkdir -p docs
    
    # Create main application file
    cat > src/app.ts << 'EOF'
import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(helmet());
app.use(cors());
app.use(morgan('combined'));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Routes
app.get('/health', (req, res) => {
  res.status(200).json({ status: 'OK', timestamp: new Date().toISOString() });
});

// Error handling middleware
app.use((err: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
  console.error(err.stack);
  res.status(500).json({ error: 'Internal Server Error' });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({ error: 'Not Found' });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

export default app;
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
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "tests"]
}
EOF
}

# Function to setup development scripts
setup_nodejs_scripts() {
    # Update package.json scripts
    node -e "
    const pkg = require('./package.json');
    pkg.scripts = {
      ...pkg.scripts,
      'start': 'node dist/app.js',
      'dev': 'nodemon src/app.ts',
      'build': 'tsc',
      'test': 'jest',
      'test:watch': 'jest --watch',
      'lint': 'eslint src/**/*.ts',
      'lint:fix': 'eslint src/**/*.ts --fix',
      'format': 'prettier --write src/**/*.ts',
      'prepare': 'husky install'
    };
    require('fs').writeFileSync('./package.json', JSON.stringify(pkg, null, 2));
    "
}

# Usage
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    case "${1:-}" in
        "setup")
            setup_nodejs_nvm "${2:-16.14.0}"
            ;;
        "create")
            create_nodejs_project "${2:-myapp}" "${3:-express}"
            ;;
        *)
            echo "Usage: $0 {setup|create} [version|name] [type]"
            exit 1
            ;;
    esac
fi
```

### Package Configuration Files

```json
{
  "name": "development-environment",
  "version": "1.0.0",
  "description": "Development environment configuration",
  "main": "dist/app.js",
  "scripts": {
    "start": "node dist/app.js",
    "dev": "nodemon src/app.ts",
    "build": "tsc",
    "test": "jest",
    "test:watch": "jest --watch",
    "test:coverage": "jest --coverage",
    "lint": "eslint src/**/*.ts",
    "lint:fix": "eslint src/**/*.ts --fix",
    "format": "prettier --write src/**/*.ts",
    "format:check": "prettier --check src/**/*.ts",
    "prepare": "husky install",
    "clean": "rimraf dist"
  },
  "dependencies": {
    "express": "^4.18.0",
    "cors": "^2.8.5",
    "helmet": "^5.1.0",
    "morgan": "^1.10.0",
    "dotenv": "^16.0.1"
  },
  "devDependencies": {
    "@types/node": "^17.0.35",
    "@types/express": "^4.17.13",
    "@types/cors": "^2.8.12",
    "@types/morgan": "^1.9.3",
    "@types/jest": "^28.1.1",
    "typescript": "^4.7.2",
    "ts-node": "^10.8.1",
    "nodemon": "^2.0.16",
    "eslint": "^8.17.0",
    "@typescript-eslint/parser": "^5.27.1",
    "@typescript-eslint/eslint-plugin": "^5.27.1",
    "prettier": "^2.6.2",
    "jest": "^28.1.1",
    "ts-jest": "^28.0.4",
    "husky": "^8.0.1",
    "lint-staged": "^13.0.0",
    "rimraf": "^3.0.2"
  },
  "lint-staged": {
    "*.{ts,js}": [
      "eslint --fix",
      "prettier --write"
    ]
  },
  "husky": {
    "hooks": {
      "pre-commit": "lint-staged",
      "pre-push": "npm test"
    }
  }
}
```

## Containerized Development

### Docker Development Environment

```dockerfile
# Dockerfile.dev - Multi-stage development environment
FROM node:16-alpine AS base

# Install system dependencies
RUN apk add --no-cache \
    git \
    python3 \
    make \
    g++ \
    curl \
    bash

# Set working directory
WORKDIR /app

# Development stage
FROM base AS development

# Install global development tools
RUN npm install -g \
    nodemon \
    typescript \
    ts-node \
    eslint \
    prettier

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Expose development port
EXPOSE 3000

# Development command
CMD ["npm", "run", "dev"]

# Production build stage
FROM base AS build

# Copy package files
COPY package*.json ./

# Install production dependencies
RUN npm ci --only=production

# Copy source code
COPY . .

# Build application
RUN npm run build

# Production stage
FROM node:16-alpine AS production

WORKDIR /app

# Copy production dependencies and built application
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/dist ./dist
COPY --from=build /app/package*.json ./

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nextjs -u 1001

USER nextjs

EXPOSE 3000

CMD ["npm", "start"]
```

### Docker Compose Development Stack

```yaml
# docker-compose.dev.yml - Development stack
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.dev
      target: development
    container_name: dev-app
    restart: unless-stopped
    environment:
      NODE_ENV: development
      DATABASE_URL: postgresql://postgres:password@postgres:5432/devdb
      REDIS_URL: redis://redis:6379
    volumes:
      - .:/app
      - /app/node_modules
    ports:
      - "3000:3000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - dev-network

  postgres:
    image: postgres:14-alpine
    container_name: dev-postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
      POSTGRES_DB: devdb
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init.sql:ro
    ports:
      - "5432:5432"
    networks:
      - dev-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres -d devdb"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    container_name: dev-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - dev-network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Development tools
  adminer:
    image: adminer
    container_name: dev-adminer
    restart: unless-stopped
    ports:
      - "8080:8080"
    networks:
      - dev-network
    depends_on:
      - postgres

  mailhog:
    image: mailhog/mailhog
    container_name: dev-mailhog
    restart: unless-stopped
    ports:
      - "1025:1025"  # SMTP
      - "8025:8025"  # Web UI
    networks:
      - dev-network

volumes:
  postgres_data:
  redis_data:

networks:
  dev-network:
    driver: bridge
```

### VS Code Development Container

```json
{
  "name": "Development Environment",
  "dockerComposeFile": "docker-compose.dev.yml",
  "service": "app",
  "workspaceFolder": "/app",
  "shutdownAction": "stopCompose",
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-vscode.vscode-typescript-next",
        "esbenp.prettier-vscode",
        "ms-vscode.vscode-eslint",
        "bradlc.vscode-tailwindcss",
        "ms-vscode.vscode-json",
        "redhat.vscode-yaml",
        "ms-python.python",
        "ms-python.black-formatter",
        "ms-python.flake8",
        "ms-python.mypy-type-checker"
      ],
      "settings": {
        "terminal.integrated.shell.linux": "/bin/bash",
        "python.defaultInterpreterPath": "/usr/local/bin/python",
        "python.formatting.provider": "black",
        "python.linting.enabled": true,
        "python.linting.flake8Enabled": true,
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
          "source.fixAll.eslint": true
        }
      }
    }
  },
  "forwardPorts": [3000, 5432, 6379, 8080, 8025],
  "portsAttributes": {
    "3000": {
      "label": "Application",
      "onAutoForward": "notify"
    },
    "8080": {
      "label": "Adminer"
    },
    "8025": {
      "label": "MailHog"
    }
  },
  "postCreateCommand": "npm install && npm run build"
}
```

## Multi-Language Environments

### Polyglot Development Setup

```bash
#!/bin/bash
# scripts/polyglot-env-setup.sh - Multi-language environment setup

set -euo pipefail

# Function to setup complete development environment
setup_polyglot_environment() {
    echo "Setting up polyglot development environment..."
    
    # Update system packages
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get upgrade -y
        sudo apt-get install -y curl wget git build-essential
    elif command -v brew &> /dev/null; then
        brew update && brew upgrade
    fi
    
    # Setup Python
    setup_python_environment
    
    # Setup Node.js
    setup_nodejs_environment
    
    # Setup Go
    setup_go_environment
    
    # Setup Rust
    setup_rust_environment
    
    # Setup Java
    setup_java_environment
    
    # Setup development tools
    setup_development_tools
    
    echo "Polyglot environment setup complete!"
}

setup_python_environment() {
    echo "Setting up Python environment..."
    
    # Install pyenv for Python version management
    if ! command -v pyenv &> /dev/null; then
        curl https://pyenv.run | bash
        export PATH="$HOME/.pyenv/bin:$PATH"
        eval "$(pyenv init -)"
        eval "$(pyenv virtualenv-init -)"
    fi
    
    # Install latest Python versions
    pyenv install 3.9.13
    pyenv install 3.10.5
    pyenv global 3.10.5
    
    # Install global Python tools
    pip install --user pipenv poetry black flake8 mypy
}

setup_nodejs_environment() {
    echo "Setting up Node.js environment..."
    
    # Install NVM
    if ! command -v nvm &> /dev/null; then
        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
        export NVM_DIR="$HOME/.nvm"
        [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
    fi
    
    # Install Node.js versions
    nvm install 16
    nvm install 18
    nvm use 18
    nvm alias default 18
    
    # Install global packages
    npm install -g typescript ts-node eslint prettier yarn pnpm
}

setup_go_environment() {
    echo "Setting up Go environment..."
    
    # Install Go
    if ! command -v go &> /dev/null; then
        GO_VERSION="1.19.1"
        wget "https://golang.org/dl/go${GO_VERSION}.linux-amd64.tar.gz"
        sudo tar -C /usr/local -xzf "go${GO_VERSION}.linux-amd64.tar.gz"
        rm "go${GO_VERSION}.linux-amd64.tar.gz"
        
        echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
        echo 'export GOPATH=$HOME/go' >> ~/.bashrc
        echo 'export PATH=$PATH:$GOPATH/bin' >> ~/.bashrc
        
        export PATH=$PATH:/usr/local/go/bin
        export GOPATH=$HOME/go
        export PATH=$PATH:$GOPATH/bin
    fi
    
    # Install Go tools
    go install golang.org/x/tools/gopls@latest
    go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
}

setup_rust_environment() {
    echo "Setting up Rust environment..."
    
    # Install Rust
    if ! command -v rustc &> /dev/null; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
    fi
    
    # Install Rust components
    rustup component add rustfmt clippy rust-analyzer
}

setup_java_environment() {
    echo "Setting up Java environment..."
    
    # Install SDKMAN for Java version management
    if ! command -v sdk &> /dev/null; then
        curl -s "https://get.sdkman.io" | bash
        source "$HOME/.sdkman/bin/sdkman-init.sh"
    fi
    
    # Install Java versions
    sdk install java 11.0.16-tem
    sdk install java 17.0.4-tem
    sdk default java 17.0.4-tem
    
    # Install build tools
    sdk install maven
    sdk install gradle
}

setup_development_tools() {
    echo "Setting up development tools..."
    
    # Install Docker
    if ! command -v docker &> /dev/null; then
        curl -fsSL https://get.docker.com -o get-docker.sh
        sh get-docker.sh
        sudo usermod -aG docker $USER
        rm get-docker.sh
    fi
    
    # Install kubectl
    if ! command -v kubectl &> /dev/null; then
        curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
        sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
        rm kubectl
    fi
    
    # Install terraform
    if ! command -v terraform &> /dev/null; then
        wget https://releases.hashicorp.com/terraform/1.2.6/terraform_1.2.6_linux_amd64.zip
        unzip terraform_1.2.6_linux_amd64.zip
        sudo mv terraform /usr/local/bin/
        rm terraform_1.2.6_linux_amd64.zip
    fi
}

# Execute setup if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    setup_polyglot_environment
fi
```

## Environment Automation

### Environment Provisioning with Ansible

```yaml
# ansible/dev-environment.yml - Ansible playbook for environment setup
---
- name: Setup Development Environment
  hosts: localhost
  connection: local
  become: yes
  
  vars:
    user_home: "{{ ansible_env.HOME }}"
    python_versions:
      - "3.9.13"
      - "3.10.5"
    node_versions:
      - "16.16.0"
      - "18.7.0"
    go_version: "1.19.1"
    
  tasks:
    - name: Update package cache
      apt:
        update_cache: yes
        cache_valid_time: 3600
      when: ansible_os_family == "Debian"
      
    - name: Install system dependencies
      package:
        name:
          - curl
          - wget
          - git
          - build-essential
          - zsh
          - tmux
          - neovim
          - htop
          - tree
          - jq
        state: present
        
    - name: Install Python build dependencies
      package:
        name:
          - make
          - build-essential
          - libssl-dev
          - zlib1g-dev
          - libbz2-dev
          - libreadline-dev
          - libsqlite3-dev
          - llvm
          - libncurses5-dev
          - libncursesw5-dev
          - xz-utils
          - tk-dev
          - libffi-dev
          - liblzma-dev
          - python3-openssl
        state: present
        
    - name: Install pyenv
      git:
        repo: https://github.com/pyenv/pyenv.git
        dest: "{{ user_home }}/.pyenv"
        
    - name: Install pyenv-virtualenv
      git:
        repo: https://github.com/pyenv/pyenv-virtualenv.git
        dest: "{{ user_home }}/.pyenv/plugins/pyenv-virtualenv"
        
    - name: Install Python versions
      shell: |
        export PATH="{{ user_home }}/.pyenv/bin:$PATH"
        eval "$(pyenv init -)"
        pyenv install {{ item }}
      loop: "{{ python_versions }}"
      args:
        creates: "{{ user_home }}/.pyenv/versions/{{ item }}"
        
    - name: Set global Python version
      shell: |
        export PATH="{{ user_home }}/.pyenv/bin:$PATH"
        eval "$(pyenv init -)"
        pyenv global {{ python_versions[-1] }}
        
    - name: Install NVM
      shell: |
        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
      args:
        creates: "{{ user_home }}/.nvm"
        
    - name: Install Node.js versions
      shell: |
        export NVM_DIR="{{ user_home }}/.nvm"
        [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
        nvm install {{ item }}
      loop: "{{ node_versions }}"
      
    - name: Install Go
      unarchive:
        src: "https://golang.org/dl/go{{ go_version }}.linux-amd64.tar.gz"
        dest: /usr/local
        remote_src: yes
        creates: /usr/local/go/bin/go
        
    - name: Install Rust
      shell: |
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
      args:
        creates: "{{ user_home }}/.cargo"
        
    - name: Install Docker
      shell: |
        curl -fsSL https://get.docker.com -o get-docker.sh
        sh get-docker.sh
        rm get-docker.sh
      args:
        creates: /usr/bin/docker
        
    - name: Add user to docker group
      user:
        name: "{{ ansible_user }}"
        groups: docker
        append: yes
        
    - name: Install kubectl
      get_url:
        url: "https://dl.k8s.io/release/v1.24.3/bin/linux/amd64/kubectl"
        dest: /usr/local/bin/kubectl
        mode: '0755'
        
    - name: Install development tools via snap
      snap:
        name:
          - code
          - postman
        classic: yes
```

### Environment Configuration Management

```bash
#!/bin/bash
# scripts/manage-environments.sh - Environment management utility

set -euo pipefail

ENVIRONMENTS_DIR="$HOME/.environments"
TEMPLATES_DIR="$HOME/.environment-templates"

# Function to create environment from template
create_environment() {
    local env_name="$1"
    local template_name="$2"
    local env_dir="$ENVIRONMENTS_DIR/$env_name"
    
    echo "Creating environment '$env_name' from template '$template_name'"
    
    # Check if template exists
    if [[ ! -d "$TEMPLATES_DIR/$template_name" ]]; then
        echo "Template '$template_name' not found!"
        list_templates
        exit 1
    fi
    
    # Create environment directory
    mkdir -p "$env_dir"
    
    # Copy template files
    cp -r "$TEMPLATES_DIR/$template_name"/* "$env_dir/"
    
    # Make scripts executable
    find "$env_dir" -name "*.sh" -exec chmod +x {} \;
    
    # Run setup script if it exists
    if [[ -f "$env_dir/setup.sh" ]]; then
        cd "$env_dir"
        ./setup.sh
    fi
    
    echo "Environment '$env_name' created successfully!"
    echo "Activate with: source $env_dir/activate.sh"
}

# Function to activate environment
activate_environment() {
    local env_name="$1"
    local env_dir="$ENVIRONMENTS_DIR/$env_name"
    
    if [[ ! -d "$env_dir" ]]; then
        echo "Environment '$env_name' not found!"
        list_environments
        exit 1
    fi
    
    if [[ -f "$env_dir/activate.sh" ]]; then
        source "$env_dir/activate.sh"
        echo "Environment '$env_name' activated"
    else
        echo "No activation script found for environment '$env_name'"
    fi
}

# Function to list available environments
list_environments() {
    echo "Available environments:"
    if [[ -d "$ENVIRONMENTS_DIR" ]]; then
        ls -1 "$ENVIRONMENTS_DIR"
    else
        echo "No environments found"
    fi
}

# Function to list available templates
list_templates() {
    echo "Available templates:"
    if [[ -d "$TEMPLATES_DIR" ]]; then
        ls -1 "$TEMPLATES_DIR"
    else
        echo "No templates found"
    fi
}

# Function to remove environment
remove_environment() {
    local env_name="$1"
    local env_dir="$ENVIRONMENTS_DIR/$env_name"
    
    if [[ ! -d "$env_dir" ]]; then
        echo "Environment '$env_name' not found!"
        exit 1
    fi
    
    read -p "Are you sure you want to remove environment '$env_name'? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$env_dir"
        echo "Environment '$env_name' removed"
    fi
}

# Main command handling
case "${1:-}" in
    "create")
        create_environment "${2:-}" "${3:-default}"
        ;;
    "activate")
        activate_environment "${2:-}"
        ;;
    "list")
        list_environments
        ;;
    "templates")
        list_templates
        ;;
    "remove")
        remove_environment "${2:-}"
        ;;
    *)
        echo "Usage: $0 {create|activate|list|templates|remove} [environment_name] [template_name]"
        echo ""
        echo "Commands:"
        echo "  create <name> [template]  - Create new environment from template"
        echo "  activate <name>          - Activate environment"
        echo "  list                     - List available environments"
        echo "  templates                - List available templates"
        echo "  remove <name>            - Remove environment"
        exit 1
        ;;
esac
```

---

*This comprehensive environment configuration guide provides automated setup, management, and maintenance of development environments across multiple languages and platforms with containerization and cloud integration capabilities.* 