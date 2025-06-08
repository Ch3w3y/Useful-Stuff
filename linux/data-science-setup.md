# Linux Data Science Setup Guide
=====================================

> **Complete guide for setting up a Linux workstation optimized for data science work**

## Table of Contents
- [System Requirements](#system-requirements)
- [Initial System Setup](#initial-system-setup)
- [Package Managers](#package-managers)
- [Development Environment](#development-environment)
- [R Environment Setup](#r-environment-setup)
- [Python Environment Setup](#python-environment-setup)
- [Database Systems](#database-systems)
- [Containerization](#containerization)
- [Version Control](#version-control)
- [Command Line Tools](#command-line-tools)
- [Text Editors and IDEs](#text-editors-and-ides)
- [System Monitoring](#system-monitoring)
- [Performance Optimization](#performance-optimization)
- [Backup and Security](#backup-and-security)
- [Useful Aliases and Functions](#useful-aliases-and-functions)
- [Troubleshooting](#troubleshooting)

---

## System Requirements

### Recommended Hardware
- **CPU**: Multi-core processor (8+ cores recommended)
- **RAM**: 16GB+ (32GB+ for large datasets)
- **Storage**: SSD with 500GB+ free space
- **GPU**: NVIDIA GPU for deep learning (optional)

### Linux Distributions
**Recommended for Data Science:**
- Ubuntu 22.04 LTS (most popular)
- Fedora 38+ (cutting-edge packages)
- Pop!_OS (NVIDIA-optimized)
- Manjaro (Arch-based, rolling release)
- openSUSE Tumbleweed

---

## Initial System Setup

### 1. System Updates
```bash
# Ubuntu/Debian
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl wget git vim build-essential

# Fedora
sudo dnf update -y
sudo dnf install -y curl wget git vim gcc gcc-c++ make

# Arch/Manjaro
sudo pacman -Syu
sudo pacman -S curl wget git vim base-devel
```

### 2. Enable Additional Repositories
```bash
# Ubuntu - Enable universe repository
sudo add-apt-repository universe

# Enable snap packages
sudo apt install snapd

# Fedora - Enable RPM Fusion
sudo dnf install -y \
  https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm \
  https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm

# Enable Flathub
flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo
```

### 3. Install Essential Development Tools
```bash
# Ubuntu/Debian
sudo apt install -y \
  build-essential \
  cmake \
  gfortran \
  libblas-dev \
  liblapack-dev \
  libxml2-dev \
  libcurl4-openssl-dev \
  libssl-dev \
  libfontconfig1-dev \
  libharfbuzz-dev \
  libfribidi-dev \
  libfreetype6-dev \
  libpng-dev \
  libtiff5-dev \
  libjpeg-dev

# Fedora
sudo dnf install -y \
  gcc gcc-c++ gcc-gfortran \
  cmake \
  blas-devel \
  lapack-devel \
  libxml2-devel \
  libcurl-devel \
  openssl-devel \
  fontconfig-devel \
  harfbuzz-devel \
  fribidi-devel \
  freetype-devel \
  libpng-devel \
  libtiff-devel \
  libjpeg-turbo-devel
```

---

## Package Managers

### 1. Homebrew for Linux
```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Add to PATH
echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"' >> ~/.bashrc
eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"

# Install useful tools
brew install htop neofetch tree fd ripgrep bat exa
```

### 2. Conda Package Manager
```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# Initialize conda
~/miniconda3/bin/conda init bash
source ~/.bashrc

# Configure conda
conda config --set auto_activate_base false
conda config --add channels conda-forge
conda config --set channel_priority strict
```

---

## Development Environment

### 1. Shell Configuration (Zsh + Oh My Zsh)
```bash
# Install Zsh
sudo apt install zsh  # Ubuntu
sudo dnf install zsh  # Fedora

# Install Oh My Zsh
sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# Install useful plugins
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# Edit ~/.zshrc
plugins=(git python conda docker zsh-autosuggestions zsh-syntax-highlighting)
```

### 2. Tmux Configuration
```bash
# Install tmux
sudo apt install tmux  # Ubuntu
sudo dnf install tmux  # Fedora

# Create tmux config
cat > ~/.tmux.conf << 'EOF'
# Prefix key
set -g prefix C-a
unbind C-b
bind C-a send-prefix

# Enable mouse mode
set -g mouse on

# Pane splitting
bind | split-window -h
bind - split-window -v

# Pane navigation
bind h select-pane -L
bind j select-pane -D
bind k select-pane -U
bind l select-pane -R

# Status bar
set -g status-bg colour235
set -g status-fg colour136
set -g status-left '#[fg=colour196]#H '
set -g status-right '#[fg=colour196]%Y-%m-%d %H:%M'

# Start windows and panes at 1
set -g base-index 1
setw -g pane-base-index 1
EOF
```

---

## R Environment Setup

### 1. Install R and RStudio
```bash
# Ubuntu - Add R repository
wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
echo "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/" | sudo tee -a /etc/apt/sources.list
sudo apt update
sudo apt install -y r-base r-base-dev

# Fedora
sudo dnf install -y R R-devel

# Install RStudio
wget https://download1.rstudio.org/electron/jammy/amd64/rstudio-2023.12.1-402-amd64.deb
sudo dpkg -i rstudio-*.deb
sudo apt-get install -f  # Fix dependencies if needed
```

### 2. R Package Development Environment
```bash
# Install system dependencies for common R packages
sudo apt install -y \
  libcairo2-dev \
  libxt-dev \
  libgdal-dev \
  libgeos-dev \
  libproj-dev \
  libudunits2-dev \
  libpoppler-cpp-dev \
  libmagick++-dev \
  librsvg2-dev \
  libwebp-dev \
  libv8-dev

# For spatial analysis
sudo apt install -y \
  libgeos-dev \
  libproj-dev \
  libgdal-dev \
  libudunits2-dev

# For text mining
sudo apt install -y \
  libpoppler-cpp-dev \
  tesseract-ocr \
  tesseract-ocr-eng \
  libtesseract-dev
```

### 3. Essential R Packages Installation Script
```bash
# Create R package installation script
cat > install_r_packages.R << 'EOF'
# Set CRAN mirror
options(repos = c(CRAN = "https://cloud.r-project.org/"))

# Core packages
install.packages(c(
  "tidyverse", "data.table", "dtplyr",
  "devtools", "remotes", "pak",
  "rmarkdown", "knitr", "bookdown",
  "shiny", "shinydashboard", "plotly",
  "DT", "htmlwidgets"
))

# Statistical packages
install.packages(c(
  "lme4", "nlme", "survival", "survminer",
  "forecast", "prophet", "tseries",
  "randomForest", "xgboost", "caret",
  "rstanarm", "brms", "MCMCglmm"
))

# Data visualization
install.packages(c(
  "ggplot2", "ggthemes", "ggrepel",
  "patchwork", "cowplot", "gganimate",
  "leaflet", "sf", "mapview"
))

# Bioconductor
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install()

cat("R packages installed successfully!\n")
EOF

# Run the installation
Rscript install_r_packages.R
```

---

## Python Environment Setup

### 1. Python Version Management (pyenv)
```bash
# Install pyenv
curl https://pyenv.run | bash

# Add to shell configuration
cat >> ~/.bashrc << 'EOF'
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
EOF

source ~/.bashrc

# Install Python versions
pyenv install 3.11.5
pyenv install 3.10.12
pyenv global 3.11.5
```

### 2. Poetry for Dependency Management
```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Add to PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Configure Poetry
poetry config virtualenvs.in-project true
```

### 3. Jupyter Lab Setup
```bash
# Create conda environment for Jupyter
conda create -n jupyter python=3.11 -y
conda activate jupyter

# Install JupyterLab and extensions
conda install -y jupyterlab nodejs
pip install jupyterlab-git jupyterlab-lsp python-lsp-server[all]

# Install kernels for different environments
python -m ipykernel install --user --name jupyter --display-name "Python (Jupyter)"

# R kernel
conda install -y r-irkernel
```

---

## Database Systems

### 1. PostgreSQL
```bash
# Install PostgreSQL
sudo apt install -y postgresql postgresql-contrib  # Ubuntu
sudo dnf install -y postgresql postgresql-server   # Fedora

# Initialize database (Fedora)
sudo postgresql-setup --initdb
sudo systemctl enable postgresql
sudo systemctl start postgresql

# Create user and database
sudo -u postgres createuser --interactive
sudo -u postgres createdb datascience

# Install client tools
pip install psycopg2-binary sqlalchemy
```

### 2. SQLite
```bash
# SQLite is usually pre-installed, but install tools
sudo apt install -y sqlite3 sqlitebrowser  # Ubuntu
sudo dnf install -y sqlite sqlite-devel    # Fedora

# Python packages
pip install sqlite3
```

### 3. DuckDB (Analytics Database)
```bash
# Install DuckDB
wget https://github.com/duckdb/duckdb/releases/latest/download/duckdb_cli-linux-amd64.zip
unzip duckdb_cli-linux-amd64.zip
sudo mv duckdb /usr/local/bin/

# Python package
pip install duckdb
```

---

## Containerization

### 1. Docker Setup
```bash
# Ubuntu
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
sudo usermod -aG docker $USER

# Fedora
sudo dnf install -y docker docker-compose
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker $USER

# Log out and back in for group changes
```

### 2. NVIDIA Docker (for GPU support)
```bash
# Install NVIDIA Docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 3. Podman (Docker alternative)
```bash
# Install Podman
sudo apt install -y podman  # Ubuntu
sudo dnf install -y podman  # Fedora

# Configure for rootless operation
echo 'export DOCKER_HOST=unix:///run/user/1000/podman/podman.sock' >> ~/.bashrc
```

---

## Command Line Tools

### 1. Modern Unix Tools
```bash
# Install modern alternatives to classic Unix tools
brew install \
  fd \          # find alternative
  ripgrep \     # grep alternative  
  bat \         # cat alternative
  exa \         # ls alternative
  dust \        # du alternative
  procs \       # ps alternative
  sd \          # sed alternative
  tokei \       # code statistics
  hyperfine \   # benchmarking
  bandwhich \   # network utilization
  bottom        # top alternative
```

### 2. Data Processing Tools
```bash
# Install data processing tools
brew install \
  jq \          # JSON processor
  yq \          # YAML processor
  xsv \         # CSV toolkit
  miller \      # Data processing
  choose \      # Human-friendly cut
  rclone \      # Cloud storage sync
  rsync \       # File synchronization
  parallel      # Parallel execution

# Python-based tools
pip install \
  csvkit \      # CSV utilities
  httpie \      # HTTP client
  glances \     # System monitoring
  ranger-fm     # File manager
```

### 3. Development Tools
```bash
# Version control and development
brew install \
  git-delta \   # Better git diffs
  gh \          # GitHub CLI
  hub \         # GitHub integration
  tig \         # Git TUI
  lazygit \     # Git TUI
  pre-commit    # Git hooks

# Code analysis
brew install \
  shellcheck \  # Shell script analysis
  yamllint \    # YAML linting
  hadolint      # Dockerfile linting
```

---

## Text Editors and IDEs

### 1. Neovim Setup
```bash
# Install Neovim
sudo apt install -y neovim  # Ubuntu
sudo dnf install -y neovim  # Fedora

# Install vim-plug
sh -c 'curl -fLo "${XDG_DATA_HOME:-$HOME/.local/share}"/nvim/site/autoload/plug.vim --create-dirs \
       https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim'

# Create basic config
mkdir -p ~/.config/nvim
cat > ~/.config/nvim/init.vim << 'EOF'
call plug#begin()
Plug 'nvim-treesitter/nvim-treesitter', {'do': ':TSUpdate'}
Plug 'neovim/nvim-lspconfig'
Plug 'hrsh7th/nvim-cmp'
Plug 'hrsh7th/cmp-nvim-lsp'
Plug 'jalvesaq/Nvim-R'
Plug 'jpalardy/vim-slime'
call plug#end()

" Basic settings
set number
set tabstop=4
set shiftwidth=4
set expandtab
set autoindent
EOF
```

### 2. VS Code
```bash
# Install VS Code
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/
echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/trusted.gpg.d/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" | sudo tee /etc/apt/sources.list.d/vscode.list
sudo apt update
sudo apt install -y code

# Essential extensions
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
code --install-extension ikuyadeu.r
code --install-extension ms-vscode.vscode-json
code --install-extension redhat.vscode-yaml
```

---

## System Monitoring

### 1. System Performance Tools
```bash
# Install monitoring tools
sudo apt install -y \
  htop \        # Process viewer
  iotop \       # I/O monitor
  nethogs \     # Network monitor
  ncdu \        # Disk usage
  tree \        # Directory tree
  lsof \        # Open files
  strace \      # System calls
  tcpdump       # Network capture

# Modern alternatives
pip install glances  # Comprehensive system monitor
brew install bottom  # Modern top alternative
```

### 2. Log Analysis
```bash
# Install log analysis tools
sudo apt install -y \
  logwatch \    # Log analyzer
  fail2ban \    # Intrusion prevention
  lynis         # Security audit

# Create log monitoring alias
echo 'alias logs="sudo journalctl -f"' >> ~/.bashrc
echo 'alias syslog="sudo tail -f /var/log/syslog"' >> ~/.bashrc
```

---

## Performance Optimization

### 1. System Tuning
```bash
# Create performance tuning script
cat > ~/tune_system.sh << 'EOF'
#!/bin/bash

# Increase file descriptor limits
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# Optimize swappiness for data science workloads
echo "vm.swappiness=10" | sudo tee -a /etc/sysctl.conf

# Increase shared memory for large datasets
echo "kernel.shmmax=134217728" | sudo tee -a /etc/sysctl.conf

# Apply changes
sudo sysctl -p

echo "System tuning complete. Reboot recommended."
EOF

chmod +x ~/tune_system.sh
```

### 2. R Performance Optimization
```bash
# Install optimized BLAS libraries
sudo apt install -y libopenblas-dev  # Ubuntu
sudo dnf install -y openblas-devel   # Fedora

# Configure R to use multiple cores
echo 'options(mc.cores = parallel::detectCores())' >> ~/.Rprofile
echo 'options(Ncpus = parallel::detectCores())' >> ~/.Rprofile

# Set up Rprofile for faster package loading
cat >> ~/.Rprofile << 'EOF'
# Fast package loading
if (interactive()) {
  options(repos = c(CRAN = "https://cloud.r-project.org/"))
  options(browserNLdisabled = TRUE)
  options(deparse.max.lines = 2)
}

# Useful startup message
cat("R environment loaded with", parallel::detectCores(), "cores available\n")
EOF
```

---

## Backup and Security

### 1. Automated Backups
```bash
# Create backup script
cat > ~/backup_data.sh << 'EOF'
#!/bin/bash

BACKUP_DIR="/backup/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Backup home directory (excluding large files)
rsync -av --exclude='*.tmp' --exclude='*.log' \
  --exclude='.cache' --exclude='Downloads' \
  ~/ "$BACKUP_DIR/home/"

# Backup conda environments
conda env export > "$BACKUP_DIR/conda_environment.yml"

# Backup installed packages
pip list --format=freeze > "$BACKUP_DIR/pip_requirements.txt"
dpkg --get-selections > "$BACKUP_DIR/apt_packages.txt"

echo "Backup completed: $BACKUP_DIR"
EOF

chmod +x ~/backup_data.sh

# Set up cron job for weekly backups
(crontab -l 2>/dev/null; echo "0 2 * * 0 ~/backup_data.sh") | crontab -
```

### 2. Security Hardening
```bash
# Install security tools
sudo apt install -y ufw fail2ban

# Configure firewall
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh

# Configure fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

---

## Useful Aliases and Functions

### 1. Data Science Aliases
```bash
# Add to ~/.bashrc or ~/.zshrc
cat >> ~/.bashrc << 'EOF'
# Data Science Aliases
alias py="python"
alias ipy="ipython"
alias jlab="jupyter lab"
alias jnb="jupyter notebook"
alias r="R --no-save"
alias rstudio="rstudio > /dev/null 2>&1 &"

# System monitoring
alias cpu="htop"
alias mem="free -h"
alias disk="df -h"
alias ports="netstat -tuln"

# File operations
alias ll="exa -la"
alias la="exa -la"
alias tree="exa --tree"
alias cat="bat"
alias find="fd"
alias grep="rg"

# Git shortcuts
alias gs="git status"
alias ga="git add"
alias gc="git commit"
alias gp="git push"
alias gl="git log --oneline"

# Quick directory access
alias proj="cd ~/projects"
alias data="cd ~/data"
alias docs="cd ~/Documents"

# Conda shortcuts
alias ca="conda activate"
alias cda="conda deactivate"
alias cl="conda list"
alias ce="conda env list"
EOF
```

### 2. Useful Functions
```bash
# Add functions to ~/.bashrc
cat >> ~/.bashrc << 'EOF'
# Create and activate conda environment
conda_create() {
    conda create -n "$1" python=3.11 -y
    conda activate "$1"
}

# Quick project setup
setup_project() {
    mkdir -p "$1"/{data,notebooks,scripts,output}
    cd "$1"
    git init
    echo "# $1" > README.md
    echo "__pycache__/\n*.pyc\n.ipynb_checkpoints/\ndata/\noutput/" > .gitignore
}

# System info function
sysinfo() {
    echo "System Information:"
    echo "=================="
    echo "OS: $(lsb_release -d | cut -f2)"
    echo "Kernel: $(uname -r)"
    echo "CPU: $(lscpu | grep 'Model name' | cut -f 2 -d ':' | awk '{$1=$1}1')"
    echo "Memory: $(free -h | awk '/^Mem:/ {print $2}')"
    echo "Disk: $(df -h / | awk 'NR==2 {print $2}')"
    echo "Python: $(python --version)"
    echo "R: $(R --version | head -n1)"
}

# Data file preview
preview() {
    if [[ "$1" == *.csv ]]; then
        head -n 10 "$1" | column -t -s ','
    elif [[ "$1" == *.json ]]; then
        head -n 20 "$1" | jq '.'
    elif [[ "$1" == *.yaml ]] || [[ "$1" == *.yml ]]; then
        head -n 20 "$1" | yq '.'
    else
        head -n 20 "$1"
    fi
}
EOF

source ~/.bashrc
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. R Package Installation Issues
```bash
# Install system dependencies
sudo apt build-dep r-base-core

# For XML packages
sudo apt install -y libxml2-dev

# For curl packages  
sudo apt install -y libcurl4-openssl-dev

# For graphics packages
sudo apt install -y libcairo2-dev libxt-dev
```

#### 2. Python Environment Issues
```bash
# Reset conda environment
conda clean --all
conda update conda

# Fix pip issues
python -m pip install --upgrade pip
python -m pip install --force-reinstall pip
```

#### 3. Permission Issues
```bash
# Fix ownership of home directory
sudo chown -R $USER:$USER ~

# Fix conda permissions
sudo chown -R $USER:$USER ~/miniconda3
```

#### 4. Performance Issues
```bash
# Check system resources
htop
iotop -o
df -h

# Clear caches
sudo apt clean
conda clean --all
pip cache purge
```

### Useful Diagnostic Commands
```bash
# System diagnostics
lscpu          # CPU information
free -h        # Memory usage
df -h          # Disk usage
lsblk          # Block devices
lsusb          # USB devices
lspci          # PCI devices
dmesg          # Kernel messages
journalctl -f  # System logs

# Network diagnostics
ip addr        # Network interfaces
ss -tuln       # Open ports
ping -c 4 google.com  # Connectivity test
```

---

## Additional Resources

### Documentation and Learning
- [Linux Command Line Basics](https://ubuntu.com/tutorials/command-line-for-beginners)
- [R Installation Guide](https://cran.r-project.org/bin/linux/)
- [Conda User Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/)
- [Docker Documentation](https://docs.docker.com/)

### Community Resources
- **Stack Overflow**: Programming questions
- **Reddit**: r/linux, r/datascience, r/rstats
- **Discord**: Linux and data science communities
- **GitHub**: Open source projects and dotfiles

---

**Happy Linux data science computing!** 🐧📊

> This guide is continuously updated. Check the repository for the latest version and contribute improvements! 