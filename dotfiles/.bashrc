# ~/.bashrc: executed by bash(1) for non-login shells
# ==================================================
# 
# Comprehensive Bash configuration for data science workflows
# Optimized for Linux environments with R, Python, and general development
# 
# Author: Data Science Team
# Date: 2024
# 
# Installation:
#   cp dotfiles/.bashrc ~/.bashrc
#   source ~/.bashrc

# =================================================
# 1. BASIC SHELL CONFIGURATION
# =================================================

# If not running interactively, don't do anything
case $- in
    *i*) ;;
      *) return;;
esac

# History configuration
HISTCONTROL=ignoreboth:erasedups
HISTSIZE=10000
HISTFILESIZE=20000
HISTTIMEFORMAT='%F %T '
shopt -s histappend
shopt -s cmdhist

# Check window size after each command
shopt -s checkwinsize

# Enable extended globbing
shopt -s extglob

# Case-insensitive globbing (used in pathname expansion)
shopt -s nocaseglob

# Correct minor typos in cd
shopt -s cdspell

# Enable programmable completion features
if ! shopt -oq posix; then
  if [ -f /usr/share/bash-completion/bash_completion ]; then
    . /usr/share/bash-completion/bash_completion
  elif [ -f /etc/bash_completion ]; then
    . /etc/bash_completion
  fi
fi

# =================================================
# 2. ENVIRONMENT VARIABLES
# =================================================

# Default editor
export EDITOR='vim'
export VISUAL='vim'

# Locale settings
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# Less configuration
export LESS='-R -S -M -i -j5'
export LESSHISTFILE=/dev/null

# Man page colors
export LESS_TERMCAP_mb=$'\E[1;31m'     # begin bold
export LESS_TERMCAP_md=$'\E[1;36m'     # begin blink
export LESS_TERMCAP_me=$'\E[0m'        # reset bold/blink
export LESS_TERMCAP_so=$'\E[01;44;33m' # begin reverse video
export LESS_TERMCAP_se=$'\E[0m'        # reset reverse video
export LESS_TERMCAP_us=$'\E[1;32m'     # begin underline
export LESS_TERMCAP_ue=$'\E[0m'        # reset underline

# Development directories
export PROJECTS_DIR="$HOME/projects"
export DATA_DIR="$HOME/data"
export SCRIPTS_DIR="$HOME/scripts"

# Create directories if they don't exist
mkdir -p "$PROJECTS_DIR" "$DATA_DIR" "$SCRIPTS_DIR"

# =================================================
# 3. PATH CONFIGURATION
# =================================================

# Function to safely add to PATH
add_to_path() {
    if [[ -d "$1" && ":$PATH:" != *":$1:"* ]]; then
        export PATH="$1:$PATH"
    fi
}

# Local binaries
add_to_path "$HOME/.local/bin"
add_to_path "$HOME/bin"

# Homebrew (if installed)
if [[ -d "/home/linuxbrew/.linuxbrew" ]]; then
    eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"
fi

# Cargo (Rust)
add_to_path "$HOME/.cargo/bin"

# Go
if [[ -d "/usr/local/go/bin" ]]; then
    add_to_path "/usr/local/go/bin"
    export GOPATH="$HOME/go"
    add_to_path "$GOPATH/bin"
fi

# =================================================
# 4. PYTHON CONFIGURATION
# =================================================

# PyEnv configuration
if [[ -d "$HOME/.pyenv" ]]; then
    export PYENV_ROOT="$HOME/.pyenv"
    add_to_path "$PYENV_ROOT/bin"
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
fi

# Poetry configuration
if command -v poetry &> /dev/null; then
    export POETRY_VENV_IN_PROJECT=1
fi

# Pip configuration
export PIP_REQUIRE_VIRTUALENV=true

# Function to temporarily disable pip virtualenv requirement
gpip() {
    PIP_REQUIRE_VIRTUALENV="" pip "$@"
}

# Python development aliases
alias py='python'
alias py3='python3'
alias ipy='ipython'
alias jlab='jupyter lab'
alias jnb='jupyter notebook'

# Virtual environment shortcuts
alias ve='python -m venv'
alias va='source ./venv/bin/activate'
alias vd='deactivate'

# =================================================
# 5. R CONFIGURATION
# =================================================

# R aliases
alias r='R --no-save'
alias R='R --no-save'
alias rstudio='rstudio > /dev/null 2>&1 &'

# R library path (if custom location)
if [[ -d "$HOME/R/library" ]]; then
    export R_LIBS_USER="$HOME/R/library"
fi

# =================================================
# 6. CONDA CONFIGURATION
# =================================================

# Conda initialization (if installed)
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    . "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

# Conda aliases
alias ca='conda activate'
alias cda='conda deactivate'
alias cl='conda list'
alias ce='conda env list'
alias ci='conda install'
alias cu='conda update'
alias cr='conda remove'

# =================================================
# 7. GIT CONFIGURATION
# =================================================

# Git aliases
alias g='git'
alias gs='git status'
alias ga='git add'
alias gaa='git add .'
alias gc='git commit'
alias gcm='git commit -m'
alias gp='git push'
alias gpl='git pull'
alias gd='git diff'
alias gl='git log --oneline'
alias gb='git branch'
alias gco='git checkout'
alias gcb='git checkout -b'
alias gm='git merge'
alias gr='git remote'
alias gf='git fetch'

# Git status in PS1 (defined later in prompt section)
parse_git_branch() {
    git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/(\1)/'
}

# =================================================
# 8. FILE OPERATIONS AND NAVIGATION
# =================================================

# Modern Unix tool aliases (if available)
if command -v exa &> /dev/null; then
    alias ls='exa'
    alias ll='exa -la'
    alias la='exa -la'
    alias tree='exa --tree'
else
    alias ll='ls -alF'
    alias la='ls -A'
    alias l='ls -CF'
fi

if command -v bat &> /dev/null; then
    alias cat='bat'
    alias c='bat'
fi

if command -v fd &> /dev/null; then
    alias find='fd'
fi

if command -v rg &> /dev/null; then
    alias grep='rg'
fi

if command -v dust &> /dev/null; then
    alias du='dust'
fi

# Traditional file operations
alias ..='cd ..'
alias ...='cd ../..'
alias ....='cd ../../..'
alias ~='cd ~'
alias cp='cp -i'
alias mv='mv -i'
alias rm='rm -i'
alias ln='ln -i'
alias mkdir='mkdir -pv'

# Directory shortcuts
alias proj='cd $PROJECTS_DIR'
alias data='cd $DATA_DIR'
alias scripts='cd $SCRIPTS_DIR'
alias docs='cd ~/Documents'
alias dl='cd ~/Downloads'
alias dt='cd ~/Desktop'

# Archive extraction
extract() {
    if [ -f "$1" ]; then
        case $1 in
            *.tar.bz2)   tar xjf "$1"     ;;
            *.tar.gz)    tar xzf "$1"     ;;
            *.bz2)       bunzip2 "$1"     ;;
            *.rar)       unrar x "$1"     ;;
            *.gz)        gunzip "$1"      ;;
            *.tar)       tar xf "$1"      ;;
            *.tbz2)      tar xjf "$1"     ;;
            *.tgz)       tar xzf "$1"     ;;
            *.zip)       unzip "$1"       ;;
            *.Z)         uncompress "$1"  ;;
            *.7z)        7z x "$1"        ;;
            *)           echo "'$1' cannot be extracted via extract()" ;;
        esac
    else
        echo "'$1' is not a valid file"
    fi
}

# =================================================
# 9. SYSTEM MONITORING
# =================================================

# System information
alias cpu='htop'
alias mem='free -h'
alias disk='df -h'
alias temp='sensors'
alias ports='netstat -tuln'
alias proc='ps aux'

# System shortcuts
alias update='sudo apt update && sudo apt upgrade'  # Ubuntu/Debian
alias install='sudo apt install'                    # Ubuntu/Debian
alias search='apt search'                           # Ubuntu/Debian

# Process management
alias psg='ps aux | grep'
alias k9='kill -9'

# Network
alias myip='curl -s ifconfig.me'
alias localip='hostname -I'
alias ping='ping -c 5'

# =================================================
# 10. DATA SCIENCE SPECIFIC ALIASES
# =================================================

# Data file operations
alias head10='head -n 10'
alias tail10='tail -n 10'
alias count='wc -l'

# CSV operations (if csvkit is installed)
if command -v csvlook &> /dev/null; then
    alias csvhead='csvlook'
    alias csvinfo='csvstat'
fi

# JSON operations (if jq is installed)
if command -v jq &> /dev/null; then
    alias json='jq .'
    alias jsonkeys='jq keys'
fi

# Docker shortcuts
alias d='docker'
alias dc='docker-compose'
alias dps='docker ps'
alias di='docker images'
alias dex='docker exec -it'
alias dl='docker logs'

# =================================================
# 11. USEFUL FUNCTIONS
# =================================================

# Create and enter directory
mkcd() {
    mkdir -p "$1" && cd "$1"
}

# Find and replace in files
findreplace() {
    if [ $# -ne 3 ]; then
        echo "Usage: findreplace <directory> <find> <replace>"
        return 1
    fi
    grep -rl "$2" "$1" | xargs sed -i "s/$2/$3/g"
}

# Quick backup
backup() {
    if [ $# -ne 1 ]; then
        echo "Usage: backup <file_or_directory>"
        return 1
    fi
    cp -r "$1" "${1}.backup.$(date +%Y%m%d_%H%M%S)"
}

# Create project structure
create_project() {
    if [ $# -ne 1 ]; then
        echo "Usage: create_project <project_name>"
        return 1
    fi
    
    local project_name="$1"
    local project_dir="$PROJECTS_DIR/$project_name"
    
    mkdir -p "$project_dir"/{data,notebooks,scripts,output,docs}
    cd "$project_dir"
    
    # Initialize git
    git init
    
    # Create README
    echo "# $project_name" > README.md
    echo "" >> README.md
    echo "## Project Structure" >> README.md
    echo "- \`data/\`: Raw and processed data" >> README.md
    echo "- \`notebooks/\`: Jupyter notebooks" >> README.md
    echo "- \`scripts/\`: Python/R scripts" >> README.md
    echo "- \`output/\`: Results and visualizations" >> README.md
    echo "- \`docs/\`: Documentation" >> README.md
    
    # Create .gitignore
    cat > .gitignore << 'EOF'
# Data files
data/
output/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt
.env

# Jupyter Notebook
.ipynb_checkpoints

# R
.Rhistory
.RData
.Ruserdata
*.Rproj
.Rproj.user/

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/

# Temporary files
*.tmp
*.temp
*~
EOF
    
    echo "Project '$project_name' created at $project_dir"
}

# System information function
sysinfo() {
    echo "System Information:"
    echo "=================="
    echo "Hostname: $(hostname)"
    echo "OS: $(lsb_release -d | cut -f2 2>/dev/null || echo "Unknown")"
    echo "Kernel: $(uname -r)"
    echo "Uptime: $(uptime -p 2>/dev/null || uptime)"
    echo "CPU: $(lscpu | grep 'Model name' | cut -f 2 -d ':' | awk '{$1=$1}1' 2>/dev/null || echo "Unknown")"
    echo "Memory: $(free -h | awk '/^Mem:/ {print $2}' 2>/dev/null || echo "Unknown")"
    echo "Disk: $(df -h / | awk 'NR==2 {print $2}' 2>/dev/null || echo "Unknown")"
    echo "Shell: $SHELL"
    echo "Terminal: $TERM"
    
    # Programming languages
    echo ""
    echo "Programming Languages:"
    echo "====================="
    
    if command -v python &> /dev/null; then
        echo "Python: $(python --version 2>&1 | awk '{print $2}')"
    fi
    
    if command -v python3 &> /dev/null; then
        echo "Python3: $(python3 --version 2>&1 | awk '{print $2}')"
    fi
    
    if command -v R &> /dev/null; then
        echo "R: $(R --version | head -n1 | awk '{print $3}')"
    fi
    
    if command -v node &> /dev/null; then
        echo "Node.js: $(node --version)"
    fi
    
    if command -v git &> /dev/null; then
        echo "Git: $(git --version | awk '{print $3}')"
    fi
    
    if command -v docker &> /dev/null; then
        echo "Docker: $(docker --version | awk '{print $3}' | sed 's/,//')"
    fi
}

# Data file preview function
preview() {
    if [ $# -ne 1 ]; then
        echo "Usage: preview <file>"
        return 1
    fi
    
    local file="$1"
    
    if [[ ! -f "$file" ]]; then
        echo "File not found: $file"
        return 1
    fi
    
    case "${file##*.}" in
        csv)
            if command -v csvlook &> /dev/null; then
                head -n 10 "$file" | csvlook
            else
                head -n 10 "$file" | column -t -s ','
            fi
            ;;
        json)
            if command -v jq &> /dev/null; then
                head -n 20 "$file" | jq '.'
            else
                head -n 20 "$file"
            fi
            ;;
        yaml|yml)
            if command -v yq &> /dev/null; then
                head -n 20 "$file" | yq '.'
            else
                head -n 20 "$file"
            fi
            ;;
        *)
            if command -v bat &> /dev/null; then
                bat --paging=never "$file" | head -n 20
            else
                head -n 20 "$file"
            fi
            ;;
    esac
}

# Quick conda environment creation
condaenv() {
    if [ $# -lt 1 ]; then
        echo "Usage: condaenv <env_name> [python_version]"
        return 1
    fi
    
    local env_name="$1"
    local python_version="${2:-3.11}"
    
    conda create -n "$env_name" python="$python_version" -y
    conda activate "$env_name"
    
    echo "Environment '$env_name' created and activated with Python $python_version"
}

# =================================================
# 12. PROMPT CONFIGURATION
# =================================================

# Colors
RED='\[\033[01;31m\]'
GREEN='\[\033[01;32m\]'
YELLOW='\[\033[01;33m\]'
BLUE='\[\033[01;34m\]'
PURPLE='\[\033[01;35m\]'
CYAN='\[\033[01;36m\]'
WHITE='\[\033[01;37m\]'
RESET='\[\033[00m\]'

# Function to get current conda environment
get_conda_env() {
    if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
        echo "($CONDA_DEFAULT_ENV) "
    fi
}

# Function to get current virtualenv
get_virtualenv() {
    if [[ -n "$VIRTUAL_ENV" ]]; then
        echo "($(basename "$VIRTUAL_ENV")) "
    fi
}

# Set up prompt
setup_prompt() {
    local conda_env git_branch venv
    
    # Only set up git prompt if we're in a git repository
    if git rev-parse --git-dir >/dev/null 2>&1; then
        git_branch=" ${YELLOW}$(parse_git_branch)${RESET}"
    else
        git_branch=""
    fi
    
    PS1="${GREEN}$(get_conda_env)$(get_virtualenv)${BLUE}\u@\h${RESET}:${PURPLE}\w${git_branch}${RESET}\$ "
}

# Set prompt command to update prompt
PROMPT_COMMAND=setup_prompt

# =================================================
# 13. COMPLETION ENHANCEMENTS
# =================================================

# Enhanced completion for common commands
if command -v complete &> /dev/null; then
    # Git completion
    if [ -f /usr/share/bash-completion/completions/git ]; then
        source /usr/share/bash-completion/completions/git
        __git_complete g _git
        __git_complete gs _git_status
        __git_complete ga _git_add
        __git_complete gc _git_commit
        __git_complete gco _git_checkout
    fi
    
    # Conda completion
    if command -v conda &> /dev/null; then
        complete -W "$(conda env list | awk 'NR>2 {print $1}')" ca
    fi
fi

# =================================================
# 14. STARTUP ACTIONS
# =================================================

# Display system info on startup (optional)
if [[ "$-" == *i* ]] && [[ -z "$TMUX" ]]; then
    echo "Welcome to your data science environment!"
    echo "Current directory: $(pwd)"
    
    if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
        echo "Active conda environment: $CONDA_DEFAULT_ENV"
    fi
    
    if [[ -n "$VIRTUAL_ENV" ]]; then
        echo "Active virtual environment: $(basename "$VIRTUAL_ENV")"
    fi
    
    echo "Type 'sysinfo' for system information"
    echo ""
fi

# =================================================
# 15. LOCAL CUSTOMIZATIONS
# =================================================

# Source local bashrc if it exists
if [[ -f ~/.bashrc.local ]]; then
    source ~/.bashrc.local
fi

# Source machine-specific configurations
if [[ -f ~/.bashrc.$(hostname) ]]; then
    source ~/.bashrc.$(hostname)
fi

# =================================================
# 16. CLEANUP AND OPTIMIZATION
# =================================================

# Remove duplicate entries from PATH
export PATH="$(echo "$PATH" | awk -v RS=':' '!a[$1]++' | tr '\n' ':')"
export PATH="${PATH%:}"  # Remove trailing colon

# Unset functions that are no longer needed
unset -f add_to_path

# =================================================
# END OF CONFIGURATION
# =================================================

# Enable color support for ls and grep
if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    alias ls='ls --color=auto'
    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi

# Final message
if [[ "$-" == *i* ]]; then
    echo "Bash configuration loaded successfully!"
fi 