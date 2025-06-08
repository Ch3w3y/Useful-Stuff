# Development Environment Dotfiles

## Overview

Comprehensive collection of dotfiles and configuration files for optimizing development environments across different operating systems and tools. Includes shell configurations, editor settings, and productivity tools.

## Table of Contents

- [Shell Configuration](#shell-configuration)
- [Development Tools](#development-tools)
- [Editor Configuration](#editor-configuration)
- [Terminal Setup](#terminal-setup)
- [Git Configuration](#git-configuration)
- [Python Environment](#python-environment)
- [System Utilities](#system-utilities)
- [Installation & Management](#installation--management)

## Shell Configuration

### Zsh Configuration (.zshrc)

```bash
# ~/.zshrc - Comprehensive Zsh configuration

# Oh My Zsh configuration
export ZSH="$HOME/.oh-my-zsh"
ZSH_THEME="powerlevel10k/powerlevel10k"

# Plugins
plugins=(
    git
    zsh-autosuggestions
    zsh-syntax-highlighting
    docker
    kubectl
    python
    pip
    virtualenv
    aws
    node
    npm
    yarn
    rust
    golang
    terraform
    ansible
)

source $ZSH/oh-my-zsh.sh

# Custom aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias grep='grep --color=auto'
alias fgrep='fgrep --color=auto'
alias egrep='egrep --color=auto'

# Development aliases
alias python='python3'
alias pip='pip3'
alias k='kubectl'
alias tf='terraform'
alias dc='docker-compose'
alias d='docker'

# Git aliases
alias g='git'
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git pull'
alias gd='git diff'
alias gb='git branch'
alias gco='git checkout'
alias gm='git merge'
alias gr='git rebase'

# Navigation aliases
alias ..='cd ..'
alias ...='cd ../..'
alias ....='cd ../../..'
alias .....='cd ../../../..'

# Custom functions
function mkcd() {
    mkdir -p "$1" && cd "$1"
}

function extract() {
    if [ -f $1 ] ; then
        case $1 in
            *.tar.bz2)   tar xjf $1     ;;
            *.tar.gz)    tar xzf $1     ;;
            *.bz2)       bunzip2 $1     ;;
            *.rar)       unrar e $1     ;;
            *.gz)        gunzip $1      ;;
            *.tar)       tar xf $1      ;;
            *.tbz2)      tar xjf $1     ;;
            *.tgz)       tar xzf $1     ;;
            *.zip)       unzip $1       ;;
            *.Z)         uncompress $1  ;;
            *.7z)        7z x $1        ;;
            *)           echo "'$1' cannot be extracted via extract()" ;;
        esac
    else
        echo "'$1' is not a valid file"
    fi
}

# Environment variables
export EDITOR='nvim'
export VISUAL='nvim'
export PAGER='less'
export BROWSER='firefox'

# Development paths
export PATH="$HOME/.local/bin:$PATH"
export PATH="$HOME/bin:$PATH"

# Python configuration
export PYTHONPATH="$HOME/.local/lib/python3.9/site-packages:$PYTHONPATH"
export PIPENV_VENV_IN_PROJECT=1

# Node.js configuration
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"

# Rust configuration
source "$HOME/.cargo/env"

# Go configuration
export GOPATH="$HOME/go"
export PATH="$GOPATH/bin:$PATH"

# AWS CLI completion
autoload bashcompinit && bashcompinit
autoload -Uz compinit && compinit
complete -C '/usr/local/bin/aws_completer' aws

# History configuration
HISTSIZE=10000
SAVEHIST=10000
setopt HIST_VERIFY
setopt SHARE_HISTORY
setopt APPEND_HISTORY
setopt INC_APPEND_HISTORY
setopt HIST_IGNORE_DUPS
setopt HIST_IGNORE_ALL_DUPS
setopt HIST_REDUCE_BLANKS
setopt HIST_IGNORE_SPACE

# Key bindings
bindkey "^[[A" history-search-backward
bindkey "^[[B" history-search-forward
bindkey "^[[H" beginning-of-line
bindkey "^[[F" end-of-line
bindkey "^[[3~" delete-char

# Auto-completion configuration
zstyle ':completion:*' auto-description 'specify: %d'
zstyle ':completion:*' completer _expand _complete _correct _approximate
zstyle ':completion:*' format 'Completing %d'
zstyle ':completion:*' group-name ''
zstyle ':completion:*' menu select=2
zstyle ':completion:*:default' list-colors ${(s.:.)LS_COLORS}
zstyle ':completion:*' list-colors ''
zstyle ':completion:*' list-prompt %SAt %p: Hit TAB for more, or the character to insert%s
zstyle ':completion:*' matcher-list '' 'm:{a-z}={A-Z}' 'm:{a-zA-Z}={A-Za-z}' 'r:|[._-]=* r:|=* l:|=*'
zstyle ':completion:*' menu select=long
zstyle ':completion:*' select-prompt %SScrolling active: current selection at %p%s
zstyle ':completion:*' use-compctl false
zstyle ':completion:*' verbose true

# Load custom configurations if they exist
[ -f ~/.zshrc.local ] && source ~/.zshrc.local
```

### Bash Configuration (.bashrc)

```bash
# ~/.bashrc - Comprehensive Bash configuration

# Source global definitions
if [ -f /etc/bashrc ]; then
    . /etc/bashrc
fi

# User specific aliases and functions
if [ -d ~/.bashrc.d ]; then
    for rc in ~/.bashrc.d/*; do
        if [ -f "$rc" ]; then
            . "$rc"
        fi
    done
fi
unset rc

# Colors for ls
export CLICOLOR=1
export LSCOLORS=GxFxCxDxBxegedabagaced

# Prompt configuration
PS1='\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '

# History configuration
HISTSIZE=10000
HISTFILESIZE=20000
HISTCONTROL=ignoreboth
shopt -s histappend
shopt -s checkwinsize

# Enable programmable completion features
if ! shopt -oq posix; then
  if [ -f /usr/share/bash-completion/bash_completion ]; then
    . /usr/share/bash-completion/bash_completion
  elif [ -f /etc/bash_completion ]; then
    . /etc/bash_completion
  fi
fi

# Common aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias grep='grep --color=auto'
alias fgrep='fgrep --color=auto'
alias egrep='egrep --color=auto'

# Development aliases
alias python='python3'
alias pip='pip3'
alias k='kubectl'
alias tf='terraform'
alias dc='docker-compose'
alias d='docker'

# Git integration for prompt
if [ -f ~/.git-prompt.sh ]; then
    source ~/.git-prompt.sh
    export GIT_PS1_SHOWDIRTYSTATE=1
    export GIT_PS1_SHOWSTASHSTATE=1
    export GIT_PS1_SHOWUNTRACKEDFILES=1
    export GIT_PS1_SHOWUPSTREAM="auto"
    PS1='\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[33m\]$(__git_ps1 " (%s)")\[\033[00m\]\$ '
fi

# Load local customizations
[ -f ~/.bashrc.local ] && source ~/.bashrc.local
```

### Fish Configuration

```fish
# ~/.config/fish/config.fish - Fish shell configuration

# Set default editor
set -gx EDITOR nvim
set -gx VISUAL nvim

# Development paths
set -gx PATH $HOME/.local/bin $PATH
set -gx PATH $HOME/bin $PATH

# Python configuration
set -gx PIPENV_VENV_IN_PROJECT 1

# Go configuration
set -gx GOPATH $HOME/go
set -gx PATH $GOPATH/bin $PATH

# Rust configuration
set -gx PATH $HOME/.cargo/bin $PATH

# Aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias python='python3'
alias pip='pip3'
alias k='kubectl'
alias tf='terraform'
alias dc='docker-compose'
alias d='docker'

# Git aliases
alias g='git'
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git pull'
alias gd='git diff'
alias gb='git branch'
alias gco='git checkout'

# Functions
function mkcd
    mkdir -p $argv[1]; and cd $argv[1]
end

function extract
    switch $argv[1]
        case "*.tar.bz2"
            tar xjf $argv[1]
        case "*.tar.gz"
            tar xzf $argv[1]
        case "*.bz2"
            bunzip2 $argv[1]
        case "*.rar"
            unrar e $argv[1]
        case "*.gz"
            gunzip $argv[1]
        case "*.tar"
            tar xf $argv[1]
        case "*.tbz2"
            tar xjf $argv[1]
        case "*.tgz"
            tar xzf $argv[1]
        case "*.zip"
            unzip $argv[1]
        case "*.Z"
            uncompress $argv[1]
        case "*.7z"
            7z x $argv[1]
        case '*'
            echo "'$argv[1]' cannot be extracted via extract()"
    end
end

# Initialize starship prompt
starship init fish | source
```

## Development Tools

### Tmux Configuration (.tmux.conf)

```bash
# ~/.tmux.conf - Tmux configuration

# Change prefix key to Ctrl-a
unbind C-b
set-option -g prefix C-a
bind-key C-a send-prefix

# Split panes using | and -
bind | split-window -h
bind - split-window -v
unbind '"'
unbind %

# Reload config file
bind r source-file ~/.tmux.conf \; display-message "Config reloaded!"

# Switch panes using Alt-arrow without prefix
bind -n M-Left select-pane -L
bind -n M-Right select-pane -R
bind -n M-Up select-pane -U
bind -n M-Down select-pane -D

# Enable mouse mode
set -g mouse on

# Don't rename windows automatically
set-option -g allow-rename off

# Set default terminal mode to 256color mode
set -g default-terminal "screen-256color"

# Enable activity monitoring
setw -g monitor-activity on
set -g visual-activity on

# Status bar configuration
set -g status-position bottom
set -g status-bg colour234
set -g status-fg colour137
set -g status-left ''
set -g status-right '#[fg=colour233,bg=colour241,bold] %d/%m #[fg=colour233,bg=colour245,bold] %H:%M:%S '
set -g status-right-length 50
set -g status-left-length 20

# Window status configuration
setw -g window-status-current-format ' #I#[fg=colour250]:#[fg=colour255]#W#[fg=colour50]#F '
setw -g window-status-format ' #I#[fg=colour237]:#[fg=colour250]#W#[fg=colour244]#F '

# Pane border colors
set -g pane-border-fg colour238
set -g pane-active-border-fg colour51

# Message colors
set -g message-bg colour166
set -g message-fg colour232

# Copy mode vim bindings
setw -g mode-keys vi
bind-key -T copy-mode-vi 'v' send -X begin-selection
bind-key -T copy-mode-vi 'y' send -X copy-selection-and-cancel

# List of plugins
set -g @plugin 'tmux-plugins/tpm'
set -g @plugin 'tmux-plugins/tmux-sensible'
set -g @plugin 'tmux-plugins/tmux-resurrect'
set -g @plugin 'tmux-plugins/tmux-continuum'

# Plugin settings
set -g @continuum-restore 'on'

# Initialize TMUX plugin manager
run '~/.tmux/plugins/tpm/tpm'
```

### Git Configuration (.gitconfig)

```ini
[user]
    name = Your Name
    email = your.email@example.com

[core]
    editor = nvim
    pager = less -FMRiX
    autocrlf = input
    safecrlf = true
    excludesfile = ~/.gitignore_global

[color]
    ui = auto
    branch = auto
    diff = auto
    status = auto

[color "branch"]
    current = yellow reverse
    local = yellow
    remote = green

[color "diff"]
    meta = yellow bold
    frag = magenta bold
    old = red bold
    new = green bold

[color "status"]
    added = yellow
    changed = green
    untracked = cyan

[alias]
    st = status
    ci = commit
    co = checkout
    br = branch
    df = diff
    dc = diff --cached
    lg = log --oneline --graph --decorate --all
    lol = log --graph --decorate --pretty=oneline --abbrev-commit
    lola = log --graph --decorate --pretty=oneline --abbrev-commit --all
    ls = ls-files
    ign = ls-files -o -i --exclude-standard
    assume = update-index --assume-unchanged
    unassume = update-index --no-assume-unchanged
    assumed = "!git ls-files -v | grep ^h | cut -c 3-"
    snapshot = !git stash save "snapshot: $(date)" && git stash apply "stash@{0}"

[push]
    default = simple

[pull]
    rebase = true

[merge]
    tool = vimdiff

[diff]
    tool = vimdiff

[credential]
    helper = cache --timeout=3600

[init]
    defaultBranch = main

[filter "lfs"]
    clean = git-lfs clean -- %f
    smudge = git-lfs smudge -- %f
    process = git-lfs filter-process
    required = true
```

### Global Git Ignore (.gitignore_global)

```gitignore
# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Editor files
*.swp
*.swo
*~
.vscode/
.idea/
*.sublime-*

# Logs
*.log
logs/

# Runtime data
pids
*.pid
*.seed

# Coverage directory used by tools like istanbul
coverage/

# Dependency directories
node_modules/
bower_components/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/
.pytest_cache/

# Virtual environments
.venv
.env

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

# Package managers
package-lock.json
yarn.lock

# Temporary files
*.tmp
*.temp
.tmp/
.temp/
```

## Editor Configuration

### Neovim Configuration (init.vim)

```vim
" ~/.config/nvim/init.vim - Neovim configuration

" Basic settings
set number
set relativenumber
set tabstop=4
set shiftwidth=4
set expandtab
set smartindent
set wrap
set smartcase
set noswapfile
set nobackup
set undodir=~/.vim/undodir
set undofile
set incsearch
set scrolloff=8
set colorcolumn=80
set signcolumn=yes

" Leader key
let mapleader = " "

" Plugins using vim-plug
call plug#begin('~/.vim/plugged')

" Essential plugins
Plug 'nvim-lua/plenary.nvim'
Plug 'nvim-telescope/telescope.nvim'
Plug 'nvim-treesitter/nvim-treesitter', {'do': ':TSUpdate'}
Plug 'nvim-tree/nvim-tree.lua'
Plug 'nvim-lualine/lualine.nvim'
Plug 'kyazdani42/nvim-web-devicons'

" LSP and completion
Plug 'neovim/nvim-lspconfig'
Plug 'hrsh7th/nvim-cmp'
Plug 'hrsh7th/cmp-nvim-lsp'
Plug 'hrsh7th/cmp-buffer'
Plug 'hrsh7th/cmp-path'
Plug 'hrsh7th/cmp-cmdline'
Plug 'L3MON4D3/LuaSnip'
Plug 'saadparwaiz1/cmp_luasnip'

" Git integration
Plug 'tpope/vim-fugitive'
Plug 'lewis6991/gitsigns.nvim'

" Color schemes
Plug 'gruvbox-community/gruvbox'
Plug 'folke/tokyonight.nvim'

" Utility plugins
Plug 'windwp/nvim-autopairs'
Plug 'numToStr/Comment.nvim'
Plug 'tpope/vim-surround'
Plug 'folke/which-key.nvim'

" Language specific
Plug 'davidhalter/jedi-vim'
Plug 'fatih/vim-go'
Plug 'rust-lang/rust.vim'

call plug#end()

" Color scheme
colorscheme gruvbox
set background=dark

" Key mappings
nnoremap <leader>pf <cmd>Telescope find_files<cr>
nnoremap <leader>ps <cmd>Telescope live_grep<cr>
nnoremap <leader>pb <cmd>Telescope buffers<cr>
nnoremap <leader>ph <cmd>Telescope help_tags<cr>

" File explorer
nnoremap <leader>e <cmd>NvimTreeToggle<cr>

" Buffer navigation
nnoremap <leader>bn <cmd>bnext<cr>
nnoremap <leader>bp <cmd>bprevious<cr>
nnoremap <leader>bd <cmd>bdelete<cr>

" Git mappings
nnoremap <leader>gs <cmd>Git<cr>
nnoremap <leader>gc <cmd>Git commit<cr>
nnoremap <leader>gp <cmd>Git push<cr>

" LSP mappings
nnoremap gd <cmd>lua vim.lsp.buf.definition()<cr>
nnoremap K <cmd>lua vim.lsp.buf.hover()<cr>
nnoremap <leader>vws <cmd>lua vim.lsp.buf.workspace_symbol()<cr>
nnoremap <leader>vd <cmd>lua vim.diagnostic.open_float()<cr>
nnoremap [d <cmd>lua vim.diagnostic.goto_next()<cr>
nnoremap ]d <cmd>lua vim.diagnostic.goto_prev()<cr>
nnoremap <leader>vca <cmd>lua vim.lsp.buf.code_action()<cr>
nnoremap <leader>vrr <cmd>lua vim.lsp.buf.references()<cr>
nnoremap <leader>vrn <cmd>lua vim.lsp.buf.rename()<cr>

" Lua configuration
lua << EOF
-- LSP configuration
local lspconfig = require('lspconfig')

-- Python
lspconfig.pyright.setup{}

-- TypeScript/JavaScript
lspconfig.tsserver.setup{}

-- Go
lspconfig.gopls.setup{}

-- Rust
lspconfig.rust_analyzer.setup{}

-- Completion setup
local cmp = require'cmp'
cmp.setup({
  snippet = {
    expand = function(args)
      require('luasnip').lsp_expand(args.body)
    end,
  },
  mapping = cmp.mapping.preset.insert({
    ['<C-b>'] = cmp.mapping.scroll_docs(-4),
    ['<C-f>'] = cmp.mapping.scroll_docs(4),
    ['<C-Space>'] = cmp.mapping.complete(),
    ['<C-e>'] = cmp.mapping.abort(),
    ['<CR>'] = cmp.mapping.confirm({ select = true }),
  }),
  sources = cmp.config.sources({
    { name = 'nvim_lsp' },
    { name = 'luasnip' },
  }, {
    { name = 'buffer' },
  })
})

-- Treesitter configuration
require'nvim-treesitter.configs'.setup {
  ensure_installed = { "python", "javascript", "typescript", "go", "rust", "lua", "vim" },
  sync_install = false,
  highlight = {
    enable = true,
  },
}

-- Status line
require('lualine').setup()

-- Auto pairs
require('nvim-autopairs').setup()

-- Comments
require('Comment').setup()

-- Which key
require('which-key').setup()

-- Git signs
require('gitsigns').setup()
EOF
```

## Python Environment

### Python Development Configuration

```python
# ~/.config/python/startup.py - Python REPL enhancements

import os
import sys
import atexit
import readline
import rlcompleter

# Enable tab completion
readline.parse_and_bind("tab: complete")

# History file
histfile = os.path.join(os.path.expanduser("~"), ".python_history")
try:
    readline.read_history_file(histfile)
    readline.set_history_length(1000)
except FileNotFoundError:
    pass

atexit.register(readline.write_history_file, histfile)

# Pretty printing
import pprint
pp = pprint.pprint

# Common imports for data science
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    print("NumPy, Pandas, and Matplotlib imported as np, pd, and plt")
except ImportError:
    pass

print("Python startup script loaded")
```

### Pip Configuration

```ini
# ~/.pip/pip.conf - Pip configuration

[global]
timeout = 60
index-url = https://pypi.org/simple/
trusted-host = pypi.org
                pypi.python.org
                files.pythonhosted.org

[install]
upgrade-strategy = eager
user = true

[list]
format = columns
```

---

*This comprehensive dotfiles collection provides optimized development environment configurations for shells, editors, and development tools to enhance productivity and workflow efficiency.* 