# Ollama Setup Guide

This guide provides comprehensive instructions for installing and configuring Ollama for use with the Enhanced YouTube Transcript Processing system.

## Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Model Management](#model-management)
5. [Configuration](#configuration)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Configuration](#advanced-configuration)

## Overview

Ollama is a local LLM serving platform that allows you to run large language models on your own hardware. This provides:

- **Privacy**: Your data never leaves your machine
- **Cost Control**: No per-token charges
- **Reliability**: No dependency on external API availability
- **Customization**: Ability to use specialized models

## System Requirements

### Minimum Requirements
- **CPU**: 4 cores, 2.0 GHz
- **RAM**: 8 GB (16 GB recommended)
- **Storage**: 20 GB free space (models can be 4-7 GB each)
- **OS**: macOS, Linux, or Windows (WSL2)

### Recommended Requirements
- **CPU**: 8+ cores, 3.0+ GHz
- **RAM**: 16-32 GB
- **GPU**: NVIDIA GPU with 8+ GB VRAM (optional but highly recommended)
- **Storage**: 100+ GB SSD space

### GPU Support
- **NVIDIA**: CUDA 11.8+ drivers
- **AMD**: ROCm support (Linux only)
- **Apple Silicon**: Native support on M1/M2/M3 Macs

## Installation

### macOS

```bash
# Method 1: Using Homebrew (recommended)
brew install ollama

# Method 2: Direct download
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve
```

### Linux

```bash
# Method 1: Installation script
curl -fsSL https://ollama.ai/install.sh | sh

# Method 2: Manual installation
# Download from https://ollama.ai/download/linux
# Extract and place in /usr/local/bin/

# Start Ollama service
ollama serve

# Or run as systemd service
sudo systemctl enable ollama
sudo systemctl start ollama
```

### Windows

1. Download Ollama for Windows from https://ollama.ai/download/windows
2. Run the installer
3. Ollama will start automatically as a Windows service

### Docker Installation

```bash
# Pull and run Ollama container
docker run -d \
  --name ollama \
  -p 11434:11434 \
  -v ollama:/root/.ollama \
  --restart unless-stopped \
  ollama/ollama

# With GPU support (NVIDIA)
docker run -d \
  --name ollama \
  --gpus all \
  -p 11434:11434 \
  -v ollama:/root/.ollama \
  --restart unless-stopped \
  ollama/ollama
```

## Model Management

### Recommended Models for YouTube Transcript Processing

#### General Purpose Models

```bash
# Llama 3.1 8B - Best balance of performance and speed
ollama pull llama3.1:8b

# Llama 3.2 3B - Lightweight option for slower systems
ollama pull llama3.2:3b

# Mistral 7B - Fast and efficient
ollama pull mistral:7b
```

#### Specialized Models

```bash
# For Chinese language content
ollama pull qwen2.5:7b

# For code and technical content
ollama pull codellama:7b

# For instruction following
ollama pull llama3.1:8b-instruct
```

### Model Management Commands

```bash
# List available models online
ollama list

# Pull a specific model
ollama pull <model-name>

# List installed models
ollama list

# Remove a model
ollama rm <model-name>

# Show model information
ollama show <model-name>

# Update all models
ollama pull --all
```

### Model Selection Guide

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| llama3.2:3b | 2GB | Fast | Good | Quick summaries, simple tasks |
| mistral:7b | 4GB | Fast | Very Good | General purpose, balanced |
| llama3.1:8b | 5GB | Medium | Excellent | Best overall choice |
| qwen2.5:7b | 4GB | Medium | Excellent | Chinese content |
| codellama:7b | 4GB | Medium | Good | Technical content |

## Configuration

### Environment Variables

Create or update your `.env` file:

```bash
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_KEEP_ALIVE=5m
OLLAMA_CONNECTION_TIMEOUT=10

# Model Selection Strategy
MODEL_SELECTION_STRATEGY=prefer_local
OLLAMA_FALLBACK_ENABLED=true
OLLAMA_FALLBACK_PROVIDER=openai

# Performance Tuning
OLLAMA_AUTO_PULL=true
OLLAMA_MODEL_CACHE_SIZE=5
OLLAMA_GPU_MEMORY_FRACTION=0.8
OLLAMA_CPU_THREADS=0  # 0 = auto-detect
OLLAMA_CONTEXT_SIZE=4096

# Language-Specific Models
CHINESE_LANGUAGE_MODEL=qwen2.5:7b
PERFORMANCE_MODEL=mistral:7b
LIGHTWEIGHT_MODEL=llama3.2:3b
```

### Application Configuration

```python
# In your application code
from src.config import settings

# Check Ollama configuration
if settings.default_llm_provider == 'ollama':
    print(f"Ollama Host: {settings.ollama_host}")
    print(f"Default Model: {settings.ollama_model}")
    print(f"Fallback Enabled: {settings.ollama_fallback_enabled}")
```

## Performance Optimization

### Hardware Optimization

#### CPU Optimization
```bash
# Set CPU thread count (0 = auto-detect optimal)
export OLLAMA_NUM_THREADS=0

# For systems with many cores, you might want to limit
export OLLAMA_NUM_THREADS=8
```

#### GPU Optimization
```bash
# Use GPU acceleration (auto-detected)
export OLLAMA_GPU_ACCELERATION=true

# Specify GPU memory fraction
export OLLAMA_GPU_MEMORY_FRACTION=0.8

# For multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1
```

#### Memory Optimization
```bash
# Adjust model context size
export OLLAMA_CONTEXT_SIZE=4096  # Smaller = less memory

# Keep models in memory longer
export OLLAMA_KEEP_ALIVE=10m

# Preload specific models
ollama run llama3.1:8b --keep-alive 30m
```

### Model Optimization

#### Quantized Models
```bash
# Use quantized versions for better performance
ollama pull llama3.1:8b-q4_0    # 4-bit quantization
ollama pull llama3.1:8b-q8_0    # 8-bit quantization
```

#### Custom Model Configuration
```bash
# Create custom model with specific parameters
cat > Modelfile << EOF
FROM llama3.1:8b
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER max_tokens 4000
EOF

ollama create youtube-summarizer -f Modelfile
```

## Troubleshooting

### Common Issues

#### 1. Ollama Service Not Starting
```bash
# Check if service is running
pgrep ollama

# Check logs
journalctl -u ollama -f  # Linux
tail -f ~/.ollama/logs/server.log  # macOS

# Restart service
sudo systemctl restart ollama  # Linux
brew services restart ollama    # macOS
```

#### 2. Model Download Failures
```bash
# Check available disk space
df -h

# Clear cache and retry
rm -rf ~/.ollama/models/<model-name>
ollama pull <model-name>

# Use specific model version
ollama pull llama3.1:8b-instruct-q4_0
```

#### 3. Out of Memory Errors
```bash
# Use smaller model
ollama pull llama3.2:3b

# Reduce context size
export OLLAMA_CONTEXT_SIZE=2048

# Clear GPU memory
nvidia-smi --gpu-reset

# Check memory usage
nvidia-smi  # GPU memory
free -h     # System memory
```

#### 4. Slow Performance
```bash
# Check GPU utilization
nvidia-smi

# Verify model is using GPU
ollama ps

# Use quantized model
ollama pull llama3.1:8b-q4_0

# Adjust CPU threads
export OLLAMA_NUM_THREADS=4
```

### Health Checks

```bash
# Test Ollama API
curl http://localhost:11434/api/tags

# Test model inference
curl http://localhost:11434/api/generate \\
  -d '{
    "model": "llama3.1:8b",
    "prompt": "Test prompt",
    "stream": false
  }'

# Check model status
ollama ps
```

### Performance Monitoring

```bash
# Monitor system resources
htop
nvidia-smi -l 1  # For GPU monitoring

# Check Ollama logs for performance metrics
tail -f ~/.ollama/logs/server.log | grep -E "(duration|tokens)"
```

## Advanced Configuration

### Multi-Model Setup

```bash
# Install multiple models for different use cases
ollama pull llama3.1:8b        # General purpose
ollama pull qwen2.5:7b         # Chinese content
ollama pull codellama:7b       # Technical content
ollama pull llama3.2:3b        # Lightweight fallback
```

### Load Balancing

```python
# Configure model load balancing in application
MODEL_LOAD_BALANCING=true
FALLBACK_MODEL_CHAIN=llama3.1:8b,mistral:7b,llama3.2:3b
```

### Custom API Endpoints

```bash
# Run Ollama on custom port
OLLAMA_HOST=0.0.0.0:11435 ollama serve

# Or set in environment
export OLLAMA_HOST=0.0.0.0:11435
```

### Security Configuration

```bash
# Run with authentication (requires reverse proxy)
# Use nginx or traefik for authentication
# Example nginx config for basic auth:

# /etc/nginx/sites-available/ollama
server {
    listen 8080;
    server_name localhost;
    
    auth_basic "Ollama API";
    auth_basic_user_file /etc/nginx/.htpasswd;
    
    location / {
        proxy_pass http://localhost:11434;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Batch Processing Optimization

```python
# Optimize for batch processing
import asyncio
import aiohttp

async def process_batch(prompts):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for prompt in prompts:
            task = process_single_prompt(session, prompt)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results
```

### Model Preloading

```bash
# Preload models for faster response times
ollama run llama3.1:8b --keep-alive 60m &
ollama run qwen2.5:7b --keep-alive 30m &
ollama run mistral:7b --keep-alive 30m &
```

## Integration with YouTube Summarizer

### Automatic Model Selection

The application automatically selects the best model based on:

1. **Content Language**: Uses `qwen2.5:7b` for Chinese content
2. **Content Length**: Uses `llama3.2:3b` for short videos
3. **Performance Requirements**: Uses `mistral:7b` for fast processing
4. **Fallback Chain**: Falls back through model hierarchy if needed

### Configuration Examples

```bash
# Minimal configuration (development)
OLLAMA_MODEL=llama3.2:3b
MODEL_SELECTION_STRATEGY=prefer_local

# Balanced configuration (production)
OLLAMA_MODEL=llama3.1:8b
MODEL_SELECTION_STRATEGY=auto
OLLAMA_FALLBACK_ENABLED=true

# High-performance configuration
OLLAMA_MODEL=mistral:7b
MODEL_SELECTION_STRATEGY=prefer_local
OLLAMA_GPU_MEMORY_FRACTION=0.9
OLLAMA_CONTEXT_SIZE=8192
```

## Getting Help

- **Ollama Documentation**: https://ollama.ai/docs
- **GitHub Issues**: https://github.com/ollama/ollama/issues
- **Discord Community**: https://discord.gg/ollama
- **Model Library**: https://ollama.ai/library

## Quick Reference

### Essential Commands
```bash
# Service management
ollama serve                    # Start service
ollama ps                      # Show running models
ollama list                    # List installed models

# Model management
ollama pull <model>            # Install model
ollama rm <model>              # Remove model
ollama show <model>            # Show model info

# Testing
curl localhost:11434/api/tags  # Test API
ollama run <model> "test"      # Quick test
```

### Environment Variables Quick Reference
```bash
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_KEEP_ALIVE=5m
OLLAMA_GPU_MEMORY_FRACTION=0.8
OLLAMA_NUM_THREADS=0
MODEL_SELECTION_STRATEGY=auto
```