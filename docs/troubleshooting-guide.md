# Troubleshooting Guide

## Overview

This comprehensive troubleshooting guide helps diagnose and resolve common issues with the YouTube Summarizer API. The guide is organized by problem categories and provides step-by-step solutions with detailed explanations.

## Table of Contents

1. [General Diagnostics](#general-diagnostics)
2. [Startup and Configuration Issues](#startup-and-configuration-issues)
3. [API Request Failures](#api-request-failures)
4. [Performance Issues](#performance-issues)
5. [Docker and Container Issues](#docker-and-container-issues)
6. [Network and Connectivity Issues](#network-and-connectivity-issues)
7. [External Service Integration Issues](#external-service-integration-issues)
8. [Security and Authentication Issues](#security-and-authentication-issues)
9. [Database and Caching Issues](#database-and-caching-issues)
10. [Monitoring and Logging Issues](#monitoring-and-logging-issues)

## General Diagnostics

### Quick Health Check

Before diving into specific issues, perform these basic checks:

```bash
# 1. Check if the application is running
curl -s http://localhost:8000/health | jq .

# 2. Check application logs
docker-compose logs app --tail=50

# 3. Check system resources
docker stats

# 4. Verify environment variables
docker-compose exec app env | grep -E "(API_KEY|ENVIRONMENT|PORT)"

# 5. Test basic functionality
curl -X POST "http://localhost:8000/api/v1/summarize" \
  -H "Content-Type: application/json" \
  -d '{"youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

### Log Analysis Commands

```bash
# View application logs
docker-compose logs app -f

# Search for errors
docker-compose logs app | grep -i error

# Check specific timeframe
docker-compose logs app --since="2023-01-01T10:00:00"

# Filter by log level
docker-compose logs app | grep -E "(ERROR|FATAL|WARNING)"

# Export logs for analysis
docker-compose logs app > app_logs_$(date +%Y%m%d).log
```

### System Information Collection

```bash
#!/bin/bash
# collect-diagnostics.sh

echo "=== System Information ==="
uname -a
docker --version
docker-compose --version

echo -e "\n=== Container Status ==="
docker-compose ps

echo -e "\n=== Resource Usage ==="
docker stats --no-stream

echo -e "\n=== Network Status ==="
docker network ls
netstat -tlnp | grep :8000

echo -e "\n=== Disk Usage ==="
df -h
docker system df

echo -e "\n=== Recent Logs ==="
docker-compose logs app --tail=20
```

## Startup and Configuration Issues

### Issue: Application Fails to Start

#### Symptoms
- Container exits immediately
- "Connection refused" errors
- Health check failures

#### Diagnosis
```bash
# Check container status
docker-compose ps

# View startup logs
docker-compose logs app

# Check for configuration errors
docker-compose config

# Verify environment file
ls -la .env && head .env
```

#### Common Causes and Solutions

1. **Missing Environment File**
   ```bash
   # Problem: .env file not found
   # Solution: Create from template
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Invalid Environment Variables**
   ```bash
   # Problem: Malformed environment variables
   # Check for syntax errors
   cat .env | grep -v '^#' | grep -v '^$'
   
   # Solution: Fix formatting
   # Remove spaces around = signs
   # Quote values with special characters
   ```

3. **Missing API Keys**
   ```bash
   # Problem: No valid API keys configured
   # Solution: Add at least one API key
   echo "OPENAI_API_KEY=your_key_here" >> .env
   # or
   echo "ANTHROPIC_API_KEY=your_key_here" >> .env
   ```

4. **Port Conflicts**
   ```bash
   # Problem: Port 8000 already in use
   # Check what's using the port
   lsof -i :8000
   netstat -tlnp | grep :8000
   
   # Solution: Change port or stop conflicting service
   # In .env file:
   echo "PORT=8001" >> .env
   ```

5. **Python Module Import Errors**
   ```bash
   # Problem: Missing dependencies
   # Check requirements installation
   docker-compose exec app pip list
   
   # Solution: Rebuild container
   docker-compose build --no-cache app
   docker-compose up -d
   ```

### Issue: Configuration Validation Errors

#### Symptoms
- Application starts but health check fails
- Validation errors in logs
- "Internal Server Error" responses

#### Solutions

1. **Validate Environment Configuration**
   ```python
   # Use this script to validate configuration
   import os
   from src.config import settings
   
   # Check required variables
   required_vars = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY']
   missing_vars = [var for var in required_vars if not os.getenv(var)]
   
   if len(missing_vars) == len(required_vars):
       print("ERROR: No API keys configured")
   
   # Validate settings
   try:
       config = settings
       print("Configuration loaded successfully")
   except Exception as e:
       print(f"Configuration error: {e}")
   ```

2. **Check LLM Provider Configuration**
   ```bash
   # Test OpenAI connection
   curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models | jq .
   
   # Test Anthropic connection
   curl -H "x-api-key: $ANTHROPIC_API_KEY" \
     https://api.anthropic.com/v1/messages \
     -X POST -d '{"model":"claude-3-sonnet-20240229","max_tokens":10,"messages":[{"role":"user","content":"Hi"}]}'
   ```

## API Request Failures

### Issue: 400 Bad Request Errors

#### Common Error Patterns

1. **Invalid YouTube URL Format**
   ```json
   {
     "error": {
       "code": "E1001",
       "message": "Invalid YouTube URL format"
     }
   }
   ```
   
   **Solution**: Verify URL format
   ```bash
   # Valid formats:
   # https://www.youtube.com/watch?v=VIDEO_ID
   # https://youtu.be/VIDEO_ID
   # https://m.youtube.com/watch?v=VIDEO_ID
   
   # Test with valid URL
   curl -X POST "http://localhost:8000/api/v1/summarize" \
     -H "Content-Type: application/json" \
     -d '{"youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
   ```

2. **Missing Request Body**
   ```bash
   # Problem: Empty or malformed JSON
   # Solution: Provide proper JSON body
   curl -X POST "http://localhost:8000/api/v1/summarize" \
     -H "Content-Type: application/json" \
     -d '{"youtube_url": "VALID_URL_HERE"}'
   ```

### Issue: 404 Not Found Errors

#### Symptoms
- Video not found errors
- "This video is unavailable" messages

#### Diagnosis and Solutions

```bash
# 1. Verify video exists and is public
youtube-dl --get-title "VIDEO_URL"

# 2. Check video accessibility
curl -I "VIDEO_URL"

# 3. Test with known working video
curl -X POST "http://localhost:8000/api/v1/summarize" \
  -H "Content-Type: application/json" \
  -d '{"youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

**Common Causes**:
- Private or unlisted videos
- Age-restricted content
- Geo-blocked videos
- Deleted or removed videos

### Issue: 422 Unprocessable Entity

#### Common Scenarios

1. **Video Too Long**
   ```json
   {
     "error": {
       "code": "E2005",
       "message": "Video duration exceeds maximum limit"
     }
   }
   ```
   
   **Solution**: Use shorter videos (< 30 minutes)

2. **No Transcript Available**
   ```json
   {
     "error": {
       "code": "E2003",
       "message": "No transcript is available for this video"
     }
   }
   ```
   
   **Solution**: Use videos with auto-generated or manual captions

3. **Live Stream Detected**
   ```json
   {
     "error": {
       "code": "E2004",
       "message": "Live streams are not supported"
     }
   }
   ```
   
   **Solution**: Use recorded (non-live) videos

### Issue: 500 Internal Server Error

#### Diagnosis Steps

1. **Check Application Logs**
   ```bash
   # Look for error traces
   docker-compose logs app | grep -A 10 -B 10 "500\|ERROR\|Exception"
   ```

2. **Test External Dependencies**
   ```bash
   # Test YouTube API accessibility
   curl -s "https://www.youtube.com/watch?v=dQw4w9WgXcQ" > /dev/null
   echo $?  # Should return 0
   
   # Test LLM API connectivity
   # (Use appropriate API test based on your configuration)
   ```

3. **Check Resource Availability**
   ```bash
   # Monitor system resources during request
   docker stats --no-stream
   
   # Check memory usage
   docker-compose exec app cat /proc/meminfo
   
   # Check disk space
   df -h
   ```

## Performance Issues

### Issue: Slow Response Times

#### Symptoms
- Requests taking longer than expected
- Timeout errors
- High CPU/memory usage

#### Diagnosis

1. **Measure Response Times**
   ```bash
   # Time a request
   time curl -X POST "http://localhost:8000/api/v1/summarize" \
     -H "Content-Type: application/json" \
     -d '{"youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
   
   # Use curl timing
   curl -w "@curl-format.txt" -X POST "http://localhost:8000/api/v1/summarize" \
     -H "Content-Type: application/json" \
     -d '{"youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
   ```

2. **Monitor System Resources**
   ```bash
   # Real-time monitoring
   docker stats
   
   # CPU usage over time
   docker-compose exec app top
   
   # Memory usage details
   docker-compose exec app cat /proc/meminfo | head -5
   ```

3. **Profile Application Performance**
   ```bash
   # Run performance tests
   python scripts/run_performance_tests.py --tests benchmarks
   
   # Check specific bottlenecks
   docker-compose exec app python -m cProfile -s cumulative src/app.py
   ```

#### Common Solutions

1. **Optimize Configuration**
   ```bash
   # Increase worker processes
   echo "WORKERS=4" >> .env
   
   # Adjust timeouts
   echo "LLM_TIMEOUT=120" >> .env
   echo "YOUTUBE_API_TIMEOUT=45" >> .env
   ```

2. **Resource Scaling**
   ```yaml
   # In docker-compose.yml
   services:
     app:
       deploy:
         resources:
           limits:
             cpus: '2.0'
             memory: 4G
   ```

3. **Enable Caching**
   ```bash
   # Add Redis caching
   echo "REDIS_URL=redis://redis:6379" >> .env
   docker-compose up -d redis
   ```

### Issue: Memory Leaks

#### Detection
```bash
# Monitor memory over time
watch -n 10 'docker stats --no-stream | grep app'

# Check for memory growth
docker-compose exec app python -c "
import psutil
import time
process = psutil.Process()
for i in range(10):
    print(f'Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB')
    time.sleep(30)
"
```

#### Solutions
```bash
# 1. Restart application regularly
docker-compose restart app

# 2. Run memory leak tests
python -m pytest tests/test_performance_benchmarks.py::TestPerformanceBenchmarks::test_memory_leak_detection -v

# 3. Profile memory usage
docker-compose exec app python -m memory_profiler src/app.py
```

## Docker and Container Issues

### Issue: Container Won't Start

#### Diagnosis
```bash
# Check container status
docker-compose ps

# View detailed container info
docker inspect youtube-summarizer-wave_app_1

# Check for port conflicts
docker port youtube-summarizer-wave_app_1

# Examine Docker logs
docker logs youtube-summarizer-wave_app_1
```

#### Common Solutions

1. **Image Build Issues**
   ```bash
   # Clean rebuild
   docker-compose build --no-cache
   docker-compose up -d
   
   # Check for base image issues
   docker pull python:3.11-slim
   ```

2. **Volume Mount Problems**
   ```bash
   # Check volume permissions
   ls -la ./logs
   
   # Fix permissions
   sudo chown -R $USER:$USER ./logs
   chmod 755 ./logs
   ```

3. **Network Issues**
   ```bash
   # Check Docker networks
   docker network ls
   
   # Recreate network
   docker-compose down
   docker network prune
   docker-compose up -d
   ```

### Issue: Container Exits Unexpectedly

#### Investigation
```bash
# Check exit code
docker-compose ps
docker inspect --format='{{.State.ExitCode}}' container_name

# Examine logs for crash reasons
docker-compose logs app --tail=100

# Check system resources
docker system df
docker system events --since="1h"
```

#### Solutions

1. **OOM (Out of Memory) Kills**
   ```bash
   # Check dmesg for OOM killer
   dmesg | grep -i "killed process"
   
   # Increase memory limits
   # In docker-compose.yml:
   services:
     app:
       mem_limit: 4g
   ```

2. **Health Check Failures**
   ```bash
   # Test health check manually
   docker-compose exec app curl -f http://localhost:8000/health
   
   # Adjust health check timing
   # In docker-compose.yml:
   healthcheck:
     interval: 30s
     timeout: 10s
     retries: 5
     start_period: 60s
   ```

## Network and Connectivity Issues

### Issue: Cannot Connect to API

#### Diagnosis
```bash
# Check if port is open
netstat -tlnp | grep :8000
ss -tlnp | grep :8000

# Test local connectivity
curl -v http://localhost:8000/health

# Test external connectivity
curl -v http://YOUR_SERVER_IP:8000/health

# Check firewall rules
sudo ufw status
iptables -L
```

#### Solutions

1. **Firewall Configuration**
   ```bash
   # Open port 8000
   sudo ufw allow 8000
   
   # For production, use reverse proxy
   sudo ufw allow 80
   sudo ufw allow 443
   ```

2. **Docker Network Issues**
   ```bash
   # Check Docker network
   docker network inspect youtube-summarizer-wave_default
   
   # Recreate network
   docker-compose down
   docker network prune
   docker-compose up -d
   ```

3. **Binding Issues**
   ```bash
   # Ensure app binds to all interfaces
   echo "HOST=0.0.0.0" >> .env
   
   # Check binding in logs
   docker-compose logs app | grep "Uvicorn running"
   ```

### Issue: Slow Network Responses

#### Diagnosis
```bash
# Test network latency
ping google.com
ping api.openai.com

# Test DNS resolution
nslookup api.openai.com
dig api.openai.com

# Monitor network traffic
docker-compose exec app netstat -i
```

#### Solutions
```bash
# 1. Use faster DNS servers
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf

# 2. Increase network timeouts
echo "YOUTUBE_API_TIMEOUT=60" >> .env
echo "LLM_TIMEOUT=120" >> .env

# 3. Check for proxy issues
unset http_proxy https_proxy
```

## External Service Integration Issues

### Issue: YouTube API Failures

#### Common Error Patterns

1. **Transcript Extraction Failures**
   ```bash
   # Test transcript availability
   docker-compose exec app python -c "
   from youtube_transcript_api import YouTubeTranscriptApi
   try:
       transcript = YouTubeTranscriptApi.get_transcript('dQw4w9WgXcQ')
       print('Transcript available')
   except Exception as e:
       print(f'Error: {e}')
   "
   ```

2. **Video Metadata Issues**
   ```bash
   # Test video metadata extraction
   docker-compose exec app python -c "
   from pytube import YouTube
   try:
       yt = YouTube('https://www.youtube.com/watch?v=dQw4w9WgXcQ')
       print(f'Title: {yt.title}')
       print(f'Duration: {yt.length}')
   except Exception as e:
       print(f'Error: {e}')
   "
   ```

#### Solutions

1. **Update YouTube Libraries**
   ```bash
   # Update dependencies
   pip install --upgrade youtube-transcript-api pytube
   
   # Rebuild container
   docker-compose build --no-cache app
   ```

2. **Handle Rate Limiting**
   ```bash
   # Add retry logic and delays
   echo "RETRY_ATTEMPTS=5" >> .env
   echo "RETRY_DELAY=2" >> .env
   ```

### Issue: LLM API Integration Problems

#### OpenAI API Issues

1. **Authentication Errors**
   ```bash
   # Test API key
   curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models
   
   # Check key format (should start with sk-)
   echo $OPENAI_API_KEY | cut -c1-3
   ```

2. **Rate Limiting**
   ```json
   {
     "error": {
       "code": "rate_limit_exceeded",
       "message": "Rate limit reached"
     }
   }
   ```
   
   **Solution**: Implement exponential backoff
   ```bash
   echo "LLM_TIMEOUT=180" >> .env
   echo "RETRY_ATTEMPTS=3" >> .env
   ```

3. **Model Availability**
   ```bash
   # Check available models
   curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models | jq '.data[].id'
   
   # Update model if needed
   echo "OPENAI_MODEL=gpt-4-turbo-preview" >> .env
   ```

#### Anthropic API Issues

1. **Authentication Errors**
   ```bash
   # Test Anthropic API key
   curl -H "x-api-key: $ANTHROPIC_API_KEY" \
     -H "anthropic-version: 2023-06-01" \
     https://api.anthropic.com/v1/messages \
     -X POST -d '{
       "model": "claude-3-sonnet-20240229",
       "max_tokens": 10,
       "messages": [{"role": "user", "content": "Hi"}]
     }'
   ```

2. **Model Configuration**
   ```bash
   # Verify model name
   echo "ANTHROPIC_MODEL=claude-3-sonnet-20240229" >> .env
   ```

## Security and Authentication Issues

### Issue: Unauthorized Access

#### Symptoms
- 401 Unauthorized responses
- API key validation failures
- CORS errors

#### Solutions

1. **API Key Configuration**
   ```bash
   # Verify API keys are set
   docker-compose exec app env | grep API_KEY
   
   # Test API key validity
   # (Use appropriate test for your LLM provider)
   ```

2. **CORS Configuration**
   ```bash
   # Update CORS settings
   echo 'CORS_ORIGINS=["https://yourdomain.com", "http://localhost:3000"]' >> .env
   
   # For development only
   echo 'CORS_ORIGINS=["*"]' >> .env
   ```

### Issue: SSL/TLS Certificate Problems

#### Diagnosis
```bash
# Test SSL certificate
curl -I https://yourdomain.com

# Check certificate details
openssl s_client -connect yourdomain.com:443 -servername yourdomain.com

# Verify certificate chain
curl -v https://yourdomain.com 2>&1 | grep -A 5 -B 5 certificate
```

#### Solutions

1. **Certificate Renewal**
   ```bash
   # Renew Let's Encrypt certificate
   sudo certbot renew
   
   # Test renewal
   sudo certbot renew --dry-run
   ```

2. **Certificate Configuration**
   ```bash
   # Check Nginx SSL configuration
   sudo nginx -t
   
   # Reload Nginx
   sudo systemctl reload nginx
   ```

## Database and Caching Issues

### Issue: Redis Connection Problems

#### Symptoms
- Cache misses
- Connection timeout errors
- Performance degradation

#### Diagnosis
```bash
# Test Redis connectivity
docker-compose exec redis redis-cli ping

# Check Redis logs
docker-compose logs redis

# Monitor Redis
docker-compose exec redis redis-cli monitor
```

#### Solutions

1. **Redis Configuration**
   ```bash
   # Check Redis configuration
   docker-compose exec redis redis-cli CONFIG GET "*"
   
   # Adjust memory settings
   docker-compose exec redis redis-cli CONFIG SET maxmemory 2gb
   ```

2. **Connection String**
   ```bash
   # Verify Redis URL
   echo "REDIS_URL=redis://redis:6379/0" >> .env
   
   # Test connection from app
   docker-compose exec app python -c "
   import redis
   r = redis.from_url('redis://redis:6379')
   print(r.ping())
   "
   ```

## Monitoring and Logging Issues

### Issue: Missing or Incomplete Logs

#### Solutions

1. **Log Level Configuration**
   ```bash
   # Increase log verbosity
   echo "LOG_LEVEL=debug" >> .env
   
   # Restart to apply changes
   docker-compose restart app
   ```

2. **Log Collection Setup**
   ```bash
   # Ensure log directory exists
   mkdir -p ./logs
   chmod 755 ./logs
   
   # Check log volume mount
   docker-compose exec app ls -la /app/logs
   ```

3. **Structured Logging**
   ```python
   # Verify log format in application
   import logging
   import json
   
   logger = logging.getLogger(__name__)
   logger.info(json.dumps({
       "message": "Test log entry",
       "level": "info",
       "timestamp": "2023-01-01T10:00:00Z"
   }))
   ```

### Issue: Health Check Failures

#### Diagnosis
```bash
# Manual health check
curl -v http://localhost:8000/health

# Check health endpoint response
docker-compose exec app curl -s http://localhost:8000/health | jq .

# Verify application readiness
docker-compose exec app python -c "
import requests
try:
    response = requests.get('http://localhost:8000/health', timeout=5)
    print(f'Status: {response.status_code}')
    print(f'Response: {response.json()}')
except Exception as e:
    print(f'Error: {e}')
"
```

#### Solutions

1. **Health Check Configuration**
   ```yaml
   # In docker-compose.yml
   healthcheck:
     test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
     interval: 30s
     timeout: 10s
     retries: 3
     start_period: 60s
   ```

2. **Application Readiness**
   ```bash
   # Ensure all dependencies are ready
   # Check database connections
   # Verify external API connectivity
   ```

## Emergency Procedures

### Complete System Recovery

```bash
#!/bin/bash
# emergency-recovery.sh

echo "Starting emergency recovery procedure..."

# 1. Stop all services
docker-compose down

# 2. Clean up Docker resources
docker system prune -f
docker volume prune -f

# 3. Backup current configuration
cp .env .env.backup.$(date +%Y%m%d_%H%M%S)

# 4. Reset to known good state
git checkout HEAD -- docker-compose.yml
git pull origin main

# 5. Rebuild from scratch
docker-compose build --no-cache

# 6. Start services
docker-compose up -d

# 7. Wait for startup
sleep 30

# 8. Verify health
curl -f http://localhost:8000/health

echo "Recovery procedure completed."
```

### Performance Emergency Response

```bash
#!/bin/bash
# performance-emergency.sh

echo "Initiating performance emergency response..."

# 1. Scale down to conserve resources
docker-compose up -d --scale app=1

# 2. Clear caches
docker-compose exec redis redis-cli FLUSHALL

# 3. Restart application
docker-compose restart app

# 4. Monitor resources
docker stats --no-stream

# 5. Check for memory leaks
docker-compose exec app ps aux | grep python

echo "Emergency response completed."
```

### Data Recovery Procedures

```bash
#!/bin/bash
# data-recovery.sh

BACKUP_DATE=${1:-$(date +%Y%m%d)}
BACKUP_DIR="/backups"

echo "Starting data recovery for date: $BACKUP_DATE"

# 1. Stop application
docker-compose stop app

# 2. Restore Redis data
if [ -f "$BACKUP_DIR/redis/dump_$BACKUP_DATE.rdb" ]; then
    docker-compose exec redis redis-cli SHUTDOWN NOSAVE
    docker cp "$BACKUP_DIR/redis/dump_$BACKUP_DATE.rdb" redis_container:/data/dump.rdb
    docker-compose start redis
fi

# 3. Restore application data
if [ -f "$BACKUP_DIR/app/app_backup_$BACKUP_DATE.tar.gz" ]; then
    tar -xzf "$BACKUP_DIR/app/app_backup_$BACKUP_DATE.tar.gz" -C ./
fi

# 4. Restart services
docker-compose start app

# 5. Verify recovery
curl -f http://localhost:8000/health

echo "Data recovery completed."
```

This troubleshooting guide provides comprehensive solutions for the most common issues encountered when deploying and operating the YouTube Summarizer API.