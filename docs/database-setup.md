# Database Setup and Configuration Guide

This guide covers the setup, configuration, and management of the PostgreSQL database for the YouTube Summarizer service.

## Database Schema Overview

The YouTube Summarizer uses PostgreSQL as its primary database with the following tables:

### Core Tables

#### 1. Videos Table
Stores core video information and metadata.

```sql
CREATE TABLE videos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id VARCHAR(50) UNIQUE NOT NULL,
    title TEXT NOT NULL,
    duration VARCHAR(20),
    url TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

#### 2. Transcripts Table
Stores video transcripts with language information.

```sql
CREATE TABLE transcripts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID REFERENCES videos(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    language VARCHAR(10) DEFAULT 'en',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

#### 3. Summaries Table
Stores AI-generated video summaries.

```sql
CREATE TABLE summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID REFERENCES videos(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    processing_time FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

#### 4. Keywords Table
Stores extracted keywords in JSON format.

```sql
CREATE TABLE keywords (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID REFERENCES videos(id) ON DELETE CASCADE,
    keywords_json JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

#### 5. Timestamped Segments Table
Stores timestamped video segments with importance ratings.

```sql
CREATE TABLE timestamped_segments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID REFERENCES videos(id) ON DELETE CASCADE,
    segments_json JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

#### 6. Processing Metadata Table
Stores workflow execution metadata and error information.

```sql
CREATE TABLE processing_metadata (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID REFERENCES videos(id) ON DELETE CASCADE,
    workflow_params JSONB,
    status VARCHAR(20) DEFAULT 'pending',
    error_info JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

### Indexes

Performance-optimized indexes are created for common query patterns:

```sql
-- Video lookup indexes
CREATE INDEX idx_videos_video_id ON videos(video_id);
CREATE INDEX idx_videos_created_at ON videos(created_at DESC);

-- Foreign key indexes
CREATE INDEX idx_transcripts_video_id ON transcripts(video_id);
CREATE INDEX idx_summaries_video_id ON summaries(video_id);
CREATE INDEX idx_keywords_video_id ON keywords(video_id);
CREATE INDEX idx_timestamped_segments_video_id ON timestamped_segments(video_id);
CREATE INDEX idx_processing_metadata_video_id ON processing_metadata(video_id);

-- Status and date indexes
CREATE INDEX idx_processing_metadata_status ON processing_metadata(status);
CREATE INDEX idx_processing_metadata_created_at ON processing_metadata(created_at DESC);

-- JSONB indexes for keyword search
CREATE INDEX idx_keywords_gin ON keywords USING gin(keywords_json);
CREATE INDEX idx_timestamped_segments_gin ON timestamped_segments USING gin(segments_json);
```

## Setup Instructions

### 1. Using Docker Compose (Recommended)

The easiest way to set up the database is using Docker Compose:

```bash
# Start PostgreSQL service
docker-compose up -d postgres

# Wait for database to be ready
docker-compose logs postgres

# Run migrations
alembic upgrade head
```

### 2. Manual PostgreSQL Installation

#### On macOS (using Homebrew)
```bash
# Install PostgreSQL
brew install postgresql@15

# Start PostgreSQL service
brew services start postgresql@15

# Create database
createdb youtube_summarizer

# Create user (optional)
psql postgres -c "CREATE USER youtube_user WITH PASSWORD 'your_password';"
psql postgres -c "GRANT ALL PRIVILEGES ON DATABASE youtube_summarizer TO youtube_user;"
```

#### On Ubuntu/Debian
```bash
# Install PostgreSQL
sudo apt update
sudo apt install postgresql-15 postgresql-contrib

# Start PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user
sudo -u postgres psql -c "CREATE DATABASE youtube_summarizer;"
sudo -u postgres psql -c "CREATE USER youtube_user WITH PASSWORD 'your_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE youtube_summarizer TO youtube_user;"
```

#### On Windows
1. Download and install PostgreSQL from [postgresql.org](https://www.postgresql.org/download/windows/)
2. Use pgAdmin or command line to create database:
   ```sql
   CREATE DATABASE youtube_summarizer;
   CREATE USER youtube_user WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE youtube_summarizer TO youtube_user;
   ```

### 3. Environment Configuration

Create or update your `.env` file:

```bash
# Database Configuration
DATABASE_URL=postgresql+asyncpg://youtube_user:your_password@localhost:5432/youtube_summarizer
POSTGRES_DB=youtube_summarizer
POSTGRES_USER=youtube_user
POSTGRES_PASSWORD=your_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
```

### 4. Run Database Migrations

```bash
# Initialize Alembic (only needed once)
alembic init alembic

# Create initial migration
alembic revision --autogenerate -m "Initial migration"

# Apply migrations
alembic upgrade head

# Verify migration
alembic current
```

## Migration Management

### Creating New Migrations

When you modify the database models, create a new migration:

```bash
# Auto-generate migration from model changes
alembic revision --autogenerate -m "Description of changes"

# Review the generated migration file
cat alembic/versions/[migration_file].py

# Apply the migration
alembic upgrade head
```

### Managing Migrations

```bash
# View migration history
alembic history

# Check current migration version
alembic current

# Upgrade to specific revision
alembic upgrade [revision_id]

# Downgrade to previous revision
alembic downgrade -1

# Downgrade to specific revision
alembic downgrade [revision_id]
```

### Migration Scripts

Use the management script for common operations:

```bash
# Create new migration
python scripts/manage_migrations.py create "Migration description"

# Apply migrations
python scripts/manage_migrations.py upgrade

# Check migration status
python scripts/manage_migrations.py status

# Rollback last migration
python scripts/manage_migrations.py downgrade
```

## Database Maintenance

### Regular Maintenance Tasks

#### 1. Cleanup Old Records

```bash
# Using the maintenance script
python scripts/database_maintenance.py cleanup --days 90

# Or using the service directly
python -c "
from src.database.maintenance import cleanup_old_records
import asyncio
asyncio.run(cleanup_old_records(days=90))
"
```

#### 2. Database Statistics

```bash
# Get database statistics
python scripts/database_maintenance.py stats

# Get table sizes
python scripts/database_maintenance.py table-sizes
```

#### 3. Health Checks

```bash
# Check database health
python scripts/database_maintenance.py health

# Check connection pool status
curl http://localhost:8000/health/database
```

### Performance Optimization

#### 1. Connection Pool Tuning

Adjust connection pool settings in `src/database/connection.py`:

```python
# For high-traffic environments
engine = create_async_engine(
    database_url,
    pool_size=30,           # Increase base connections
    max_overflow=50,        # Increase overflow
    pool_timeout=30,        # Connection timeout
    pool_recycle=3600,      # Recycle connections hourly
    echo=False              # Disable SQL logging in production
)
```

#### 2. Query Optimization

- Use eager loading for related data:
  ```python
  video = await session.get(
      Video, 
      video_id, 
      options=[
          selectinload(Video.transcripts),
          selectinload(Video.summaries),
          selectinload(Video.keywords)
      ]
  )
  ```

- Use indexes for common queries:
  ```sql
  -- Add custom indexes for your query patterns
  CREATE INDEX idx_videos_title_search ON videos USING gin(to_tsvector('english', title));
  ```

#### 3. Monitoring Queries

Enable slow query logging in PostgreSQL:

```sql
-- In postgresql.conf
log_min_duration_statement = 1000  -- Log queries > 1 second
log_statement = 'all'              -- Log all statements (dev only)
```

## Backup and Recovery

### Automated Backups

```bash
# Create backup script
cat > backup_db.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/youtube_summarizer_$DATE.sql"

mkdir -p $BACKUP_DIR
pg_dump -h localhost -U youtube_user -d youtube_summarizer > $BACKUP_FILE
gzip $BACKUP_FILE

# Keep only last 7 days
find $BACKUP_DIR -name "*.gz" -mtime +7 -delete
EOF

chmod +x backup_db.sh

# Add to crontab for daily backups
echo "0 2 * * * /path/to/backup_db.sh" | crontab -
```

### Restore from Backup

```bash
# Restore from backup
gunzip -c youtube_summarizer_20240708_020000.sql.gz | psql -h localhost -U youtube_user -d youtube_summarizer

# Or restore specific tables
pg_restore -h localhost -U youtube_user -d youtube_summarizer -t videos backup_file.sql
```

## Troubleshooting

### Common Issues

#### 1. Connection Refused
```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql

# Check port availability
netstat -ln | grep 5432

# Test connection
psql -h localhost -U youtube_user -d youtube_summarizer -c "SELECT 1;"
```

#### 2. Migration Errors
```bash
# Check migration status
alembic current

# Reset to specific version
alembic downgrade [revision_id]
alembic upgrade head

# Manual fix and create new migration
alembic revision -m "Fix migration issue"
```

#### 3. Performance Issues
```bash
# Check active connections
psql -U youtube_user -d youtube_summarizer -c "
  SELECT count(*) as active_connections 
  FROM pg_stat_activity 
  WHERE state = 'active';
"

# Check slow queries
psql -U youtube_user -d youtube_summarizer -c "
  SELECT query, mean_exec_time, calls 
  FROM pg_stat_statements 
  ORDER BY mean_exec_time DESC 
  LIMIT 10;
"
```

#### 4. Disk Space Issues
```bash
# Check database size
psql -U youtube_user -d youtube_summarizer -c "
  SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
  FROM pg_tables 
  ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
"

# Vacuum and analyze
psql -U youtube_user -d youtube_summarizer -c "VACUUM ANALYZE;"
```

## Security Considerations

### 1. Database User Permissions

```sql
-- Create read-only user for monitoring
CREATE USER youtube_readonly WITH PASSWORD 'readonly_password';
GRANT CONNECT ON DATABASE youtube_summarizer TO youtube_readonly;
GRANT USAGE ON SCHEMA public TO youtube_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO youtube_readonly;

-- Create backup user
CREATE USER youtube_backup WITH PASSWORD 'backup_password';
GRANT CONNECT ON DATABASE youtube_summarizer TO youtube_backup;
GRANT USAGE ON SCHEMA public TO youtube_backup;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO youtube_backup;
```

### 2. SSL Configuration

```bash
# In postgresql.conf
ssl = on
ssl_cert_file = 'server.crt'
ssl_key_file = 'server.key'
ssl_ca_file = 'ca.crt'

# In pg_hba.conf
hostssl all all 0.0.0.0/0 md5
```

### 3. Connection Security

```python
# Use SSL in production
DATABASE_URL = "postgresql+asyncpg://user:pass@host:5432/db?ssl=require"

# Or with certificate verification
DATABASE_URL = "postgresql+asyncpg://user:pass@host:5432/db?ssl=require&sslmode=verify-full"
```

## Production Deployment

### Docker Deployment

```yaml
# docker-compose.production.yml
version: '3.8'
services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: youtube_summarizer
      POSTGRES_USER: youtube_user
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    secrets:
      - postgres_password
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'

volumes:
  postgres_data:

secrets:
  postgres_password:
    external: true
```

### Kubernetes Deployment

```yaml
# postgres-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: POSTGRES_DB
          value: youtube_summarizer
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: username
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
```

This comprehensive guide covers all aspects of database setup, configuration, and maintenance for the YouTube Summarizer service. Follow these instructions to ensure a robust and scalable database deployment.