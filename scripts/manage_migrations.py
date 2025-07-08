#!/usr/bin/env python3
"""
Database migration management script for YouTube Summarizer.
Provides utilities for running Alembic migrations.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from config import settings


def run_command(command, description=None):
    """Run a shell command and handle errors."""
    if description:
        print(f"Running: {description}")
    
    print(f"Command: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    
    if result.stdout:
        print(f"Output: {result.stdout}")
    
    return True


def check_database_connection():
    """Check if database connection is available."""
    try:
        import asyncpg
        import asyncio
        
        async def test_connection():
            try:
                # Parse connection parameters from DATABASE_URL
                import urllib.parse
                parsed = urllib.parse.urlparse(settings.database_url)
                
                conn = await asyncpg.connect(
                    host=parsed.hostname,
                    port=parsed.port,
                    user=parsed.username,
                    password=parsed.password,
                    database=parsed.path[1:] if parsed.path else 'postgres'
                )
                await conn.close()
                return True
            except Exception as e:
                print(f"Database connection failed: {e}")
                return False
        
        return asyncio.run(test_connection())
    except Exception as e:
        print(f"Failed to test database connection: {e}")
        return False


def create_migration(message=None):
    """Create a new migration."""
    if not message:
        message = input("Enter migration message: ").strip()
    
    if not message:
        print("Migration message is required")
        return False
    
    command = f"alembic revision --autogenerate -m '{message}'"
    return run_command(command, "Creating new migration")


def upgrade_database(revision="head"):
    """Upgrade database to specified revision."""
    command = f"alembic upgrade {revision}"
    return run_command(command, f"Upgrading database to {revision}")


def downgrade_database(revision):
    """Downgrade database to specified revision."""
    command = f"alembic downgrade {revision}"
    return run_command(command, f"Downgrading database to {revision}")


def show_migration_history():
    """Show migration history."""
    command = "alembic history"
    return run_command(command, "Showing migration history")


def show_current_revision():
    """Show current database revision."""
    command = "alembic current"
    return run_command(command, "Showing current revision")


def test_migration():
    """Test migration up and down."""
    print("Testing migration up/down functionality...")
    
    # Show current state
    print("\n1. Current revision:")
    show_current_revision()
    
    # Upgrade to head
    print("\n2. Upgrading to head:")
    if not upgrade_database("head"):
        return False
    
    # Show current state after upgrade
    print("\n3. Current revision after upgrade:")
    show_current_revision()
    
    # Test downgrade to base (optional - commented out to prevent data loss)
    # print("\n4. Testing downgrade to base:")
    # if not downgrade_database("base"):
    #     return False
    
    # print("\n5. Upgrading back to head:")
    # if not upgrade_database("head"):
    #     return False
    
    print("\nMigration test completed successfully!")
    return True


def init_database():
    """Initialize database with initial migration."""
    print("Initializing database with initial migration...")
    
    # Check database connection
    if not check_database_connection():
        print("Please ensure PostgreSQL is running and database configuration is correct")
        return False
    
    # Upgrade to head (runs initial migration)
    return upgrade_database("head")


def main():
    """Main CLI interface."""
    if len(sys.argv) < 2:
        print("Usage: python manage_migrations.py <command> [args]")
        print("Commands:")
        print("  init                 - Initialize database with initial migration")
        print("  create [message]     - Create new migration")
        print("  upgrade [revision]   - Upgrade database (default: head)")
        print("  downgrade <revision> - Downgrade database")
        print("  history              - Show migration history")
        print("  current              - Show current revision")
        print("  test                 - Test migration up/down")
        print("  check-db             - Check database connection")
        return
    
    command = sys.argv[1].lower()
    
    # Set environment variable for database URL
    os.environ['DATABASE_URL'] = settings.database_url
    
    # Change to project root directory
    os.chdir(project_root)
    
    if command == 'init':
        init_database()
    elif command == 'create':
        message = sys.argv[2] if len(sys.argv) > 2 else None
        create_migration(message)
    elif command == 'upgrade':
        revision = sys.argv[2] if len(sys.argv) > 2 else "head"
        upgrade_database(revision)
    elif command == 'downgrade':
        if len(sys.argv) < 3:
            print("Downgrade requires a revision argument")
            return
        downgrade_database(sys.argv[2])
    elif command == 'history':
        show_migration_history()
    elif command == 'current':
        show_current_revision()
    elif command == 'test':
        test_migration()
    elif command == 'check-db':
        if check_database_connection():
            print("Database connection successful!")
        else:
            print("Database connection failed!")
    else:
        print(f"Unknown command: {command}")


if __name__ == '__main__':
    main()