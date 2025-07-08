#!/usr/bin/env python3
"""
Database maintenance CLI commands.

Provides command-line interface for database maintenance operations
including cleanup, health checks, and optimization tasks.
"""

import asyncio
import argparse
import logging
import json
from datetime import datetime
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from database.maintenance import (
    DatabaseMaintenance, cleanup_old_records, get_database_health_detailed, 
    run_maintenance_tasks
)
from database.monitor import db_monitor


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def print_json_result(result: dict, title: str = None):
    """Print result as formatted JSON."""
    if title:
        print(f"\n=== {title} ===")
    print(json.dumps(result, indent=2, default=str))


async def cmd_cleanup(args):
    """Run database cleanup command."""
    print(f"Running database cleanup (keep {args.days} days, dry_run={args.dry_run})...")
    
    result = await cleanup_old_records(
        days_to_keep=args.days,
        dry_run=args.dry_run
    )
    
    print_json_result(result, "Cleanup Results")
    
    if not args.dry_run and result.get('total_deleted', 0) > 0:
        print(f"\n✅ Successfully deleted {result['total_deleted']} old records")
    elif args.dry_run:
        print(f"\n📊 Dry run: would delete {result.get('total_deleted', 0)} records")
    else:
        print("\n✅ No old records found to delete")


async def cmd_health(args):
    """Run database health check command."""
    print("Running database health check...")
    
    result = await get_database_health_detailed()
    
    print_json_result(result, "Database Health Check")
    
    # Summary
    basic_health = result.get('basic_health', {})
    stats = result.get('statistics', {})
    monitoring = result.get('monitoring', {})
    
    print("\n=== Health Summary ===")
    print(f"Database Status: {basic_health.get('status', 'unknown')}")
    print(f"Response Time: {basic_health.get('response_time_ms', 'N/A')} ms")
    print(f"Total Records: {stats.get('total_records', 'N/A')}")
    print(f"Success Rate: {monitoring.get('metrics', {}).get('success_rate', 'N/A'):.1f}%")
    print(f"Error Rate: {monitoring.get('metrics', {}).get('error_rate', 'N/A'):.1f}%")


async def cmd_stats(args):
    """Get database statistics command."""
    print("Getting database statistics...")
    
    maintenance = DatabaseMaintenance()
    result = await maintenance.get_database_statistics()
    
    print_json_result(result, "Database Statistics")
    
    # Summary table
    if 'table_counts' in result:
        print("\n=== Table Summary ===")
        print(f"{'Table':<25} {'Records':<10} {'Oldest':<12} {'Newest':<12}")
        print("-" * 65)
        
        for table, count in result['table_counts'].items():
            oldest = result.get('oldest_records', {}).get(table, 'N/A')
            newest = result.get('newest_records', {}).get(table, 'N/A')
            
            # Format dates
            if oldest and oldest != 'N/A':
                oldest = oldest[:10]  # Just the date part
            if newest and newest != 'N/A':
                newest = newest[:10]  # Just the date part
                
            print(f"{table:<25} {count:<10} {oldest:<12} {newest:<12}")


async def cmd_vacuum(args):
    """Run vacuum analyze command."""
    print("Running VACUUM ANALYZE on database tables...")
    
    maintenance = DatabaseMaintenance()
    result = await maintenance.vacuum_analyze_tables()
    
    print_json_result(result, "Vacuum Results")
    
    if result.get('tables_processed'):
        print(f"\n✅ VACUUM ANALYZE completed on {len(result['tables_processed'])} tables")
    
    if result.get('errors'):
        print(f"\n❌ {len(result['errors'])} errors occurred")


async def cmd_integrity(args):
    """Run data integrity check command."""
    print("Running data integrity check...")
    
    maintenance = DatabaseMaintenance()
    result = await maintenance.check_data_integrity()
    
    print_json_result(result, "Integrity Check Results")
    
    total_issues = result.get('total_issues', 0)
    if total_issues == 0:
        print("\n✅ No data integrity issues found")
    else:
        print(f"\n⚠️  Found {total_issues} data integrity issues:")
        for issue in result.get('issues_found', []):
            print(f"  - {issue}")


async def cmd_full_maintenance(args):
    """Run full maintenance suite command."""
    print("Running full database maintenance...")
    
    result = await run_maintenance_tasks(
        cleanup_days=args.cleanup_days,
        run_vacuum=args.vacuum,
        check_integrity=args.integrity
    )
    
    print_json_result(result, "Full Maintenance Results")
    
    # Summary
    tasks_run = result.get('tasks_run', [])
    print(f"\n✅ Completed {len(tasks_run)} maintenance tasks: {', '.join(tasks_run)}")
    
    if result.get('errors'):
        print(f"\n❌ {len(result['errors'])} errors occurred")


async def cmd_monitor_status(args):
    """Get monitoring system status."""
    print("Getting database monitoring status...")
    
    health_status = db_monitor.get_health_status()
    recent_errors = db_monitor.get_recent_errors(limit=args.limit)
    
    print_json_result(health_status, "Monitor Health Status")
    
    if recent_errors:
        print_json_result(recent_errors, f"Recent Errors (last {len(recent_errors)})")
    
    # Summary
    print("\n=== Monitor Summary ===")
    metrics = health_status.get('metrics', {})
    print(f"Status: {health_status.get('status', 'unknown')}")
    print(f"Total Operations: {metrics.get('total_operations', 0)}")
    print(f"Success Rate: {metrics.get('success_rate', 0):.1f}%")
    print(f"Average Response Time: {metrics.get('average_response_time', 0):.3f}s")
    print(f"Active Alerts: {health_status.get('active_alerts', 0)}")


async def cmd_monitor_reset(args):
    """Reset monitoring metrics."""
    if not args.confirm:
        print("Are you sure you want to reset monitoring metrics? Use --confirm to proceed.")
        return
    
    print("Resetting database monitoring metrics...")
    db_monitor.reset_metrics()
    print("✅ Monitoring metrics reset successfully")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Database maintenance utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check database health
  python database_maintenance.py health
  
  # Cleanup records older than 30 days (dry run)
  python database_maintenance.py cleanup --days 30 --dry-run
  
  # Actually cleanup old records
  python database_maintenance.py cleanup --days 30
  
  # Get database statistics
  python database_maintenance.py stats
  
  # Run full maintenance
  python database_maintenance.py full-maintenance --cleanup-days 30
  
  # Check monitoring status
  python database_maintenance.py monitor-status
        """
    )
    
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old database records')
    cleanup_parser.add_argument('--days', type=int, default=30, help='Days to keep (default: 30)')
    cleanup_parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted without deleting')
    
    # Health command
    health_parser = subparsers.add_parser('health', help='Check database health')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Get database statistics')
    
    # Vacuum command
    vacuum_parser = subparsers.add_parser('vacuum', help='Run VACUUM ANALYZE on tables')
    
    # Integrity command
    integrity_parser = subparsers.add_parser('integrity', help='Check data integrity')
    
    # Full maintenance command
    full_parser = subparsers.add_parser('full-maintenance', help='Run full maintenance suite')
    full_parser.add_argument('--cleanup-days', type=int, default=30, help='Days to keep for cleanup')
    full_parser.add_argument('--no-vacuum', dest='vacuum', action='store_false', help='Skip vacuum analyze')
    full_parser.add_argument('--no-integrity', dest='integrity', action='store_false', help='Skip integrity check')
    
    # Monitor status command
    monitor_parser = subparsers.add_parser('monitor-status', help='Get monitoring system status')
    monitor_parser.add_argument('--limit', type=int, default=10, help='Number of recent errors to show')
    
    # Monitor reset command
    reset_parser = subparsers.add_parser('monitor-reset', help='Reset monitoring metrics')
    reset_parser.add_argument('--confirm', action='store_true', help='Confirm the reset operation')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    setup_logging(args.verbose)
    
    # Command dispatch
    commands = {
        'cleanup': cmd_cleanup,
        'health': cmd_health,
        'stats': cmd_stats,
        'vacuum': cmd_vacuum,
        'integrity': cmd_integrity,
        'full-maintenance': cmd_full_maintenance,
        'monitor-status': cmd_monitor_status,
        'monitor-reset': cmd_monitor_reset,
    }
    
    command_func = commands.get(args.command)
    if command_func:
        try:
            if asyncio.iscoroutinefunction(command_func):
                asyncio.run(command_func(args))
            else:
                command_func(args)
        except KeyboardInterrupt:
            print("\n⚠️  Operation cancelled by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()


if __name__ == "__main__":
    main()