#!/usr/bin/env python3
"""
Test runner script for status tracking system tests.

This script runs all status tracking related tests with proper configuration
and generates comprehensive coverage reports.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(command, description):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=False)
        print(f"\n✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description='Run status tracking system tests')
    parser.add_argument('--coverage', action='store_true', help='Run with coverage reporting')
    parser.add_argument('--performance', action='store_true', help='Include performance tests')
    parser.add_argument('--integration', action='store_true', help='Include integration tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--fail-fast', '-x', action='store_true', help='Stop on first failure')
    parser.add_argument('--parallel', '-n', type=int, help='Number of parallel workers')
    parser.add_argument('--markers', '-m', help='Run tests with specific markers')
    parser.add_argument('--output-dir', help='Output directory for reports', default='test_reports')
    
    args = parser.parse_args()
    
    # Set up test environment
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Base pytest command
    pytest_cmd = ['python', '-m', 'pytest']
    
    # Add verbose flag
    if args.verbose:
        pytest_cmd.append('-v')
    
    # Add fail-fast flag
    if args.fail_fast:
        pytest_cmd.append('-x')
    
    # Add parallel execution
    if args.parallel:
        pytest_cmd.extend(['-n', str(args.parallel)])
    
    # Add markers
    if args.markers:
        pytest_cmd.extend(['-m', args.markers])
    
    # Define test modules
    test_modules = [
        'tests/test_status_tracking_comprehensive.py',
        'tests/test_status_api_endpoints.py',
        'src/services/status_service.test.py',
        'src/services/status_updater.test.py',
        'src/services/status_integration.test.py',
        'src/services/status_filtering.test.py',
        'src/services/status_events.test.py',
        'src/api/status.test.py'
    ]
    
    # Add performance tests if requested
    if args.performance:
        test_modules.append('tests/test_status_performance.py')
    
    # Filter existing test files
    existing_tests = []
    for test_file in test_modules:
        if Path(test_file).exists():
            existing_tests.append(test_file)
        else:
            print(f"⚠️  Test file not found: {test_file}")
    
    if not existing_tests:
        print("❌ No test files found!")
        return 1
    
    success_count = 0
    total_tests = 0
    
    # Run unit tests
    print("\n🧪 Running Status Tracking Unit Tests")
    print("=" * 60)
    
    unit_test_cmd = pytest_cmd + existing_tests
    
    if args.coverage:
        unit_test_cmd.extend([
            '--cov=src.services',
            '--cov=src.api',
            '--cov=src.database.status_models',
            f'--cov-report=html:{output_dir}/coverage_html',
            f'--cov-report=xml:{output_dir}/coverage.xml',
            '--cov-report=term-missing',
            '--cov-fail-under=80'
        ])
    
    # Add JUnit XML output
    unit_test_cmd.extend(['--junit-xml', f'{output_dir}/junit_unit.xml'])
    
    if run_command(' '.join(unit_test_cmd), "Unit Tests"):
        success_count += 1
    total_tests += 1
    
    # Run integration tests if requested
    if args.integration:
        print("\n🔗 Running Integration Tests")
        integration_cmd = pytest_cmd + [
            'tests/test_status_tracking_comprehensive.py::TestStatusTrackingIntegration',
            '--junit-xml', f'{output_dir}/junit_integration.xml'
        ]
        
        if run_command(' '.join(integration_cmd), "Integration Tests"):
            success_count += 1
        total_tests += 1
    
    # Run performance tests if requested
    if args.performance:
        print("\n⚡ Running Performance Tests")
        perf_cmd = pytest_cmd + [
            'tests/test_status_performance.py',
            '-s',  # Don't capture output for performance metrics
            '--junit-xml', f'{output_dir}/junit_performance.xml'
        ]
        
        if run_command(' '.join(perf_cmd), "Performance Tests"):
            success_count += 1
        total_tests += 1
    
    # Run specific component tests
    component_tests = [
        ('Status Service', ['src/services/status_service.test.py']),
        ('Status Events', ['src/services/status_events.test.py']),
        ('Status Filtering', ['src/services/status_filtering.test.py']),
        ('Status Integration', ['src/services/status_integration.test.py']),
        ('Status API', ['src/api/status.test.py'])
    ]
    
    for component_name, test_files in component_tests:
        existing_component_tests = [f for f in test_files if Path(f).exists()]
        if existing_component_tests:
            print(f"\n🔧 Running {component_name} Tests")
            component_cmd = pytest_cmd + existing_component_tests + [
                '--junit-xml', f'{output_dir}/junit_{component_name.lower().replace(" ", "_")}.xml'
            ]
            
            if run_command(' '.join(component_cmd), f"{component_name} Tests"):
                success_count += 1
            total_tests += 1
    
    # Generate final report
    print("\n" + "="*60)
    print("📊 TEST EXECUTION SUMMARY")
    print("="*60)
    print(f"✅ Passed: {success_count}/{total_tests}")
    print(f"❌ Failed: {total_tests - success_count}/{total_tests}")
    
    if args.coverage and success_count > 0:
        print(f"\n📈 Coverage report generated:")
        print(f"   HTML: {output_dir}/coverage_html/index.html")
        print(f"   XML:  {output_dir}/coverage.xml")
    
    print(f"\n📋 Test reports generated in: {output_dir}/")
    
    # Additional validation commands
    if success_count == total_tests:
        print("\n🎉 All tests passed! Running additional validation...")
        
        # Check test coverage
        if args.coverage:
            coverage_cmd = [
                'python', '-m', 'coverage', 'report',
                '--show-missing',
                '--skip-covered'
            ]
            run_command(' '.join(coverage_cmd), "Coverage Report")
        
        # Run linting on test files
        if Path('pyproject.toml').exists() or Path('setup.cfg').exists():
            lint_cmd = ['python', '-m', 'flake8'] + existing_tests
            run_command(' '.join(lint_cmd), "Test Linting")
        
        # Check for test file naming conventions
        print("\n📝 Checking test file conventions...")
        for test_file in existing_tests:
            if not Path(test_file).name.startswith('test_'):
                print(f"⚠️  Test file doesn't follow naming convention: {test_file}")
        
        print("\n🏆 All validation checks completed!")
        return 0
    else:
        print(f"\n💥 {total_tests - success_count} test suite(s) failed!")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)