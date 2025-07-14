#!/usr/bin/env python3
"""
Comprehensive test runner for batch processing components.

This script provides automated testing and coverage analysis for all batch processing
components including unit tests, integration tests, performance tests, and stress tests.
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any


class BatchTestRunner:
    """Comprehensive test runner for batch processing components."""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent.parent
        self.test_dir = self.base_dir / "tests"
        self.src_dir = self.base_dir / "src"
        self.reports_dir = self.base_dir / "test_reports"
        
        # Ensure reports directory exists
        self.reports_dir.mkdir(exist_ok=True)
        
        # Test categories
        self.test_categories = {
            'unit': [
                'test_batch_service_comprehensive.py',
                'test_queue_service_comprehensive.py',
                'test_concurrent_batch_service_comprehensive.py',
                'test_batch_processor_comprehensive.py'
            ],
            'integration': [
                'test_batch_integration_e2e.py'
            ],
            'performance': [
                'test_batch_performance.py'
            ],
            'stress': [
                'test_batch_stress_load.py'
            ],
            'existing': [
                'test_batch_service.test.py',
                'test_queue_service.test.py',
                'test_concurrent_batch_service.test.py'
            ]
        }
    
    def discover_test_files(self, category: str = None) -> List[str]:
        """Discover test files by category or all."""
        test_files = []
        
        if category and category in self.test_categories:
            # Get specific category
            for test_file in self.test_categories[category]:
                test_path = self.test_dir / test_file
                if test_path.exists():
                    test_files.append(str(test_path))
        elif category == 'all' or category is None:
            # Get all test files
            for cat_files in self.test_categories.values():
                for test_file in cat_files:
                    test_path = self.test_dir / test_file
                    if test_path.exists():
                        test_files.append(str(test_path))
        
        return test_files
    
    def run_test_category(self, category: str, **kwargs) -> Dict[str, Any]:
        """Run tests for a specific category."""
        print(f"\n{'='*60}")
        print(f"RUNNING {category.upper()} TESTS")
        print(f"{'='*60}")
        
        test_files = self.discover_test_files(category)
        if not test_files:
            print(f"No test files found for category: {category}")
            return {'status': 'skipped', 'reason': 'no_files'}
        
        # Prepare pytest command
        pytest_args = [
            sys.executable, '-m', 'pytest',
            '--verbose',
            '--tb=short',
            '--disable-warnings'
        ]
        
        # Add coverage if requested
        if kwargs.get('coverage', False):
            pytest_args.extend([
                '--cov=src',
                '--cov-report=html',
                '--cov-report=term-missing',
                '--cov-report=xml'
            ])
        
        # Add timeout for stress tests
        if category == 'stress':
            pytest_args.extend(['--timeout=1800'])  # 30 minute timeout
        elif category == 'performance':
            pytest_args.extend(['--timeout=900'])   # 15 minute timeout
        
        # Add output file
        output_file = self.reports_dir / f"{category}_test_results.txt"
        pytest_args.extend([f'--html={self.reports_dir}/{category}_report.html', '--self-contained-html'])
        
        # Add test files
        pytest_args.extend(test_files)
        
        # Run tests
        start_time = time.time()
        
        try:
            print(f"Running command: {' '.join(pytest_args)}")
            result = subprocess.run(
                pytest_args,
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=kwargs.get('timeout', 3600)  # 1 hour default timeout
            )
            
            duration = time.time() - start_time
            
            # Save output
            with open(output_file, 'w') as f:
                f.write(f"Test Category: {category}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Duration: {duration:.2f} seconds\n")
                f.write(f"Return Code: {result.returncode}\n")
                f.write("\n" + "="*50 + "\n")
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\n" + "="*50 + "\n")
                f.write("STDERR:\n")
                f.write(result.stderr)
            
            # Parse results
            test_results = self._parse_pytest_results(result.stdout, result.stderr)
            test_results.update({
                'category': category,
                'duration': duration,
                'return_code': result.returncode,
                'status': 'passed' if result.returncode == 0 else 'failed',
                'output_file': str(output_file)
            })
            
            # Print summary
            self._print_category_summary(test_results)
            
            return test_results
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"❌ Tests timed out after {duration:.2f} seconds")
            return {
                'category': category,
                'status': 'timeout',
                'duration': duration,
                'return_code': -1
            }
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ Error running tests: {e}")
            return {
                'category': category,
                'status': 'error',
                'duration': duration,
                'error': str(e),
                'return_code': -1
            }
    
    def _parse_pytest_results(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse pytest output to extract test results."""
        results = {
            'tests_run': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': 0,
            'warnings': 0
        }
        
        # Look for result summary
        lines = stdout.split('\n') + stderr.split('\n')
        
        for line in lines:
            # Look for final summary line
            if '==' in line and ('passed' in line or 'failed' in line or 'error' in line):
                # Parse various pytest summary formats
                parts = line.split()
                
                for i, part in enumerate(parts):
                    if part.isdigit():
                        count = int(part)
                        if i + 1 < len(parts):
                            result_type = parts[i + 1]
                            if 'passed' in result_type:
                                results['passed'] = count
                            elif 'failed' in result_type:
                                results['failed'] = count
                            elif 'skipped' in result_type:
                                results['skipped'] = count
                            elif 'error' in result_type:
                                results['errors'] = count
            
            # Look for warnings
            if 'warning' in line.lower():
                results['warnings'] += 1
        
        results['tests_run'] = results['passed'] + results['failed'] + results['skipped'] + results['errors']
        
        return results
    
    def _print_category_summary(self, results: Dict[str, Any]) -> None:
        """Print summary for test category."""
        category = results['category']
        status = results['status']
        duration = results.get('duration', 0)
        
        print(f"\n{category.upper()} TESTS SUMMARY:")
        print(f"Status: {status}")
        print(f"Duration: {duration:.2f} seconds")
        
        if 'tests_run' in results:
            print(f"Tests Run: {results['tests_run']}")
            print(f"Passed: {results['passed']}")
            print(f"Failed: {results['failed']}")
            print(f"Skipped: {results['skipped']}")
            print(f"Errors: {results['errors']}")
            
            if results['tests_run'] > 0:
                pass_rate = (results['passed'] / results['tests_run']) * 100
                print(f"Pass Rate: {pass_rate:.1f}%")
        
        if status == 'passed':
            print("✅ All tests passed!")
        elif status == 'failed':
            print("❌ Some tests failed.")
        elif status == 'timeout':
            print("⏰ Tests timed out.")
        elif status == 'error':
            print("💥 Error occurred during testing.")
    
    def run_all_tests(self, **kwargs) -> Dict[str, Any]:
        """Run all test categories."""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE BATCH PROCESSING TEST SUITE")
        print(f"{'='*80}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        overall_start_time = time.time()
        results = {
            'start_time': datetime.now().isoformat(),
            'categories': {},
            'overall_summary': {}
        }
        
        # Test categories to run (in order)
        categories_to_run = ['unit', 'integration', 'performance']
        
        if kwargs.get('include_stress', False):
            categories_to_run.append('stress')
        
        if kwargs.get('include_existing', False):
            categories_to_run.append('existing')
        
        # Run each category
        for category in categories_to_run:
            try:
                category_results = self.run_test_category(category, **kwargs)
                results['categories'][category] = category_results
            except KeyboardInterrupt:
                print(f"\n⏹️  Testing interrupted by user")
                break
            except Exception as e:
                print(f"\n💥 Error running {category} tests: {e}")
                results['categories'][category] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Calculate overall summary
        overall_duration = time.time() - overall_start_time
        results['end_time'] = datetime.now().isoformat()
        results['overall_duration'] = overall_duration
        
        # Aggregate results
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_skipped = 0
        total_errors = 0
        categories_passed = 0
        categories_run = len(results['categories'])
        
        for category, category_results in results['categories'].items():
            if category_results.get('status') == 'passed':
                categories_passed += 1
            
            if 'tests_run' in category_results:
                total_tests += category_results['tests_run']
                total_passed += category_results['passed']
                total_failed += category_results['failed']
                total_skipped += category_results['skipped']
                total_errors += category_results['errors']
        
        results['overall_summary'] = {
            'categories_run': categories_run,
            'categories_passed': categories_passed,
            'total_tests': total_tests,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'total_skipped': total_skipped,
            'total_errors': total_errors,
            'overall_duration': overall_duration
        }
        
        # Print overall summary
        self._print_overall_summary(results)
        
        # Generate summary report
        self._generate_summary_report(results)
        
        return results
    
    def _print_overall_summary(self, results: Dict[str, Any]) -> None:
        """Print overall test summary."""
        summary = results['overall_summary']
        
        print(f"\n{'='*80}")
        print("OVERALL TEST SUMMARY")
        print(f"{'='*80}")
        
        print(f"Duration: {summary['overall_duration']:.2f} seconds")
        print(f"Categories: {summary['categories_passed']}/{summary['categories_run']} passed")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['total_passed']}")
        print(f"Failed: {summary['total_failed']}")
        print(f"Skipped: {summary['total_skipped']}")
        print(f"Errors: {summary['total_errors']}")
        
        if summary['total_tests'] > 0:
            pass_rate = (summary['total_passed'] / summary['total_tests']) * 100
            print(f"Overall Pass Rate: {pass_rate:.1f}%")
        
        # Print category breakdown
        print(f"\nCategory Breakdown:")
        for category, category_results in results['categories'].items():
            status = category_results.get('status', 'unknown')
            duration = category_results.get('duration', 0)
            tests = category_results.get('tests_run', 0)
            
            status_emoji = {
                'passed': '✅',
                'failed': '❌',
                'timeout': '⏰',
                'error': '💥',
                'skipped': '⏭️'
            }.get(status, '❓')
            
            print(f"  {status_emoji} {category:<12} {status:<8} {tests:>3} tests {duration:>6.1f}s")
        
        # Overall result
        all_passed = summary['categories_passed'] == summary['categories_run']
        if all_passed and summary['total_failed'] == 0 and summary['total_errors'] == 0:
            print(f"\n🎉 ALL TESTS PASSED! 🎉")
        else:
            print(f"\n⚠️  SOME TESTS FAILED OR HAD ISSUES")
    
    def _generate_summary_report(self, results: Dict[str, Any]) -> str:
        """Generate summary report file."""
        report_file = self.reports_dir / "test_summary.txt"
        
        try:
            with open(report_file, 'w') as f:
                f.write("BATCH PROCESSING TEST SUITE SUMMARY REPORT\n")
                f.write("="*50 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Start Time: {results['start_time']}\n")
                f.write(f"End Time: {results['end_time']}\n")
                f.write(f"Duration: {results['overall_duration']:.2f} seconds\n\n")
                
                # Overall summary
                summary = results['overall_summary']
                f.write("OVERALL SUMMARY\n")
                f.write("-"*15 + "\n")
                f.write(f"Categories Run: {summary['categories_run']}\n")
                f.write(f"Categories Passed: {summary['categories_passed']}\n")
                f.write(f"Total Tests: {summary['total_tests']}\n")
                f.write(f"Passed: {summary['total_passed']}\n")
                f.write(f"Failed: {summary['total_failed']}\n")
                f.write(f"Skipped: {summary['total_skipped']}\n")
                f.write(f"Errors: {summary['total_errors']}\n")
                
                if summary['total_tests'] > 0:
                    pass_rate = (summary['total_passed'] / summary['total_tests']) * 100
                    f.write(f"Pass Rate: {pass_rate:.1f}%\n")
                f.write("\n")
                
                # Category details
                f.write("CATEGORY DETAILS\n")
                f.write("-"*16 + "\n")
                for category, category_results in results['categories'].items():
                    f.write(f"\n{category.upper()} TESTS:\n")
                    f.write(f"  Status: {category_results.get('status', 'unknown')}\n")
                    f.write(f"  Duration: {category_results.get('duration', 0):.2f}s\n")
                    
                    if 'tests_run' in category_results:
                        f.write(f"  Tests Run: {category_results['tests_run']}\n")
                        f.write(f"  Passed: {category_results['passed']}\n")
                        f.write(f"  Failed: {category_results['failed']}\n")
                        f.write(f"  Skipped: {category_results['skipped']}\n")
                        f.write(f"  Errors: {category_results['errors']}\n")
                    
                    if 'output_file' in category_results:
                        f.write(f"  Output File: {category_results['output_file']}\n")
            
            print(f"\n📄 Summary report generated: {report_file}")
            return str(report_file)
            
        except Exception as e:
            print(f"Error generating summary report: {e}")
            return ""
    
    def run_coverage_analysis(self) -> Dict[str, Any]:
        """Run coverage analysis using the coverage analysis module."""
        print(f"\n{'='*60}")
        print("RUNNING COVERAGE ANALYSIS")
        print(f"{'='*60}")
        
        try:
            # Import and run coverage analysis
            sys.path.insert(0, str(self.test_dir))
            from test_coverage_analysis import CoverageAnalyzer
            
            analyzer = CoverageAnalyzer(str(self.src_dir), str(self.test_dir))
            results = analyzer.run_coverage_analysis()
            
            # Generate reports in our reports directory
            analyzer.generate_html_report()
            analyzer.generate_text_report(str(self.reports_dir / "coverage_report.txt"))
            analyzer.generate_json_report(str(self.reports_dir / "coverage_report.json"))
            
            return results
            
        except Exception as e:
            print(f"❌ Coverage analysis failed: {e}")
            return {'error': str(e)}


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Comprehensive Batch Processing Test Runner")
    
    parser.add_argument('--category', 
                       choices=['unit', 'integration', 'performance', 'stress', 'existing', 'all'],
                       default='all',
                       help='Test category to run')
    
    parser.add_argument('--coverage', action='store_true',
                       help='Include coverage analysis')
    
    parser.add_argument('--include-stress', action='store_true',
                       help='Include stress tests (can be slow)')
    
    parser.add_argument('--include-existing', action='store_true',
                       help='Include existing test files')
    
    parser.add_argument('--timeout', type=int, default=3600,
                       help='Timeout for test execution (seconds)')
    
    parser.add_argument('--base-dir', type=str,
                       help='Base directory (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Create test runner
    runner = BatchTestRunner(args.base_dir)
    
    try:
        if args.category == 'all':
            # Run all tests
            results = runner.run_all_tests(
                coverage=args.coverage,
                include_stress=args.include_stress,
                include_existing=args.include_existing,
                timeout=args.timeout
            )
            
            # Run coverage analysis if requested
            if args.coverage:
                coverage_results = runner.run_coverage_analysis()
                results['coverage_analysis'] = coverage_results
            
            # Determine exit code
            summary = results['overall_summary']
            if summary['categories_passed'] == summary['categories_run'] and summary['total_failed'] == 0:
                print("\n🎉 All tests completed successfully!")
                return 0
            else:
                print("\n⚠️  Some tests failed or had issues.")
                return 1
                
        else:
            # Run specific category
            results = runner.run_test_category(args.category, 
                                             coverage=args.coverage,
                                             timeout=args.timeout)
            
            if results['status'] == 'passed':
                print(f"\n✅ {args.category} tests passed!")
                return 0
            else:
                print(f"\n❌ {args.category} tests failed.")
                return 1
    
    except KeyboardInterrupt:
        print("\n⏹️  Testing interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)