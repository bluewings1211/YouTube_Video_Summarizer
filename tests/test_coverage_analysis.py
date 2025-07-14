"""
Test coverage analysis and reporting for batch processing components.

This script provides comprehensive coverage analysis including:
- Coverage measurement for all batch processing modules
- Detailed coverage reports with line-by-line analysis
- Missing coverage identification and recommendations
- Coverage trends and quality metrics
- HTML and text report generation
- Integration with CI/CD pipelines
"""

import pytest
import coverage
import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import importlib.util


class CoverageAnalyzer:
    """Analyze test coverage for batch processing components."""
    
    def __init__(self, source_dir: str = "src", test_dir: str = "tests"):
        self.source_dir = Path(source_dir)
        self.test_dir = Path(test_dir)
        self.cov = coverage.Coverage()
        self.coverage_data = {}
        self.analysis_results = {}
        
    def discover_batch_modules(self) -> List[str]:
        """Discover all batch processing related modules."""
        batch_modules = []
        
        # Core batch processing modules
        batch_patterns = [
            "services/batch_service.py",
            "services/queue_service.py",
            "services/concurrent_batch_service.py",
            "services/batch_processor.py",
            "database/batch_models.py",
            "utils/batch_logger.py",
            "utils/batch_monitor.py",
            "utils/concurrency_manager.py",
            "api/batch.py"
        ]
        
        for pattern in batch_patterns:
            module_path = self.source_dir / pattern
            if module_path.exists():
                batch_modules.append(str(module_path))
        
        return batch_modules
    
    def discover_test_files(self) -> List[str]:
        """Discover all test files related to batch processing."""
        test_files = []
        
        # Comprehensive test patterns
        test_patterns = [
            "test_batch_service*.py",
            "test_queue_service*.py",
            "test_concurrent_batch*.py",
            "test_batch_processor*.py",
            "test_batch_integration*.py",
            "test_batch_performance*.py",
            "test_batch_stress*.py",
            "test_batch_*.py"
        ]
        
        for pattern in test_patterns:
            for test_file in self.test_dir.glob(pattern):
                if test_file.is_file():
                    test_files.append(str(test_file))
        
        return test_files
    
    def run_coverage_analysis(self) -> Dict[str, Any]:
        """Run comprehensive coverage analysis."""
        print("Starting coverage analysis for batch processing components...")
        
        # Start coverage measurement
        self.cov.start()
        
        try:
            # Import and analyze batch modules
            batch_modules = self.discover_batch_modules()
            test_files = self.discover_test_files()
            
            print(f"Found {len(batch_modules)} batch modules")
            print(f"Found {len(test_files)} test files")
            
            # Run tests with coverage
            test_results = self._run_tests_with_coverage(test_files)
            
            # Stop coverage measurement
            self.cov.stop()
            self.cov.save()
            
            # Generate coverage analysis
            coverage_results = self._analyze_coverage(batch_modules)
            
            # Combine results
            self.analysis_results = {
                'timestamp': datetime.now().isoformat(),
                'modules_analyzed': len(batch_modules),
                'test_files_run': len(test_files),
                'test_results': test_results,
                'coverage_results': coverage_results,
                'recommendations': self._generate_recommendations(coverage_results)
            }
            
            return self.analysis_results
            
        except Exception as e:
            self.cov.stop()
            raise e
    
    def _run_tests_with_coverage(self, test_files: List[str]) -> Dict[str, Any]:
        """Run tests with coverage measurement."""
        test_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'skipped_tests': 0,
            'test_details': []
        }
        
        for test_file in test_files:
            try:
                # Run pytest for each test file
                result = subprocess.run([
                    sys.executable, '-m', 'pytest', 
                    test_file, 
                    '-v', 
                    '--tb=short',
                    '--disable-warnings'
                ], capture_output=True, text=True, timeout=300)
                
                # Parse test results
                file_results = self._parse_pytest_output(result.stdout, result.stderr)
                test_results['test_details'].append({
                    'file': test_file,
                    'results': file_results,
                    'return_code': result.returncode
                })
                
                # Aggregate results
                test_results['total_tests'] += file_results.get('total', 0)
                test_results['passed_tests'] += file_results.get('passed', 0)
                test_results['failed_tests'] += file_results.get('failed', 0)
                test_results['skipped_tests'] += file_results.get('skipped', 0)
                
            except subprocess.TimeoutExpired:
                test_results['test_details'].append({
                    'file': test_file,
                    'results': {'error': 'timeout'},
                    'return_code': -1
                })
            except Exception as e:
                test_results['test_details'].append({
                    'file': test_file,
                    'results': {'error': str(e)},
                    'return_code': -1
                })
        
        return test_results
    
    def _parse_pytest_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse pytest output to extract test results."""
        results = {'total': 0, 'passed': 0, 'failed': 0, 'skipped': 0}
        
        # Look for result summary line
        lines = stdout.split('\n')
        for line in lines:
            if 'passed' in line or 'failed' in line or 'skipped' in line:
                # Try to extract numbers
                if '==' in line and ('passed' in line or 'failed' in line):
                    # Example: "== 5 passed, 2 failed in 1.23s =="
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
        
        results['total'] = results['passed'] + results['failed'] + results['skipped']
        return results
    
    def _analyze_coverage(self, modules: List[str]) -> Dict[str, Any]:
        """Analyze coverage for specified modules."""
        coverage_results = {
            'overall_coverage': 0.0,
            'module_coverage': {},
            'uncovered_lines': {},
            'coverage_summary': {}
        }
        
        try:
            # Generate coverage report
            total_statements = 0
            total_missing = 0
            
            for module_path in modules:
                try:
                    # Get coverage data for this module
                    analysis = self.cov.analysis2(module_path)
                    statements, missing, excluded, plugin = analysis
                    
                    # Calculate coverage percentage
                    if statements:
                        covered = len(statements) - len(missing)
                        coverage_percent = (covered / len(statements)) * 100
                    else:
                        coverage_percent = 100.0
                    
                    module_name = os.path.basename(module_path)
                    coverage_results['module_coverage'][module_name] = {
                        'coverage_percent': coverage_percent,
                        'statements': len(statements),
                        'missing': len(missing),
                        'covered': len(statements) - len(missing)
                    }
                    
                    if missing:
                        coverage_results['uncovered_lines'][module_name] = sorted(missing)
                    
                    total_statements += len(statements)
                    total_missing += len(missing)
                    
                except Exception as e:
                    coverage_results['module_coverage'][os.path.basename(module_path)] = {
                        'error': str(e)
                    }
            
            # Calculate overall coverage
            if total_statements > 0:
                coverage_results['overall_coverage'] = ((total_statements - total_missing) / total_statements) * 100
            
            # Generate summary
            coverage_results['coverage_summary'] = {
                'total_statements': total_statements,
                'total_missing': total_missing,
                'total_covered': total_statements - total_missing,
                'modules_analyzed': len([m for m in coverage_results['module_coverage'].values() if 'error' not in m])
            }
            
        except Exception as e:
            coverage_results['error'] = str(e)
        
        return coverage_results
    
    def _generate_recommendations(self, coverage_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on coverage analysis."""
        recommendations = []
        
        overall_coverage = coverage_results.get('overall_coverage', 0)
        module_coverage = coverage_results.get('module_coverage', {})
        
        # Overall coverage recommendations
        if overall_coverage < 80:
            recommendations.append(
                f"Overall coverage is {overall_coverage:.1f}%. "
                f"Consider adding more tests to reach 80%+ coverage."
            )
        elif overall_coverage < 90:
            recommendations.append(
                f"Overall coverage is {overall_coverage:.1f}%. "
                f"Good coverage, but consider targeting 90%+ for critical components."
            )
        else:
            recommendations.append(
                f"Excellent overall coverage at {overall_coverage:.1f}%!"
            )
        
        # Module-specific recommendations
        low_coverage_modules = []
        for module, data in module_coverage.items():
            if isinstance(data, dict) and 'coverage_percent' in data:
                coverage_percent = data['coverage_percent']
                if coverage_percent < 70:
                    low_coverage_modules.append((module, coverage_percent))
        
        if low_coverage_modules:
            recommendations.append(
                "Modules with low coverage (< 70%):"
            )
            for module, coverage in sorted(low_coverage_modules, key=lambda x: x[1]):
                recommendations.append(f"  - {module}: {coverage:.1f}%")
        
        # Specific testing recommendations
        critical_modules = [
            'batch_service.py',
            'queue_service.py',
            'concurrent_batch_service.py',
            'batch_processor.py'
        ]
        
        for critical_module in critical_modules:
            if critical_module in module_coverage:
                data = module_coverage[critical_module]
                if isinstance(data, dict) and 'coverage_percent' in data:
                    if data['coverage_percent'] < 85:
                        recommendations.append(
                            f"Critical module {critical_module} has {data['coverage_percent']:.1f}% coverage. "
                            f"Consider adding more comprehensive tests."
                        )
        
        # Test quality recommendations
        if not recommendations:
            recommendations.append("Coverage analysis looks good! Consider adding more edge case tests.")
        
        return recommendations
    
    def generate_html_report(self, output_file: str = "coverage_report.html") -> str:
        """Generate HTML coverage report."""
        try:
            self.cov.html_report(directory="htmlcov", title="Batch Processing Coverage Report")
            print(f"HTML coverage report generated in 'htmlcov' directory")
            return "htmlcov/index.html"
        except Exception as e:
            print(f"Error generating HTML report: {e}")
            return ""
    
    def generate_text_report(self, output_file: str = "coverage_report.txt") -> str:
        """Generate text coverage report."""
        try:
            with open(output_file, 'w') as f:
                # Write header
                f.write("BATCH PROCESSING COVERAGE ANALYSIS REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Write test results
                if 'test_results' in self.analysis_results:
                    test_results = self.analysis_results['test_results']
                    f.write("TEST EXECUTION SUMMARY\n")
                    f.write("-" * 25 + "\n")
                    f.write(f"Total Tests: {test_results['total_tests']}\n")
                    f.write(f"Passed: {test_results['passed_tests']}\n")
                    f.write(f"Failed: {test_results['failed_tests']}\n")
                    f.write(f"Skipped: {test_results['skipped_tests']}\n\n")
                
                # Write coverage results
                if 'coverage_results' in self.analysis_results:
                    coverage_results = self.analysis_results['coverage_results']
                    f.write("COVERAGE ANALYSIS\n")
                    f.write("-" * 17 + "\n")
                    f.write(f"Overall Coverage: {coverage_results.get('overall_coverage', 0):.1f}%\n\n")
                    
                    # Module breakdown
                    f.write("MODULE BREAKDOWN\n")
                    f.write("-" * 16 + "\n")
                    module_coverage = coverage_results.get('module_coverage', {})
                    for module, data in sorted(module_coverage.items()):
                        if isinstance(data, dict) and 'coverage_percent' in data:
                            f.write(f"{module:<30} {data['coverage_percent']:>6.1f}% "
                                   f"({data['covered']}/{data['statements']} lines)\n")
                    f.write("\n")
                    
                    # Uncovered lines
                    uncovered = coverage_results.get('uncovered_lines', {})
                    if uncovered:
                        f.write("UNCOVERED LINES\n")
                        f.write("-" * 15 + "\n")
                        for module, lines in uncovered.items():
                            f.write(f"{module}: {', '.join(map(str, lines))}\n")
                        f.write("\n")
                
                # Write recommendations
                if 'recommendations' in self.analysis_results:
                    f.write("RECOMMENDATIONS\n")
                    f.write("-" * 15 + "\n")
                    for i, rec in enumerate(self.analysis_results['recommendations'], 1):
                        f.write(f"{i}. {rec}\n")
            
            print(f"Text coverage report generated: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"Error generating text report: {e}")
            return ""
    
    def generate_json_report(self, output_file: str = "coverage_report.json") -> str:
        """Generate JSON coverage report for programmatic use."""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.analysis_results, f, indent=2, default=str)
            
            print(f"JSON coverage report generated: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"Error generating JSON report: {e}")
            return ""


def test_coverage_analysis():
    """Test function for coverage analysis."""
    analyzer = CoverageAnalyzer()
    
    # Run coverage analysis
    results = analyzer.run_coverage_analysis()
    
    # Generate reports
    html_report = analyzer.generate_html_report()
    text_report = analyzer.generate_text_report()
    json_report = analyzer.generate_json_report()
    
    # Print summary
    print("\nCOVERAGE ANALYSIS SUMMARY")
    print("=" * 30)
    
    if 'coverage_results' in results:
        coverage_results = results['coverage_results']
        overall_coverage = coverage_results.get('overall_coverage', 0)
        print(f"Overall Coverage: {overall_coverage:.1f}%")
        
        # Coverage by category
        module_coverage = coverage_results.get('module_coverage', {})
        service_modules = [k for k in module_coverage.keys() if 'service' in k]
        api_modules = [k for k in module_coverage.keys() if 'api' in k]
        util_modules = [k for k in module_coverage.keys() if 'util' in k or 'batch_' in k]
        
        if service_modules:
            service_avg = sum(module_coverage[m].get('coverage_percent', 0) for m in service_modules) / len(service_modules)
            print(f"Service Layer Coverage: {service_avg:.1f}%")
        
        if api_modules:
            api_avg = sum(module_coverage[m].get('coverage_percent', 0) for m in api_modules) / len(api_modules)
            print(f"API Layer Coverage: {api_avg:.1f}%")
        
        if util_modules:
            util_avg = sum(module_coverage[m].get('coverage_percent', 0) for m in util_modules) / len(util_modules)
            print(f"Utility Layer Coverage: {util_avg:.1f}%")
    
    if 'test_results' in results:
        test_results = results['test_results']
        total_tests = test_results['total_tests']
        passed_tests = test_results['passed_tests']
        if total_tests > 0:
            pass_rate = (passed_tests / total_tests) * 100
            print(f"Test Pass Rate: {pass_rate:.1f}% ({passed_tests}/{total_tests})")
    
    print("\nReports Generated:")
    if html_report:
        print(f"  - HTML: {html_report}")
    if text_report:
        print(f"  - Text: {text_report}")
    if json_report:
        print(f"  - JSON: {json_report}")
    
    # Assert coverage thresholds
    if 'coverage_results' in results:
        overall_coverage = results['coverage_results'].get('overall_coverage', 0)
        assert overall_coverage > 70, f"Overall coverage {overall_coverage:.1f}% is below 70% threshold"
        
        # Check critical module coverage
        critical_modules = [
            'batch_service.py',
            'queue_service.py', 
            'concurrent_batch_service.py',
            'batch_processor.py'
        ]
        
        module_coverage = results['coverage_results'].get('module_coverage', {})
        for critical_module in critical_modules:
            if critical_module in module_coverage:
                data = module_coverage[critical_module]
                if isinstance(data, dict) and 'coverage_percent' in data:
                    coverage_percent = data['coverage_percent']
                    assert coverage_percent > 60, f"Critical module {critical_module} coverage {coverage_percent:.1f}% is below 60%"
    
    return results


def run_coverage_cli():
    """Command line interface for coverage analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch Processing Coverage Analysis")
    parser.add_argument("--source-dir", default="src", help="Source directory")
    parser.add_argument("--test-dir", default="tests", help="Test directory")
    parser.add_argument("--output-html", default="coverage_report.html", help="HTML output file")
    parser.add_argument("--output-text", default="coverage_report.txt", help="Text output file")
    parser.add_argument("--output-json", default="coverage_report.json", help="JSON output file")
    parser.add_argument("--threshold", type=float, default=80.0, help="Coverage threshold")
    
    args = parser.parse_args()
    
    analyzer = CoverageAnalyzer(args.source_dir, args.test_dir)
    
    try:
        # Run analysis
        results = analyzer.run_coverage_analysis()
        
        # Generate reports
        analyzer.generate_html_report(args.output_html)
        analyzer.generate_text_report(args.output_text)
        analyzer.generate_json_report(args.output_json)
        
        # Check threshold
        overall_coverage = results.get('coverage_results', {}).get('overall_coverage', 0)
        
        if overall_coverage >= args.threshold:
            print(f"\n✅ Coverage {overall_coverage:.1f}% meets threshold {args.threshold}%")
            return 0
        else:
            print(f"\n❌ Coverage {overall_coverage:.1f}% below threshold {args.threshold}%")
            return 1
            
    except Exception as e:
        print(f"❌ Coverage analysis failed: {e}")
        return 1


if __name__ == "__main__":
    # Check if running as CLI or test
    if len(sys.argv) > 1:
        exit_code = run_coverage_cli()
        sys.exit(exit_code)
    else:
        # Run as pytest
        test_coverage_analysis()