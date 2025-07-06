#!/usr/bin/env python3
"""
Environment setup script for Enhanced YouTube Transcript Processing.
This script helps users configure their environment for different deployment scenarios.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional

def print_banner():
    """Print setup banner."""
    print("=" * 80)
    print("Enhanced YouTube Transcript Processing - Environment Setup")
    print("=" * 80)
    print()

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent

def copy_env_template(environment: str, project_root: Path) -> bool:
    """Copy environment template to .env file."""
    template_file = project_root / f".env.{environment}"
    target_file = project_root / ".env"
    
    if not template_file.exists():
        print(f"❌ Template file {template_file} not found!")
        return False
    
    # Backup existing .env file if it exists
    if target_file.exists():
        backup_file = project_root / ".env.backup"
        shutil.copy2(target_file, backup_file)
        print(f"📁 Backed up existing .env to {backup_file}")
    
    # Copy template
    shutil.copy2(template_file, target_file)
    print(f"✅ Created .env from {template_file}")
    return True

def create_directories(project_root: Path) -> None:
    """Create necessary directories."""
    directories = [
        "logs",
        "credentials", 
        "data",
        "tmp"
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(exist_ok=True)
        print(f"📁 Created directory: {dir_path}")

def setup_ollama_models(models: List[str]) -> None:
    """Setup recommended Ollama models."""
    print("\n🤖 Setting up Ollama models...")
    
    # Check if Ollama is installed
    if shutil.which("ollama") is None:
        print("❌ Ollama not found. Please install Ollama first.")
        print("   Visit: https://ollama.ai/download")
        return
    
    # Check if Ollama service is running
    import subprocess
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Ollama service is not running. Please start it first:")
            print("   Run: ollama serve")
            return
    except Exception:
        print("❌ Could not connect to Ollama service.")
        return
    
    print("📦 Pulling recommended models...")
    for model in models:
        print(f"   Pulling {model}...")
        try:
            result = subprocess.run(["ollama", "pull", model], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"   ✅ {model} installed successfully")
            else:
                print(f"   ❌ Failed to install {model}: {result.stderr}")
        except Exception as e:
            print(f"   ❌ Error installing {model}: {e}")

def validate_environment(project_root: Path) -> List[str]:
    """Validate the environment setup."""
    issues = []
    
    # Check if .env file exists
    env_file = project_root / ".env"
    if not env_file.exists():
        issues.append(".env file not found")
    
    # Check if required directories exist
    required_dirs = ["logs", "credentials"]
    for directory in required_dirs:
        dir_path = project_root / directory
        if not dir_path.exists():
            issues.append(f"Directory {directory} not found")
    
    # Check if requirements.txt exists
    requirements_file = project_root / "requirements.txt"
    if not requirements_file.exists():
        issues.append("requirements.txt not found")
    
    return issues

def print_next_steps(environment: str, project_root: Path) -> None:
    """Print next steps for the user."""
    print("\n🎉 Environment setup complete!")
    print("\n📋 Next steps:")
    
    print("1. Install Python dependencies:")
    print("   pip install -r requirements.txt")
    
    print("\n2. Edit your .env file to add your API keys:")
    print(f"   nano {project_root}/.env")
    print("   - Set OPENAI_API_KEY if using OpenAI")
    print("   - Set ANTHROPIC_API_KEY if using Anthropic")
    print("   - Set other configuration values as needed")
    
    if environment in ["development", "docker"]:
        print("\n3. Start Ollama (if using local models):")
        print("   ollama serve")
    
    print("\n4. Test your configuration:")
    print("   python src/config.py")
    
    print("\n5. Run the application:")
    if environment == "docker":
        print("   docker-compose up")
    else:
        print("   python src/app.py")
    
    print("\n📚 Additional resources:")
    print(f"   - Ollama setup guide: {project_root}/docs/ollama-setup-guide.md")
    print(f"   - API documentation: {project_root}/docs/api-documentation.md")
    print(f"   - Troubleshooting: {project_root}/docs/troubleshooting-guide.md")

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="Setup environment for Enhanced YouTube Transcript Processing"
    )
    parser.add_argument(
        "environment",
        choices=["development", "production", "docker"],
        help="Environment to setup"
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["llama3.1:8b", "mistral:7b", "llama3.2:3b"],
        help="Ollama models to install"
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip Ollama model installation"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing setup"
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    project_root = get_project_root()
    print(f"🏠 Project root: {project_root}")
    
    if args.validate_only:
        print("\n🔍 Validating environment setup...")
        issues = validate_environment(project_root)
        if issues:
            print("❌ Issues found:")
            for issue in issues:
                print(f"   - {issue}")
            sys.exit(1)
        else:
            print("✅ Environment validation passed!")
            sys.exit(0)
    
    print(f"\n🚀 Setting up {args.environment} environment...")
    
    # Create directories
    create_directories(project_root)
    
    # Copy environment template
    if not copy_env_template(args.environment, project_root):
        sys.exit(1)
    
    # Setup Ollama models if requested and not skipped
    if not args.skip_models and args.environment in ["development", "docker"]:
        setup_ollama_models(args.models)
    
    # Validate setup
    issues = validate_environment(project_root)
    if issues:
        print("\n⚠️  Some issues found:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("\n✅ Environment validation passed!")
    
    # Print next steps
    print_next_steps(args.environment, project_root)

if __name__ == "__main__":
    main()