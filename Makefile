.PHONY: help build up down restart logs shell test clean lint format

# Default target
help:
	@echo "YouTube Summarizer Development Commands"
	@echo "======================================"
	@echo "build       - Build Docker images"
	@echo "up          - Start development environment"
	@echo "down        - Stop development environment"
	@echo "restart     - Restart services"
	@echo "logs        - View logs"
	@echo "shell       - Open shell in app container"
	@echo "test        - Run tests"
	@echo "lint        - Run linting"
	@echo "format      - Format code"
	@echo "clean       - Clean up containers and volumes"

# Build Docker images
build:
	docker-compose build

# Start development environment
up:
	docker-compose up -d

# Stop development environment
down:
	docker-compose down

# Restart services
restart:
	docker-compose restart

# View logs
logs:
	docker-compose logs -f

# Open shell in app container
shell:
	docker-compose exec app bash

# Run tests
test:
	docker-compose exec app pytest tests/ -v

# Run linting
lint:
	docker-compose exec app flake8 src tests
	docker-compose exec app mypy src

# Format code
format:
	docker-compose exec app black src tests

# Clean up
clean:
	docker-compose down -v
	docker system prune -f