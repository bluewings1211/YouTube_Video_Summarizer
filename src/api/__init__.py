"""
API package for YouTube Summarizer application.

This package contains FastAPI routers and endpoints for the application.
"""

from .history import router as history_router

__all__ = [
    'history_router',
]