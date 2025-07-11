"""
Shared data types for semantic analysis.

This module contains data classes and types used across the semantic analysis
system to avoid circular imports.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional


@dataclass
class TranscriptSegment:
    """Represents a single transcript segment with metadata."""
    start_time: float
    end_time: Optional[float]
    text: str
    duration: Optional[float] = None
    speaker: Optional[str] = None
    
    @property
    def word_count(self) -> int:
        """Get word count of the segment."""
        return len(self.text.split())
    
    @property
    def char_count(self) -> int:
        """Get character count of the segment."""
        return len(self.text)


@dataclass
class SemanticCluster:
    """Represents a semantic cluster of transcript segments."""
    cluster_id: str
    segments: List[TranscriptSegment]
    theme: str
    importance_score: float
    start_time: float
    end_time: float
    summary: str
    keywords: List[str]
    
    @property
    def duration(self) -> float:
        """Get total duration of the cluster."""
        return self.end_time - self.start_time
    
    @property
    def word_count(self) -> int:
        """Get total word count of the cluster."""
        return sum(segment.word_count for segment in self.segments)


@dataclass
class SemanticTimestamp:
    """Represents a semantically-derived timestamp."""
    timestamp: float
    title: str
    description: str
    confidence_score: float
    semantic_cluster_id: Optional[str] = None
    keywords: Optional[List[str]] = None
    importance_score: Optional[float] = None
    content_type: Optional[str] = None  # "introduction", "main_point", "conclusion", etc.
    metadata: Optional[Dict[str, Any]] = None