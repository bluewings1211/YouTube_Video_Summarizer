"""
Language Detection utilities for YouTube video content.

This module provides functionality to detect the primary language of YouTube videos
using metadata analysis, transcript content analysis, and video title/description
processing. Specifically designed to distinguish between English and Chinese content.
"""

import logging
import re
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


class LanguageCode(Enum):
    """Supported language codes for detection."""
    ENGLISH = "en"
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"
    CHINESE_GENERIC = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    UNKNOWN = "unknown"


@dataclass
class LanguageDetectionResult:
    """Results from language detection analysis."""
    detected_language: LanguageCode
    confidence_score: float  # 0.0 to 1.0
    detection_method: str
    metadata_language: Optional[str] = None
    title_language: Optional[str] = None
    description_language: Optional[str] = None
    transcript_language: Optional[str] = None
    alternative_languages: List[Tuple[LanguageCode, float]] = None
    detection_timestamp: str = None
    
    def __post_init__(self):
        if self.detection_timestamp is None:
            self.detection_timestamp = datetime.utcnow().isoformat()
        if self.alternative_languages is None:
            self.alternative_languages = []


class LanguageDetectionError(Exception):
    """Exception raised when language detection fails."""
    
    def __init__(self, message: str, video_id: str = "", detection_method: str = ""):
        super().__init__(message)
        self.message = message
        self.video_id = video_id
        self.detection_method = detection_method
        self.timestamp = datetime.utcnow().isoformat()


class YouTubeLanguageDetector:
    """
    Detects the primary language of YouTube videos using multiple detection methods.
    
    This class combines YouTube metadata analysis, content analysis, and heuristic
    methods to accurately identify video language, with special focus on English
    vs Chinese language detection.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Language detection patterns
        self.english_patterns = {
            'common_words': [
                'the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with',
                'for', 'as', 'was', 'on', 'are', 'you', 'all', 'not', 'be', 'this',
                'have', 'from', 'or', 'one', 'had', 'but', 'words', 'which', 'she',
                'at', 'we', 'what', 'your', 'when', 'him', 'my', 'has', 'its'
            ],
            'titles': [
                'how to', 'tutorial', 'review', 'episode', 'part', 'guide',
                'tips', 'tricks', 'best', 'top', 'ultimate', 'complete'
            ],
            'indicators': [
                'subscribe', 'like', 'comment', 'share', 'follow', 'playlist',
                'channel', 'video', 'watch', 'episode', 'series', 'season'
            ]
        }
        
        self.chinese_patterns = {
            'unicode_ranges': [
                (0x4E00, 0x9FFF),  # CJK Unified Ideographs
                (0x3400, 0x4DBF),  # CJK Extension A
                (0x20000, 0x2A6DF),  # CJK Extension B
                (0x2A700, 0x2B73F),  # CJK Extension C
                (0x2B740, 0x2B81F),  # CJK Extension D
                (0x2B820, 0x2CEAF),  # CJK Extension E
                (0x2CEB0, 0x2EBEF),  # CJK Extension F
                (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
                (0x2F800, 0x2FA1F)  # CJK Compatibility Supplement
            ],
            'common_chars': [
                '的', '是', '了', '在', '有', '人', '这', '我', '个', '上',
                '来', '说', '们', '到', '时', '要', '就', '会', '可', '也',
                '和', '你', '他', '我们', '什么', '没有', '这个', '那个', '怎么'
            ],
            'titles': [
                '教程', '教学', '如何', '方法', '技巧', '攻略', '指南', '评测',
                '介绍', '分享', '解说', '实况', '直播', '合集', '系列', '第'
            ]
        }
    
    def detect_language_from_metadata(self, video_metadata: Dict[str, Any]) -> LanguageDetectionResult:
        """
        Detect language using YouTube video metadata.
        
        Args:
            video_metadata: Dictionary containing video metadata from YouTube
            
        Returns:
            LanguageDetectionResult with detection information
            
        Raises:
            LanguageDetectionError: If metadata analysis fails
        """
        try:
            self.logger.debug("Starting metadata-based language detection")
            
            # Extract language indicators from metadata
            metadata_language = video_metadata.get('language')
            default_audio_language = video_metadata.get('defaultAudioLanguage')
            title = video_metadata.get('title', '')
            description = video_metadata.get('description', '')
            
            # Initialize detection scores
            language_scores = {
                LanguageCode.ENGLISH: 0.0,
                LanguageCode.CHINESE_SIMPLIFIED: 0.0,
                LanguageCode.CHINESE_TRADITIONAL: 0.0,
                LanguageCode.CHINESE_GENERIC: 0.0
            }
            
            # Analyze explicit language metadata
            if metadata_language:
                language_scores.update(self._score_language_code(metadata_language, 0.8))
                self.logger.debug(f"Metadata language: {metadata_language}")
            
            if default_audio_language:
                language_scores.update(self._score_language_code(default_audio_language, 0.7))
                self.logger.debug(f"Default audio language: {default_audio_language}")
            
            # Analyze title content
            if title:
                title_scores = self._analyze_text_content(title)
                for lang, score in title_scores.items():
                    language_scores[lang] += score * 0.6  # Title has high importance
            
            # Analyze description content (first 500 chars)
            if description:
                desc_sample = description[:500]
                desc_scores = self._analyze_text_content(desc_sample)
                for lang, score in desc_scores.items():
                    language_scores[lang] += score * 0.3  # Description has lower importance
            
            # Determine best match
            best_language = max(language_scores.items(), key=lambda x: x[1])
            detected_language, confidence = best_language
            
            # Normalize confidence score
            max_possible_score = 1.8  # 0.8 + 0.7 + 0.3 (max from each source)
            normalized_confidence = min(1.0, confidence / max_possible_score)
            
            # Create alternative languages list
            alternatives = [
                (lang, score / max_possible_score) 
                for lang, score in sorted(language_scores.items(), key=lambda x: x[1], reverse=True)[1:]
                if score > 0.1
            ]
            
            # If confidence is very low, default to unknown
            if normalized_confidence < 0.2:
                detected_language = LanguageCode.UNKNOWN
                normalized_confidence = 0.1
            
            result = LanguageDetectionResult(
                detected_language=detected_language,
                confidence_score=round(normalized_confidence, 3),
                detection_method="metadata_analysis",
                metadata_language=metadata_language or default_audio_language,
                title_language=self._get_dominant_language(title) if title else None,
                description_language=self._get_dominant_language(desc_sample) if description else None,
                alternative_languages=alternatives
            )
            
            self.logger.info(f"Metadata detection: {detected_language.value} (confidence: {normalized_confidence:.3f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Metadata language detection failed: {str(e)}")
            raise LanguageDetectionError(f"Metadata analysis failed: {str(e)}", detection_method="metadata_analysis")
    
    def detect_language_from_transcript(self, transcript_text: str) -> LanguageDetectionResult:
        """
        Detect language using transcript content analysis.
        
        Args:
            transcript_text: Full transcript text content
            
        Returns:
            LanguageDetectionResult with detection information
            
        Raises:
            LanguageDetectionError: If transcript analysis fails
        """
        try:
            self.logger.debug("Starting transcript-based language detection")
            
            if not transcript_text or not transcript_text.strip():
                raise LanguageDetectionError("Empty transcript provided for analysis")
            
            # Analyze transcript content
            language_scores = self._analyze_text_content(transcript_text)
            
            # Determine best match
            best_language = max(language_scores.items(), key=lambda x: x[1])
            detected_language, confidence = best_language
            
            # Create alternative languages list
            alternatives = [
                (lang, score) 
                for lang, score in sorted(language_scores.items(), key=lambda x: x[1], reverse=True)[1:]
                if score > 0.1
            ]
            
            # If confidence is very low, default to unknown
            if confidence < 0.3:
                detected_language = LanguageCode.UNKNOWN
                confidence = 0.2
            
            result = LanguageDetectionResult(
                detected_language=detected_language,
                confidence_score=round(confidence, 3),
                detection_method="transcript_analysis",
                transcript_language=detected_language.value,
                alternative_languages=alternatives
            )
            
            self.logger.info(f"Transcript detection: {detected_language.value} (confidence: {confidence:.3f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Transcript language detection failed: {str(e)}")
            raise LanguageDetectionError(f"Transcript analysis failed: {str(e)}", detection_method="transcript_analysis")
    
    def detect_language_comprehensive(
        self, 
        video_metadata: Dict[str, Any], 
        transcript_text: Optional[str] = None
    ) -> LanguageDetectionResult:
        """
        Perform comprehensive language detection using all available methods.
        
        Args:
            video_metadata: Video metadata from YouTube
            transcript_text: Optional transcript text for additional analysis
            
        Returns:
            LanguageDetectionResult with combined detection information
            
        Raises:
            LanguageDetectionError: If comprehensive analysis fails
        """
        try:
            self.logger.debug("Starting comprehensive language detection")
            
            # Start with metadata detection
            metadata_result = self.detect_language_from_metadata(video_metadata)
            
            # If we have transcript, combine results
            if transcript_text and transcript_text.strip():
                transcript_result = self.detect_language_from_transcript(transcript_text)
                
                # Combine scores with weighted average
                # Metadata gets 60% weight, transcript gets 40% weight
                combined_confidence = (
                    metadata_result.confidence_score * 0.6 + 
                    transcript_result.confidence_score * 0.4
                )
                
                # Choose language based on highest combined score
                if (transcript_result.detected_language == metadata_result.detected_language or
                    metadata_result.confidence_score > 0.7):
                    # Languages agree or metadata is very confident
                    detected_language = metadata_result.detected_language
                elif transcript_result.confidence_score > 0.8:
                    # Transcript is very confident and differs from metadata
                    detected_language = transcript_result.detected_language
                else:
                    # Default to metadata result
                    detected_language = metadata_result.detected_language
                
                result = LanguageDetectionResult(
                    detected_language=detected_language,
                    confidence_score=round(combined_confidence, 3),
                    detection_method="comprehensive_analysis",
                    metadata_language=metadata_result.metadata_language,
                    title_language=metadata_result.title_language,
                    description_language=metadata_result.description_language,
                    transcript_language=transcript_result.transcript_language,
                    alternative_languages=metadata_result.alternative_languages
                )
                
                self.logger.info(f"Comprehensive detection: {detected_language.value} (confidence: {combined_confidence:.3f})")
                return result
            
            else:
                # Only metadata available
                self.logger.info("Using metadata-only detection (no transcript provided)")
                return metadata_result
                
        except Exception as e:
            self.logger.error(f"Comprehensive language detection failed: {str(e)}")
            raise LanguageDetectionError(f"Comprehensive analysis failed: {str(e)}", detection_method="comprehensive_analysis")
    
    def is_chinese_content(self, detection_result: LanguageDetectionResult) -> bool:
        """
        Determine if content is primarily Chinese language.
        
        Args:
            detection_result: Result from language detection
            
        Returns:
            True if content is Chinese, False otherwise
        """
        chinese_languages = {
            LanguageCode.CHINESE_SIMPLIFIED,
            LanguageCode.CHINESE_TRADITIONAL,
            LanguageCode.CHINESE_GENERIC
        }
        return detection_result.detected_language in chinese_languages
    
    def is_english_content(self, detection_result: LanguageDetectionResult) -> bool:
        """
        Determine if content is primarily English language.
        
        Args:
            detection_result: Result from language detection
            
        Returns:
            True if content is English, False otherwise
        """
        return detection_result.detected_language == LanguageCode.ENGLISH
    
    def get_preferred_transcript_languages(self, detection_result: LanguageDetectionResult) -> List[str]:
        """
        Get ordered list of preferred transcript languages based on detection.
        
        Args:
            detection_result: Result from language detection
            
        Returns:
            List of language codes in order of preference
        """
        detected = detection_result.detected_language
        
        if detected == LanguageCode.ENGLISH:
            return ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU']
        elif detected == LanguageCode.CHINESE_SIMPLIFIED:
            return ['zh-CN', 'zh', 'zh-Hans', 'zh-TW', 'zh-Hant']
        elif detected == LanguageCode.CHINESE_TRADITIONAL:
            return ['zh-TW', 'zh-Hant', 'zh', 'zh-CN', 'zh-Hans']
        elif detected == LanguageCode.CHINESE_GENERIC:
            return ['zh', 'zh-CN', 'zh-TW', 'zh-Hans', 'zh-Hant']
        elif detected == LanguageCode.JAPANESE:
            return ['ja', 'ja-JP']
        elif detected == LanguageCode.KOREAN:
            return ['ko', 'ko-KR']
        else:
            # Default fallback with common languages
            return ['en', 'zh-CN', 'zh-TW', 'zh', 'ja', 'ko', 'es', 'fr', 'de']
    
    def _score_language_code(self, language_code: str, weight: float = 1.0) -> Dict[LanguageCode, float]:
        """Score language codes against our supported languages."""
        scores = {lang: 0.0 for lang in LanguageCode}
        
        if not language_code:
            return scores
        
        lang_lower = language_code.lower()
        
        # English variations
        if lang_lower in ['en', 'en-us', 'en-gb', 'en-ca', 'en-au', 'english']:
            scores[LanguageCode.ENGLISH] = weight
        
        # Chinese variations
        elif lang_lower in ['zh-cn', 'zh-hans', 'chinese_simplified', 'chinese (simplified)']:
            scores[LanguageCode.CHINESE_SIMPLIFIED] = weight
        elif lang_lower in ['zh-tw', 'zh-hant', 'chinese_traditional', 'chinese (traditional)']:
            scores[LanguageCode.CHINESE_TRADITIONAL] = weight
        elif lang_lower in ['zh', 'chinese', 'chi']:
            scores[LanguageCode.CHINESE_GENERIC] = weight
        
        # Other languages
        elif lang_lower in ['ja', 'ja-jp', 'japanese']:
            scores[LanguageCode.JAPANESE] = weight * 0.1  # Lower weight for non-target languages
        elif lang_lower in ['ko', 'ko-kr', 'korean']:
            scores[LanguageCode.KOREAN] = weight * 0.1
        
        return scores
    
    def _analyze_text_content(self, text: str) -> Dict[LanguageCode, float]:
        """Analyze text content to score language likelihood."""
        if not text:
            return {lang: 0.0 for lang in LanguageCode}
        
        text_lower = text.lower()
        text_len = len(text)
        
        # Initialize scores
        scores = {
            LanguageCode.ENGLISH: 0.0,
            LanguageCode.CHINESE_SIMPLIFIED: 0.0,
            LanguageCode.CHINESE_TRADITIONAL: 0.0,
            LanguageCode.CHINESE_GENERIC: 0.0,
            LanguageCode.JAPANESE: 0.0,
            LanguageCode.KOREAN: 0.0,
            LanguageCode.UNKNOWN: 0.0
        }
        
        # Count Chinese characters
        chinese_char_count = self._count_chinese_characters(text)
        chinese_ratio = chinese_char_count / max(text_len, 1)
        
        # If significant Chinese character presence
        if chinese_ratio > 0.3:
            scores[LanguageCode.CHINESE_GENERIC] = chinese_ratio * 0.8
            scores[LanguageCode.CHINESE_SIMPLIFIED] = chinese_ratio * 0.7
            scores[LanguageCode.CHINESE_TRADITIONAL] = chinese_ratio * 0.6
        
        # Count English indicators
        english_word_count = 0
        words = text_lower.split()
        
        for word in words:
            if word in self.english_patterns['common_words']:
                english_word_count += 1
        
        # Check for English title patterns
        for pattern in self.english_patterns['titles']:
            if pattern in text_lower:
                english_word_count += 2
        
        # Check for English indicators
        for indicator in self.english_patterns['indicators']:
            if indicator in text_lower:
                english_word_count += 1
        
        # Calculate English score
        if len(words) > 0:
            english_ratio = english_word_count / len(words)
            scores[LanguageCode.ENGLISH] = min(1.0, english_ratio * 1.2)
        
        # Check for Chinese title patterns
        chinese_title_score = 0
        for pattern in self.chinese_patterns['titles']:
            if pattern in text:
                chinese_title_score += 0.2
        
        # Add Chinese title bonus
        if chinese_title_score > 0:
            scores[LanguageCode.CHINESE_GENERIC] += chinese_title_score
            scores[LanguageCode.CHINESE_SIMPLIFIED] += chinese_title_score * 0.8
            scores[LanguageCode.CHINESE_TRADITIONAL] += chinese_title_score * 0.6
        
        # Normalize scores to ensure they don't exceed 1.0
        for lang in scores:
            scores[lang] = min(1.0, scores[lang])
        
        return scores
    
    def _count_chinese_characters(self, text: str) -> int:
        """Count Chinese characters in text using Unicode ranges."""
        count = 0
        for char in text:
            char_code = ord(char)
            for start, end in self.chinese_patterns['unicode_ranges']:
                if start <= char_code <= end:
                    count += 1
                    break
        return count
    
    def _get_dominant_language(self, text: str) -> Optional[str]:
        """Get the dominant language of a text sample."""
        if not text:
            return None
        
        scores = self._analyze_text_content(text)
        best_language = max(scores.items(), key=lambda x: x[1])
        
        if best_language[1] > 0.3:  # Minimum confidence threshold
            return best_language[0].value
        
        return None


# Convenience functions
def detect_video_language(video_metadata: Dict[str, Any], transcript_text: Optional[str] = None) -> LanguageDetectionResult:
    """
    Convenience function to detect video language.
    
    Args:
        video_metadata: Video metadata from YouTube
        transcript_text: Optional transcript text
        
    Returns:
        LanguageDetectionResult with detection information
    """
    detector = YouTubeLanguageDetector()
    return detector.detect_language_comprehensive(video_metadata, transcript_text)


def is_chinese_video(video_metadata: Dict[str, Any], transcript_text: Optional[str] = None) -> bool:
    """
    Convenience function to check if video is Chinese.
    
    Args:
        video_metadata: Video metadata from YouTube
        transcript_text: Optional transcript text
        
    Returns:
        True if video is Chinese, False otherwise
    """
    detector = YouTubeLanguageDetector()
    result = detector.detect_language_comprehensive(video_metadata, transcript_text)
    return detector.is_chinese_content(result)


def is_english_video(video_metadata: Dict[str, Any], transcript_text: Optional[str] = None) -> bool:
    """
    Convenience function to check if video is English.
    
    Args:
        video_metadata: Video metadata from YouTube
        transcript_text: Optional transcript text
        
    Returns:
        True if video is English, False otherwise
    """
    detector = YouTubeLanguageDetector()
    result = detector.detect_language_comprehensive(video_metadata, transcript_text)
    return detector.is_english_content(result)


def get_preferred_transcript_languages(video_metadata: Dict[str, Any], transcript_text: Optional[str] = None) -> List[str]:
    """
    Get preferred transcript languages based on video language detection.
    
    Args:
        video_metadata: Video metadata from YouTube
        transcript_text: Optional transcript text
        
    Returns:
        List of language codes in order of preference
    """
    detector = YouTubeLanguageDetector()
    result = detector.detect_language_comprehensive(video_metadata, transcript_text)
    return detector.get_preferred_transcript_languages(result)