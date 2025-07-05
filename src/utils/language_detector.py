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


def detect_mixed_language_content(text: str, chunk_size: int = 500) -> Dict[str, Any]:
    """
    Detect mixed-language content by analyzing text segments.
    
    Args:
        text: Text content to analyze for mixed languages
        chunk_size: Size of chunks to analyze for language switching
        
    Returns:
        Dictionary with mixed-language analysis results
    """
    if not text or not text.strip():
        return {
            'is_mixed': False,
            'primary_language': 'en',
            'segments': [],
            'language_distribution': {}
        }
    
    detector = YouTubeLanguageDetector()
    
    # Split text into chunks for analysis
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_text = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk_text)
    
    # Analyze each chunk
    chunk_languages = []
    language_counts = {}
    
    for i, chunk in enumerate(chunks):
        scores = detector._analyze_text_content(chunk)
        best_language = max(scores.items(), key=lambda x: x[1])
        
        if best_language[1] > 0.3:  # Minimum confidence threshold
            detected_lang = best_language[0].value
            chunk_languages.append({
                'chunk_index': i,
                'language': detected_lang,
                'confidence': best_language[1],
                'text_preview': chunk[:100] + "..." if len(chunk) > 100 else chunk
            })
            
            # Count language occurrences
            language_counts[detected_lang] = language_counts.get(detected_lang, 0) + 1
    
    # Determine if content is mixed
    unique_languages = set(segment['language'] for segment in chunk_languages)
    is_mixed = len(unique_languages) > 1
    
    # Calculate language distribution
    total_chunks = len(chunk_languages)
    language_distribution = {}
    if total_chunks > 0:
        for lang, count in language_counts.items():
            language_distribution[lang] = {
                'percentage': (count / total_chunks) * 100,
                'chunk_count': count
            }
    
    # Determine primary language
    primary_language = 'en'  # Default
    if language_distribution:
        primary_language = max(language_distribution.items(), key=lambda x: x[1]['percentage'])[0]
    
    return {
        'is_mixed': is_mixed,
        'primary_language': primary_language,
        'unique_languages': list(unique_languages),
        'segments': chunk_languages,
        'language_distribution': language_distribution,
        'total_chunks_analyzed': total_chunks,
        'confidence_threshold': 0.3
    }


def segment_mixed_language_content(text: str, target_languages: List[str] = None) -> Dict[str, Any]:
    """
    Segment mixed-language content into language-specific sections.
    
    Args:
        text: Text content to segment
        target_languages: List of target languages to segment for
        
    Returns:
        Dictionary with segmented content by language
    """
    if target_languages is None:
        target_languages = ['en', 'zh-CN', 'zh-TW', 'zh']
    
    mixed_analysis = detect_mixed_language_content(text)
    
    if not mixed_analysis['is_mixed']:
        # Not mixed content, return as single segment
        return {
            'is_segmented': False,
            'primary_language': mixed_analysis['primary_language'],
            'segments': [{
                'language': mixed_analysis['primary_language'],
                'content': text,
                'start_position': 0,
                'end_position': len(text),
                'confidence': 0.9
            }]
        }
    
    # Group consecutive chunks by language
    segments = []
    current_segment = None
    
    words = text.split()
    chunk_size = 500
    
    for segment_info in mixed_analysis['segments']:
        chunk_index = segment_info['chunk_index']
        language = segment_info['language']
        confidence = segment_info['confidence']
        
        # Calculate word positions for this chunk
        start_word = chunk_index * chunk_size
        end_word = min((chunk_index + 1) * chunk_size, len(words))
        chunk_text = ' '.join(words[start_word:end_word])
        
        # If same language as current segment, extend it
        if (current_segment and 
            current_segment['language'] == language and
            current_segment['end_word'] == start_word):
            
            current_segment['content'] += ' ' + chunk_text
            current_segment['end_word'] = end_word
            current_segment['confidence'] = (current_segment['confidence'] + confidence) / 2
        else:
            # Start new segment
            if current_segment:
                segments.append(current_segment)
            
            current_segment = {
                'language': language,
                'content': chunk_text,
                'start_word': start_word,
                'end_word': end_word,
                'confidence': confidence
            }
    
    # Add the last segment
    if current_segment:
        segments.append(current_segment)
    
    # Calculate character positions
    char_position = 0
    for segment in segments:
        segment['start_position'] = char_position
        segment['end_position'] = char_position + len(segment['content'])
        char_position = segment['end_position'] + 1  # +1 for space between segments
    
    return {
        'is_segmented': True,
        'primary_language': mixed_analysis['primary_language'],
        'segments': segments,
        'total_segments': len(segments),
        'language_distribution': mixed_analysis['language_distribution']
    }


def detect_language(text: str) -> str:
    """
    Simple language detection for text content.
    
    Args:
        text: Text content to analyze
        
    Returns:
        Language code ('en', 'zh', 'zh-CN', 'zh-TW', etc.)
    """
    if not text or not text.strip():
        return 'en'  # Default to English for empty text
    
    detector = YouTubeLanguageDetector()
    
    # Use the text content analysis directly
    scores = detector._analyze_text_content(text)
    best_language = max(scores.items(), key=lambda x: x[1])
    
    if best_language[1] > 0.3:  # Minimum confidence threshold
        detected_lang = best_language[0]
        
        # Convert to standard language codes
        if detected_lang == LanguageCode.ENGLISH:
            return 'en'
        elif detected_lang == LanguageCode.CHINESE_SIMPLIFIED:
            return 'zh-CN'
        elif detected_lang == LanguageCode.CHINESE_TRADITIONAL:
            return 'zh-TW'
        elif detected_lang == LanguageCode.CHINESE_GENERIC:
            return 'zh'
        elif detected_lang == LanguageCode.JAPANESE:
            return 'ja'
        elif detected_lang == LanguageCode.KOREAN:
            return 'ko'
    
    return 'en'  # Default fallback


def ensure_chinese_encoding(text: str) -> str:
    """
    Ensure proper encoding for Chinese text content.
    
    Args:
        text: Input text that may contain Chinese characters
        
    Returns:
        Properly encoded text string
    """
    if not text:
        return text
    
    try:
        # Ensure the text is properly encoded as UTF-8
        if isinstance(text, bytes):
            # If it's bytes, decode as UTF-8
            text = text.decode('utf-8', errors='replace')
        
        # Normalize Unicode characters (helpful for Chinese text)
        import unicodedata
        normalized_text = unicodedata.normalize('NFC', text)
        
        # Ensure we can encode back to UTF-8 (validates the text)
        normalized_text.encode('utf-8')
        
        return normalized_text
        
    except (UnicodeDecodeError, UnicodeEncodeError) as e:
        logger.warning(f"Encoding issue with Chinese text: {str(e)}")
        # Fallback: try to clean up the text
        try:
            # Replace problematic characters and return
            return text.encode('utf-8', errors='replace').decode('utf-8')
        except:
            # Last resort: return original text
            return text


def is_chinese_text(text: str) -> bool:
    """
    Check if text contains Chinese characters.
    
    Args:
        text: Text to check
        
    Returns:
        True if text contains Chinese characters, False otherwise
    """
    if not text:
        return False
    
    detector = YouTubeLanguageDetector()
    chinese_count = detector._count_chinese_characters(text)
    text_length = len(text)
    
    # Consider text Chinese if at least 10% of characters are Chinese
    return chinese_count > 0 and (chinese_count / text_length) >= 0.1


def get_chinese_optimized_prompts() -> Dict[str, str]:
    """
    Get optimized prompts for Chinese language content processing.
    
    Returns:
        Dictionary of prompt templates optimized for Chinese content
    """
    return {
        'summarization_system': """你是一个专业的中文内容总结专家。请根据以下指导原则创建一个专业的摘要：

指导原则：
- 专注于主要观点和关键见解
- 保持原文的语调和重要背景
- 使用清晰、简洁的语言
- 在相关时包含具体细节
- 确保摘要连贯且结构良好
- 保持中文表达的自然性和流畅性""",

        'keywords_system': """你是一个专业的中文关键词提取专家。请从提供的文本中提取最重要的关键词。

指导原则：
- 专注于最相关和重要的术语
- 包括单个词汇和短语
- 避免常见的停用词，除非它们是重要短语的一部分
- 考虑文本的上下文和领域
- 提供对搜索和分类有用的关键词
- 优先选择具有实际意义的中文词汇""",

        'timestamps_system': """你是一个专业的中文视频内容分析专家。请分析提供的转录文本，识别最重要的时间戳。

指导原则：
- 专注于关键见解、重要公告或转折点
- 为每个时间戳提供简要描述
- 按重要性评分（1-10分）
- 选择观众希望直接跳转到的时刻
- 确保描述准确反映该时间点的内容
- 使用自然的中文表达"""
    }


def optimize_chinese_content_for_llm(text: str, task_type: str = "summarization") -> Dict[str, str]:
    """
    Optimize Chinese content for LLM processing.
    
    Args:
        text: Chinese text content
        task_type: Type of task ("summarization", "keywords", "timestamps")
        
    Returns:
        Dictionary with optimized prompts and content
    """
    # Ensure proper encoding
    optimized_text = ensure_chinese_encoding(text)
    
    # Get Chinese-optimized prompts
    prompts = get_chinese_optimized_prompts()
    
    # Select appropriate system prompt
    system_prompt_key = f"{task_type}_system"
    system_prompt = prompts.get(system_prompt_key, prompts['summarization_system'])
    
    # Create task-specific user prompt
    if task_type == "summarization":
        user_prompt = f"""请总结以下中文文本内容：

{optimized_text}

请提供一个专业的摘要："""
    
    elif task_type == "keywords":
        user_prompt = f"""请从以下中文文本中提取关键词：

{optimized_text}

请提供最重要的关键词，每行一个，不要编号或项目符号："""
    
    elif task_type == "timestamps":
        user_prompt = f"""请分析以下中文转录文本并识别最重要的时间戳：

{optimized_text}

请提供格式如下的时间戳：
[XXX.X]s (评分: X/10) - 描述"""
    
    else:
        user_prompt = f"""请处理以下中文文本内容：

{optimized_text}"""
    
    return {
        'system_prompt': system_prompt,
        'user_prompt': user_prompt,
        'optimized_text': optimized_text
    }