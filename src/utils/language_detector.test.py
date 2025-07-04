"""
Unit tests for language detection functionality.

This module tests the YouTubeLanguageDetector class and related functions
to ensure accurate language detection for English and Chinese content.
"""

import unittest
from unittest.mock import Mock, patch
from src.utils.language_detector import (
    YouTubeLanguageDetector, 
    LanguageDetectionResult, 
    LanguageDetectionError,
    LanguageCode,
    detect_video_language,
    is_chinese_video,
    is_english_video,
    get_preferred_transcript_languages
)


class TestLanguageDetector(unittest.TestCase):
    """Test cases for YouTubeLanguageDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = YouTubeLanguageDetector()
        
        # Sample English metadata
        self.english_metadata = {
            'title': 'How to Learn Python Programming - Complete Tutorial',
            'description': 'Learn Python programming from scratch with this comprehensive tutorial.',
            'language': 'en',
            'defaultAudioLanguage': 'en-US',
            'channel_name': 'Programming Channel'
        }
        
        # Sample Chinese metadata
        self.chinese_metadata = {
            'title': 'Python编程教程 - 完整指南',
            'description': '从零开始学习Python编程的完整教程',
            'language': 'zh-CN',
            'defaultAudioLanguage': 'zh',
            'channel_name': '编程频道'
        }
        
        # Sample English transcript
        self.english_transcript = """
        Hello everyone and welcome to this tutorial. Today we're going to learn
        about Python programming. Python is a powerful programming language that
        is easy to learn and use. In this video, we will cover the basics of
        Python syntax, variables, functions, and more.
        """
        
        # Sample Chinese transcript
        self.chinese_transcript = """
        大家好，欢迎来到这个教程。今天我们要学习Python编程。Python是一种强大的编程语言，
        易于学习和使用。在这个视频中，我们将介绍Python语法、变量、函数等基础知识。
        """
    
    def test_detect_language_from_metadata_english(self):
        """Test English language detection from metadata."""
        result = self.detector.detect_language_from_metadata(self.english_metadata)
        
        self.assertEqual(result.detected_language, LanguageCode.ENGLISH)
        self.assertGreater(result.confidence_score, 0.7)
        self.assertEqual(result.detection_method, "metadata_analysis")
        self.assertIsNotNone(result.metadata_language)
        self.assertIsNotNone(result.title_language)
    
    def test_detect_language_from_metadata_chinese(self):
        """Test Chinese language detection from metadata."""
        result = self.detector.detect_language_from_metadata(self.chinese_metadata)
        
        self.assertIn(result.detected_language, [
            LanguageCode.CHINESE_SIMPLIFIED,
            LanguageCode.CHINESE_GENERIC
        ])
        self.assertGreater(result.confidence_score, 0.6)
        self.assertEqual(result.detection_method, "metadata_analysis")
        self.assertIsNotNone(result.metadata_language)
    
    def test_detect_language_from_transcript_english(self):
        """Test English language detection from transcript."""
        result = self.detector.detect_language_from_transcript(self.english_transcript)
        
        self.assertEqual(result.detected_language, LanguageCode.ENGLISH)
        self.assertGreater(result.confidence_score, 0.5)
        self.assertEqual(result.detection_method, "transcript_analysis")
        self.assertEqual(result.transcript_language, "en")
    
    def test_detect_language_from_transcript_chinese(self):
        """Test Chinese language detection from transcript."""
        result = self.detector.detect_language_from_transcript(self.chinese_transcript)
        
        self.assertIn(result.detected_language, [
            LanguageCode.CHINESE_SIMPLIFIED,
            LanguageCode.CHINESE_GENERIC
        ])
        self.assertGreater(result.confidence_score, 0.5)
        self.assertEqual(result.detection_method, "transcript_analysis")
    
    def test_detect_language_comprehensive_english(self):
        """Test comprehensive English language detection."""
        result = self.detector.detect_language_comprehensive(
            self.english_metadata, 
            self.english_transcript
        )
        
        self.assertEqual(result.detected_language, LanguageCode.ENGLISH)
        self.assertGreater(result.confidence_score, 0.6)
        self.assertEqual(result.detection_method, "comprehensive_analysis")
        self.assertIsNotNone(result.metadata_language)
        self.assertIsNotNone(result.transcript_language)
    
    def test_detect_language_comprehensive_chinese(self):
        """Test comprehensive Chinese language detection."""
        result = self.detector.detect_language_comprehensive(
            self.chinese_metadata, 
            self.chinese_transcript
        )
        
        self.assertIn(result.detected_language, [
            LanguageCode.CHINESE_SIMPLIFIED,
            LanguageCode.CHINESE_GENERIC
        ])
        self.assertGreater(result.confidence_score, 0.6)
        self.assertEqual(result.detection_method, "comprehensive_analysis")
    
    def test_detect_language_metadata_only(self):
        """Test language detection with metadata only."""
        result = self.detector.detect_language_comprehensive(self.english_metadata)
        
        self.assertEqual(result.detected_language, LanguageCode.ENGLISH)
        self.assertEqual(result.detection_method, "metadata_analysis")
        self.assertIsNone(result.transcript_language)
    
    def test_empty_transcript_error(self):
        """Test error handling for empty transcript."""
        with self.assertRaises(LanguageDetectionError):
            self.detector.detect_language_from_transcript("")
        
        with self.assertRaises(LanguageDetectionError):
            self.detector.detect_language_from_transcript("   ")
    
    def test_is_chinese_content(self):
        """Test Chinese content identification."""
        chinese_result = LanguageDetectionResult(
            detected_language=LanguageCode.CHINESE_SIMPLIFIED,
            confidence_score=0.8,
            detection_method="test"
        )
        
        english_result = LanguageDetectionResult(
            detected_language=LanguageCode.ENGLISH,
            confidence_score=0.8,
            detection_method="test"
        )
        
        self.assertTrue(self.detector.is_chinese_content(chinese_result))
        self.assertFalse(self.detector.is_chinese_content(english_result))
    
    def test_is_english_content(self):
        """Test English content identification."""
        english_result = LanguageDetectionResult(
            detected_language=LanguageCode.ENGLISH,
            confidence_score=0.8,
            detection_method="test"
        )
        
        chinese_result = LanguageDetectionResult(
            detected_language=LanguageCode.CHINESE_SIMPLIFIED,
            confidence_score=0.8,
            detection_method="test"
        )
        
        self.assertTrue(self.detector.is_english_content(english_result))
        self.assertFalse(self.detector.is_english_content(chinese_result))
    
    def test_get_preferred_transcript_languages_english(self):
        """Test preferred transcript languages for English content."""
        english_result = LanguageDetectionResult(
            detected_language=LanguageCode.ENGLISH,
            confidence_score=0.8,
            detection_method="test"
        )
        
        languages = self.detector.get_preferred_transcript_languages(english_result)
        
        self.assertIn('en', languages)
        self.assertEqual(languages[0], 'en')
        self.assertIn('en-US', languages)
    
    def test_get_preferred_transcript_languages_chinese(self):
        """Test preferred transcript languages for Chinese content."""
        chinese_result = LanguageDetectionResult(
            detected_language=LanguageCode.CHINESE_SIMPLIFIED,
            confidence_score=0.8,
            detection_method="test"
        )
        
        languages = self.detector.get_preferred_transcript_languages(chinese_result)
        
        self.assertIn('zh-CN', languages)
        self.assertEqual(languages[0], 'zh-CN')
        self.assertIn('zh', languages)
    
    def test_score_language_code(self):
        """Test language code scoring."""
        # Test English codes
        scores = self.detector._score_language_code('en', 1.0)
        self.assertEqual(scores[LanguageCode.ENGLISH], 1.0)
        
        scores = self.detector._score_language_code('en-US', 1.0)
        self.assertEqual(scores[LanguageCode.ENGLISH], 1.0)
        
        # Test Chinese codes
        scores = self.detector._score_language_code('zh-CN', 1.0)
        self.assertEqual(scores[LanguageCode.CHINESE_SIMPLIFIED], 1.0)
        
        scores = self.detector._score_language_code('zh-TW', 1.0)
        self.assertEqual(scores[LanguageCode.CHINESE_TRADITIONAL], 1.0)
        
        scores = self.detector._score_language_code('zh', 1.0)
        self.assertEqual(scores[LanguageCode.CHINESE_GENERIC], 1.0)
    
    def test_count_chinese_characters(self):
        """Test Chinese character counting."""
        # Test Chinese text
        chinese_text = "这是中文文本"
        count = self.detector._count_chinese_characters(chinese_text)
        self.assertEqual(count, 6)  # All characters are Chinese
        
        # Test mixed text
        mixed_text = "Hello 世界"
        count = self.detector._count_chinese_characters(mixed_text)
        self.assertEqual(count, 2)  # Only "世界" are Chinese
        
        # Test English text
        english_text = "Hello World"
        count = self.detector._count_chinese_characters(english_text)
        self.assertEqual(count, 0)  # No Chinese characters
    
    def test_analyze_text_content_english(self):
        """Test text content analysis for English."""
        scores = self.detector._analyze_text_content(self.english_transcript)
        
        self.assertGreater(scores[LanguageCode.ENGLISH], 0.3)
        self.assertLess(scores[LanguageCode.CHINESE_GENERIC], 0.1)
    
    def test_analyze_text_content_chinese(self):
        """Test text content analysis for Chinese."""
        scores = self.detector._analyze_text_content(self.chinese_transcript)
        
        self.assertGreater(scores[LanguageCode.CHINESE_GENERIC], 0.3)
        self.assertLess(scores[LanguageCode.ENGLISH], 0.1)
    
    def test_mixed_language_content(self):
        """Test detection for mixed language content."""
        mixed_metadata = {
            'title': 'Python Programming 编程教程',
            'description': 'Learn Python programming 学习编程',
            'language': 'en',
            'defaultAudioLanguage': 'zh'
        }
        
        result = self.detector.detect_language_from_metadata(mixed_metadata)
        
        # Should detect based on metadata language preference
        self.assertIsNotNone(result.detected_language)
        self.assertGreater(result.confidence_score, 0.3)
    
    def test_unknown_language_fallback(self):
        """Test fallback to unknown language for unclear content."""
        unclear_metadata = {
            'title': 'abc xyz 123',
            'description': 'test test test',
            'language': None,
            'defaultAudioLanguage': None
        }
        
        result = self.detector.detect_language_from_metadata(unclear_metadata)
        
        # Should have low confidence or unknown language
        self.assertTrue(
            result.detected_language == LanguageCode.UNKNOWN or 
            result.confidence_score < 0.5
        )


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.english_metadata = {
            'title': 'English Tutorial',
            'language': 'en'
        }
        
        self.chinese_metadata = {
            'title': '中文教程',
            'language': 'zh-CN'
        }
    
    def test_detect_video_language(self):
        """Test detect_video_language convenience function."""
        result = detect_video_language(self.english_metadata)
        
        self.assertIsInstance(result, LanguageDetectionResult)
        self.assertEqual(result.detected_language, LanguageCode.ENGLISH)
    
    def test_is_chinese_video(self):
        """Test is_chinese_video convenience function."""
        self.assertFalse(is_chinese_video(self.english_metadata))
        self.assertTrue(is_chinese_video(self.chinese_metadata))
    
    def test_is_english_video(self):
        """Test is_english_video convenience function."""
        self.assertTrue(is_english_video(self.english_metadata))
        self.assertFalse(is_english_video(self.chinese_metadata))
    
    def test_get_preferred_transcript_languages_function(self):
        """Test get_preferred_transcript_languages convenience function."""
        languages = get_preferred_transcript_languages(self.english_metadata)
        
        self.assertIsInstance(languages, list)
        self.assertIn('en', languages)
        self.assertEqual(languages[0], 'en')


class TestLanguageDetectionResult(unittest.TestCase):
    """Test cases for LanguageDetectionResult dataclass."""
    
    def test_language_detection_result_creation(self):
        """Test LanguageDetectionResult creation and auto-fields."""
        result = LanguageDetectionResult(
            detected_language=LanguageCode.ENGLISH,
            confidence_score=0.85,
            detection_method="test"
        )
        
        self.assertEqual(result.detected_language, LanguageCode.ENGLISH)
        self.assertEqual(result.confidence_score, 0.85)
        self.assertEqual(result.detection_method, "test")
        self.assertIsNotNone(result.detection_timestamp)
        self.assertEqual(result.alternative_languages, [])
    
    def test_language_detection_result_with_alternatives(self):
        """Test LanguageDetectionResult with alternative languages."""
        alternatives = [(LanguageCode.CHINESE_GENERIC, 0.3), (LanguageCode.JAPANESE, 0.1)]
        
        result = LanguageDetectionResult(
            detected_language=LanguageCode.ENGLISH,
            confidence_score=0.85,
            detection_method="test",
            alternative_languages=alternatives
        )
        
        self.assertEqual(len(result.alternative_languages), 2)
        self.assertEqual(result.alternative_languages[0][0], LanguageCode.CHINESE_GENERIC)


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    unittest.main()