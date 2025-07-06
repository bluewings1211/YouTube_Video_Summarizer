"""
Comprehensive unit tests for language detection and processing logic.

This module tests all aspects of language detection including:
- Automatic language detection from metadata and transcripts
- Language-specific processing workflows
- Mixed-language content handling
- Chinese content preservation
- Configuration options
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Import the modules to test
from src.utils.language_detector import (
    YouTubeLanguageDetector, LanguageDetectionResult, LanguageCode,
    detect_video_language, is_chinese_video, is_english_video,
    get_preferred_transcript_languages, optimize_chinese_content_for_llm,
    ensure_chinese_encoding, detect_mixed_language_content,
    segment_mixed_language_content, detect_language, is_chinese_text,
    get_chinese_optimized_prompts
)
from src.flow import (
    LanguageProcessingConfig, create_chinese_optimized_config,
    create_english_optimized_config, create_multilingual_config,
    YouTubeSummarizerFlow
)


class TestLanguageDetection(unittest.TestCase):
    """Test basic language detection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = YouTubeLanguageDetector()
        
        # Sample video metadata for testing
        self.english_metadata = {
            'title': 'How to Build a React Application',
            'description': 'Learn how to create a modern React app with best practices',
            'language': 'en',
            'defaultAudioLanguage': 'en-US'
        }
        
        self.chinese_metadata = {
            'title': '如何学习Python编程',
            'description': '这是一个关于Python编程的教程，适合初学者学习',
            'language': 'zh-CN',
            'defaultAudioLanguage': 'zh-CN'
        }
        
        self.mixed_metadata = {
            'title': 'Python Tutorial 学习指南',
            'description': 'Learn Python programming 学习Python编程',
            'language': 'en',
            'defaultAudioLanguage': 'en'
        }
        
        # Sample transcript texts
        self.english_transcript = """
        Hello everyone and welcome to this tutorial. Today we're going to learn about 
        Python programming. Python is a powerful and versatile programming language 
        that is great for beginners. We'll cover variables, functions, and control flow.
        """
        
        self.chinese_transcript = """
        大家好，欢迎来到这个教程。今天我们要学习Python编程。Python是一种强大且
        多功能的编程语言，非常适合初学者。我们将涵盖变量、函数和控制流程。
        """
        
        self.mixed_transcript = """
        Hello everyone, 大家好! Today we're going to learn about Python programming.
        Python是一种强大的编程语言。We'll start with basic concepts 我们从基础概念开始。
        Let's begin our journey into programming 让我们开始编程之旅吧!
        """
    
    def test_english_language_detection_from_metadata(self):
        """Test English language detection from metadata."""
        result = self.detector.detect_language_from_metadata(self.english_metadata)
        
        self.assertIsInstance(result, LanguageDetectionResult)
        self.assertEqual(result.detected_language, LanguageCode.ENGLISH)
        self.assertGreater(result.confidence_score, 0.5)
        self.assertEqual(result.detection_method, "metadata_analysis")
        self.assertEqual(result.metadata_language, "en-US")
    
    def test_chinese_language_detection_from_metadata(self):
        """Test Chinese language detection from metadata."""
        result = self.detector.detect_language_from_metadata(self.chinese_metadata)
        
        self.assertIsInstance(result, LanguageDetectionResult)
        self.assertIn(result.detected_language, [
            LanguageCode.CHINESE_SIMPLIFIED, 
            LanguageCode.CHINESE_GENERIC
        ])
        self.assertGreater(result.confidence_score, 0.5)
        self.assertEqual(result.detection_method, "metadata_analysis")
    
    def test_english_language_detection_from_transcript(self):
        """Test English language detection from transcript."""
        result = self.detector.detect_language_from_transcript(self.english_transcript)
        
        self.assertIsInstance(result, LanguageDetectionResult)
        self.assertEqual(result.detected_language, LanguageCode.ENGLISH)
        self.assertGreater(result.confidence_score, 0.5)
        self.assertEqual(result.detection_method, "transcript_analysis")
    
    def test_chinese_language_detection_from_transcript(self):
        """Test Chinese language detection from transcript."""
        result = self.detector.detect_language_from_transcript(self.chinese_transcript)
        
        self.assertIsInstance(result, LanguageDetectionResult)
        self.assertIn(result.detected_language, [
            LanguageCode.CHINESE_SIMPLIFIED,
            LanguageCode.CHINESE_TRADITIONAL,
            LanguageCode.CHINESE_GENERIC
        ])
        self.assertGreater(result.confidence_score, 0.5)
        self.assertEqual(result.detection_method, "transcript_analysis")
    
    def test_comprehensive_language_detection(self):
        """Test comprehensive language detection combining metadata and transcript."""
        result = self.detector.detect_language_comprehensive(
            self.english_metadata, 
            self.english_transcript
        )
        
        self.assertIsInstance(result, LanguageDetectionResult)
        self.assertEqual(result.detected_language, LanguageCode.ENGLISH)
        self.assertGreater(result.confidence_score, 0.5)
        self.assertEqual(result.detection_method, "comprehensive_analysis")
        self.assertIsNotNone(result.metadata_language)
        self.assertIsNotNone(result.transcript_language)
    
    def test_language_detection_with_disagreement(self):
        """Test language detection when metadata and transcript disagree."""
        # Use Chinese metadata with English transcript
        result = self.detector.detect_language_comprehensive(
            self.chinese_metadata,
            self.english_transcript
        )
        
        self.assertIsInstance(result, LanguageDetectionResult)
        # Should favor metadata when there's disagreement but not high confidence
        self.assertIn(result.detected_language, [
            LanguageCode.CHINESE_SIMPLIFIED,
            LanguageCode.CHINESE_GENERIC,
            LanguageCode.ENGLISH  # May choose English if transcript confidence is very high
        ])
    
    def test_language_utility_functions(self):
        """Test utility functions for language checking."""
        # Test is_chinese_content
        chinese_result = LanguageDetectionResult(
            detected_language=LanguageCode.CHINESE_SIMPLIFIED,
            confidence_score=0.8,
            detection_method="test"
        )
        self.assertTrue(self.detector.is_chinese_content(chinese_result))
        
        # Test is_english_content
        english_result = LanguageDetectionResult(
            detected_language=LanguageCode.ENGLISH,
            confidence_score=0.8,
            detection_method="test"
        )
        self.assertTrue(self.detector.is_english_content(english_result))
        self.assertFalse(self.detector.is_chinese_content(english_result))
    
    def test_preferred_transcript_languages(self):
        """Test getting preferred transcript languages."""
        english_result = LanguageDetectionResult(
            detected_language=LanguageCode.ENGLISH,
            confidence_score=0.8,
            detection_method="test"
        )
        
        languages = self.detector.get_preferred_transcript_languages(english_result)
        self.assertIsInstance(languages, list)
        self.assertIn('en', languages)
        self.assertIn('en-US', languages)
        
        chinese_result = LanguageDetectionResult(
            detected_language=LanguageCode.CHINESE_SIMPLIFIED,
            confidence_score=0.8,
            detection_method="test"
        )
        
        languages = self.detector.get_preferred_transcript_languages(chinese_result)
        self.assertIsInstance(languages, list)
        self.assertIn('zh-CN', languages)
        self.assertIn('zh', languages)


class TestMixedLanguageDetection(unittest.TestCase):
    """Test mixed-language content detection and handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mixed_text = """
        Hello everyone, 大家好! Today we're going to learn about Python programming.
        Python是一种强大的编程语言。We'll start with basic concepts 我们从基础概念开始。
        Let's begin our journey into programming 让我们开始编程之旅吧!
        """
        
        self.english_text = """
        Hello everyone and welcome to this tutorial. Today we're going to learn about
        Python programming. Python is a powerful programming language.
        """
        
        self.chinese_text = """
        大家好，欢迎来到这个教程。今天我们要学习Python编程。Python是一种强大且
        多功能的编程语言，非常适合初学者。
        """
    
    def test_mixed_language_content_detection(self):
        """Test detection of mixed-language content."""
        result = detect_mixed_language_content(self.mixed_text)
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result['is_mixed'])
        self.assertGreater(len(result['unique_languages']), 1)
        self.assertIn('language_distribution', result)
        self.assertGreater(result['total_chunks_analyzed'], 0)
    
    def test_single_language_content_detection(self):
        """Test detection of single-language content."""
        english_result = detect_mixed_language_content(self.english_text)
        self.assertFalse(english_result['is_mixed'])
        self.assertEqual(len(english_result['unique_languages']), 1)
        
        chinese_result = detect_mixed_language_content(self.chinese_text)
        self.assertFalse(chinese_result['is_mixed'])
        self.assertEqual(len(chinese_result['unique_languages']), 1)
    
    def test_language_segmentation(self):
        """Test segmentation of mixed-language content."""
        result = segment_mixed_language_content(self.mixed_text)
        
        self.assertIsInstance(result, dict)
        if result['is_segmented']:
            self.assertGreater(len(result['segments']), 1)
            self.assertIn('primary_language', result)
            
            # Check segment structure
            for segment in result['segments']:
                self.assertIn('language', segment)
                self.assertIn('content', segment)
                self.assertIn('confidence', segment)
    
    def test_segmentation_of_single_language(self):
        """Test segmentation of single-language content."""
        result = segment_mixed_language_content(self.english_text)
        
        self.assertIsInstance(result, dict)
        self.assertFalse(result['is_segmented'])
        self.assertEqual(len(result['segments']), 1)
        self.assertEqual(result['segments'][0]['content'], self.english_text)


class TestChineseContentProcessing(unittest.TestCase):
    """Test Chinese-specific content processing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.chinese_text = "这是一个关于Python编程的教程"
        self.english_text = "This is a tutorial about Python programming"
        self.mixed_text = "This is 中文 mixed content"
    
    def test_chinese_encoding_preservation(self):
        """Test Chinese character encoding preservation."""
        preserved = ensure_chinese_encoding(self.chinese_text)
        
        self.assertIsInstance(preserved, str)
        self.assertEqual(preserved, self.chinese_text)  # Should remain the same
        
        # Test with problematic encoding scenarios
        byte_text = self.chinese_text.encode('utf-8')
        preserved_from_bytes = ensure_chinese_encoding(byte_text)
        self.assertIsInstance(preserved_from_bytes, str)
        self.assertEqual(preserved_from_bytes, self.chinese_text)
    
    def test_chinese_text_detection(self):
        """Test Chinese text detection utility."""
        self.assertTrue(is_chinese_text(self.chinese_text))
        self.assertFalse(is_chinese_text(self.english_text))
        self.assertTrue(is_chinese_text(self.mixed_text))  # Should detect Chinese presence
    
    def test_chinese_optimized_prompts(self):
        """Test Chinese-optimized prompt generation."""
        prompts = get_chinese_optimized_prompts()
        
        self.assertIsInstance(prompts, dict)
        self.assertIn('summarization_system', prompts)
        self.assertIn('keywords_system', prompts)
        self.assertIn('timestamps_system', prompts)
        
        # Check that prompts contain Chinese text
        for prompt in prompts.values():
            self.assertTrue(is_chinese_text(prompt))
    
    def test_chinese_content_optimization(self):
        """Test Chinese content optimization for LLM processing."""
        result = optimize_chinese_content_for_llm(self.chinese_text, "summarization")
        
        self.assertIsInstance(result, dict)
        self.assertIn('system_prompt', result)
        self.assertIn('user_prompt', result)
        self.assertIn('optimized_text', result)
        
        # Check that system prompt is in Chinese
        self.assertTrue(is_chinese_text(result['system_prompt']))
        
        # Test different task types
        keywords_result = optimize_chinese_content_for_llm(self.chinese_text, "keywords")
        self.assertIn('system_prompt', keywords_result)
        self.assertTrue(is_chinese_text(keywords_result['system_prompt']))
        
        timestamps_result = optimize_chinese_content_for_llm(self.chinese_text, "timestamps")
        self.assertIn('system_prompt', timestamps_result)
        self.assertTrue(is_chinese_text(timestamps_result['system_prompt']))


class TestLanguageProcessingConfiguration(unittest.TestCase):
    """Test language processing configuration options."""
    
    def test_default_language_processing_config(self):
        """Test default language processing configuration."""
        config = LanguageProcessingConfig()
        
        self.assertTrue(config.enable_language_detection)
        self.assertTrue(config.enable_chinese_optimization)
        self.assertEqual(config.default_language, "en")
        self.assertIn("en", config.preferred_languages)
        self.assertIn("zh-CN", config.preferred_languages)
        self.assertTrue(config.chinese_prompt_optimization)
        self.assertEqual(config.language_confidence_threshold, 0.5)
        self.assertEqual(config.mixed_language_handling, "primary")
        self.assertTrue(config.preserve_chinese_encoding)
        self.assertTrue(config.enable_transcript_language_preference)
    
    def test_chinese_optimized_config(self):
        """Test Chinese-optimized workflow configuration."""
        config = create_chinese_optimized_config()
        
        self.assertTrue(config.language_processing.enable_chinese_optimization)
        self.assertEqual(config.language_processing.default_language, "zh-CN")
        self.assertEqual(config.language_processing.preferred_languages[0], "zh-CN")
        self.assertLess(
            config.language_processing.language_confidence_threshold, 
            0.5  # Lower threshold for better Chinese detection
        )
        self.assertTrue(config.language_processing.preserve_chinese_encoding)
    
    def test_english_optimized_config(self):
        """Test English-optimized workflow configuration."""
        config = create_english_optimized_config()
        
        self.assertFalse(config.language_processing.enable_chinese_optimization)
        self.assertEqual(config.language_processing.default_language, "en")
        self.assertEqual(config.language_processing.preferred_languages[0], "en")
        self.assertFalse(config.language_processing.chinese_prompt_optimization)
        self.assertFalse(config.language_processing.preserve_chinese_encoding)
    
    def test_multilingual_config(self):
        """Test multilingual workflow configuration."""
        config = create_multilingual_config()
        
        self.assertTrue(config.language_processing.enable_language_detection)
        self.assertTrue(config.language_processing.enable_chinese_optimization)
        self.assertEqual(config.language_processing.mixed_language_handling, "segment")
        self.assertGreater(len(config.language_processing.preferred_languages), 4)
        self.assertIn("en", config.language_processing.preferred_languages)
        self.assertIn("zh-CN", config.language_processing.preferred_languages)
        self.assertIn("ja", config.language_processing.preferred_languages)
    
    def test_custom_language_config_overrides(self):
        """Test custom language configuration overrides."""
        custom_config = create_chinese_optimized_config(
            language_confidence_threshold=0.8,
            mixed_language_handling="dual",
            preferred_languages=["zh-TW", "zh-CN", "en"]
        )
        
        self.assertEqual(custom_config.language_processing.language_confidence_threshold, 0.8)
        self.assertEqual(custom_config.language_processing.mixed_language_handling, "dual")
        self.assertEqual(custom_config.language_processing.preferred_languages[0], "zh-TW")


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions for language detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.english_metadata = {
            'title': 'Python Tutorial',
            'description': 'Learn Python programming',
            'language': 'en'
        }
        
        self.chinese_metadata = {
            'title': 'Python教程',
            'description': '学习Python编程',
            'language': 'zh-CN'
        }
        
        self.english_transcript = "Hello everyone, welcome to this Python tutorial."
        self.chinese_transcript = "大家好，欢迎来到这个Python教程。"
    
    def test_detect_video_language_function(self):
        """Test the detect_video_language convenience function."""
        result = detect_video_language(self.english_metadata, self.english_transcript)
        
        self.assertIsInstance(result, LanguageDetectionResult)
        self.assertEqual(result.detected_language, LanguageCode.ENGLISH)
        
        result = detect_video_language(self.chinese_metadata, self.chinese_transcript)
        self.assertIn(result.detected_language, [
            LanguageCode.CHINESE_SIMPLIFIED,
            LanguageCode.CHINESE_GENERIC
        ])
    
    def test_is_chinese_video_function(self):
        """Test the is_chinese_video convenience function."""
        is_chinese = is_chinese_video(self.chinese_metadata, self.chinese_transcript)
        self.assertTrue(is_chinese)
        
        is_chinese = is_chinese_video(self.english_metadata, self.english_transcript)
        self.assertFalse(is_chinese)
    
    def test_is_english_video_function(self):
        """Test the is_english_video convenience function."""
        is_english = is_english_video(self.english_metadata, self.english_transcript)
        self.assertTrue(is_english)
        
        is_english = is_english_video(self.chinese_metadata, self.chinese_transcript)
        self.assertFalse(is_english)
    
    def test_get_preferred_transcript_languages_function(self):
        """Test the get_preferred_transcript_languages convenience function."""
        languages = get_preferred_transcript_languages(self.english_metadata)
        self.assertIsInstance(languages, list)
        self.assertIn('en', languages)
        
        languages = get_preferred_transcript_languages(self.chinese_metadata)
        self.assertIsInstance(languages, list)
        self.assertIn('zh-CN', languages)
    
    def test_detect_language_simple_function(self):
        """Test the simple detect_language function."""
        detected = detect_language(self.english_transcript)
        self.assertEqual(detected, 'en')
        
        detected = detect_language(self.chinese_transcript)
        self.assertIn(detected, ['zh', 'zh-CN', 'zh-TW'])
        
        # Test empty text
        detected = detect_language("")
        self.assertEqual(detected, 'en')  # Default fallback


class TestWorkflowIntegration(unittest.TestCase):
    """Test language detection integration with workflow."""
    
    @patch('src.utils.language_detector.YouTubeLanguageDetector')
    def test_workflow_language_detection_integration(self, mock_detector_class):
        """Test language detection integration in workflow."""
        # Mock the language detector
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        
        mock_result = LanguageDetectionResult(
            detected_language=LanguageCode.CHINESE_SIMPLIFIED,
            confidence_score=0.8,
            detection_method="test"
        )
        mock_detector.detect_language_comprehensive.return_value = mock_result
        
        # Create a workflow with language detection enabled
        config = create_chinese_optimized_config()
        workflow = YouTubeSummarizerFlow(config)
        
        # Verify language detector was initialized
        self.assertIsNotNone(workflow.language_detector)
        
        # Test language detection method
        video_metadata = {'title': 'Test', 'language': 'zh-CN'}
        result = workflow._detect_video_language(video_metadata, "测试文本")
        
        self.assertEqual(result.detected_language, LanguageCode.CHINESE_SIMPLIFIED)
        self.assertEqual(result.confidence_score, 0.8)
    
    def test_language_specific_processing_application(self):
        """Test application of language-specific processing."""
        config = create_chinese_optimized_config()
        workflow = YouTubeSummarizerFlow(config)
        
        # Mock a Chinese detection result
        workflow.detected_language_result = LanguageDetectionResult(
            detected_language=LanguageCode.CHINESE_SIMPLIFIED,
            confidence_score=0.8,
            detection_method="test"
        )
        
        # Test Chinese processing
        result = workflow._apply_language_specific_processing("测试文本", "summarization")
        
        self.assertIsInstance(result, dict)
        self.assertIn('system_prompt', result)
        self.assertIn('user_prompt', result)
        self.assertIn('optimized_text', result)
        
        # Should contain Chinese text for Chinese optimization
        self.assertTrue(is_chinese_text(result['system_prompt']))
    
    def test_mixed_language_content_handling(self):
        """Test mixed-language content handling in workflow."""
        config = create_multilingual_config()
        workflow = YouTubeSummarizerFlow(config)
        
        # Mock mixed language detection
        workflow.detected_language_result = LanguageDetectionResult(
            detected_language=LanguageCode.ENGLISH,
            confidence_score=0.6,
            detection_method="test"
        )
        
        # Mock store with mixed language analysis
        workflow.store = {
            'language_detection_result': {
                'mixed_language_analysis': {
                    'is_mixed': True,
                    'primary_language': 'en',
                    'language_distribution': {
                        'en': {'percentage': 60, 'chunk_count': 3},
                        'zh': {'percentage': 40, 'chunk_count': 2}
                    }
                }
            }
        }
        
        mixed_text = "Hello 你好 this is mixed content"
        result = workflow._handle_mixed_language_content(mixed_text)
        
        self.assertIsInstance(result, dict)
        self.assertIn('strategy', result)
        self.assertTrue(result.get('is_mixed', False))


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)