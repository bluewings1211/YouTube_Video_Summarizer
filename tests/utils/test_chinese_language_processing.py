"""
Unit tests for Chinese language processing functionality.

This module tests Chinese language support including:
- Language detection
- Text encoding handling
- Chinese-optimized prompts
- Content optimization for LLMs
"""

import pytest
import unicodedata
from unittest.mock import Mock, patch

from src.utils.language_detector import (
    detect_language, ensure_chinese_encoding, is_chinese_text,
    get_chinese_optimized_prompts, optimize_chinese_content_for_llm,
    YouTubeLanguageDetector, LanguageCode
)


class TestChineseLanguageDetection:
    """Test Chinese language detection functionality."""
    
    def test_detect_language_english(self):
        """Test language detection for English text."""
        english_text = "This is a test in English language"
        result = detect_language(english_text)
        assert result == 'en'
    
    def test_detect_language_chinese_simplified(self):
        """Test language detection for Simplified Chinese text."""
        chinese_text = "这是一个中文测试文本"
        result = detect_language(chinese_text)
        assert result in ['zh', 'zh-CN']
    
    def test_detect_language_chinese_traditional(self):
        """Test language detection for Traditional Chinese text."""
        traditional_text = "這是一個繁體中文測試文本"
        result = detect_language(traditional_text)
        assert result in ['zh', 'zh-CN', 'zh-TW']
    
    def test_detect_language_mixed_content(self):
        """Test language detection for mixed Chinese-English content."""
        mixed_text = "这是中文 and this is English mixed content"
        result = detect_language(mixed_text)
        # Should detect based on dominant language
        assert result in ['zh', 'zh-CN', 'en']
    
    def test_detect_language_empty_text(self):
        """Test language detection for empty text."""
        result = detect_language("")
        assert result == 'en'  # Default fallback
        
        result = detect_language(None)
        assert result == 'en'  # Default fallback
    
    def test_detect_language_japanese(self):
        """Test language detection for Japanese text."""
        japanese_text = "これは日本語のテストです"
        result = detect_language(japanese_text)
        assert result == 'ja'
    
    def test_detect_language_korean(self):
        """Test language detection for Korean text."""
        korean_text = "이것은 한국어 테스트입니다"
        result = detect_language(korean_text)
        assert result == 'ko'


class TestChineseTextValidation:
    """Test Chinese text validation and identification."""
    
    def test_is_chinese_text_pure_chinese(self):
        """Test Chinese text identification for pure Chinese content."""
        chinese_text = "这是一个纯中文文本内容"
        assert is_chinese_text(chinese_text) is True
    
    def test_is_chinese_text_mixed_content(self):
        """Test Chinese text identification for mixed content."""
        mixed_text = "这是中文 and English mixed content"
        # Should return True if at least 10% are Chinese characters
        assert is_chinese_text(mixed_text) is True
    
    def test_is_chinese_text_english_only(self):
        """Test Chinese text identification for English-only content."""
        english_text = "This is pure English content"
        assert is_chinese_text(english_text) is False
    
    def test_is_chinese_text_minimal_chinese(self):
        """Test Chinese text identification with minimal Chinese content."""
        minimal_chinese = "Test with one 中 character"
        # Less than 10% Chinese characters
        assert is_chinese_text(minimal_chinese) is False
    
    def test_is_chinese_text_empty(self):
        """Test Chinese text identification for empty text."""
        assert is_chinese_text("") is False
        assert is_chinese_text(None) is False
    
    def test_is_chinese_text_traditional_chinese(self):
        """Test Chinese text identification for Traditional Chinese."""
        traditional_text = "這是繁體中文內容"
        assert is_chinese_text(traditional_text) is True


class TestChineseEncodingHandling:
    """Test Chinese text encoding and normalization."""
    
    def test_ensure_chinese_encoding_normal_text(self):
        """Test encoding handling for normal Chinese text."""
        chinese_text = "这是正常的中文文本"
        result = ensure_chinese_encoding(chinese_text)
        assert result == chinese_text
        assert isinstance(result, str)
    
    def test_ensure_chinese_encoding_unicode_normalization(self):
        """Test Unicode normalization for Chinese text."""
        # Create text with non-normalized Unicode
        text_with_combining = "中文测试"  # May contain combining characters
        result = ensure_chinese_encoding(text_with_combining)
        
        # Should be NFC normalized
        expected = unicodedata.normalize('NFC', text_with_combining)
        assert result == expected
    
    def test_ensure_chinese_encoding_bytes_input(self):
        """Test encoding handling for bytes input."""
        chinese_text = "中文字节测试"
        byte_input = chinese_text.encode('utf-8')
        
        result = ensure_chinese_encoding(byte_input)
        assert result == chinese_text
        assert isinstance(result, str)
    
    def test_ensure_chinese_encoding_empty_text(self):
        """Test encoding handling for empty/None text."""
        assert ensure_chinese_encoding("") == ""
        assert ensure_chinese_encoding(None) is None
    
    def test_ensure_chinese_encoding_error_handling(self):
        """Test encoding error handling."""
        # This should not raise an exception
        with patch('src.utils.language_detector.unicodedata.normalize') as mock_normalize:
            mock_normalize.side_effect = UnicodeError("Test error")
            
            result = ensure_chinese_encoding("测试文本")
            # Should fallback gracefully
            assert isinstance(result, str)


class TestChineseOptimizedPrompts:
    """Test Chinese-optimized prompt generation."""
    
    def test_get_chinese_optimized_prompts_structure(self):
        """Test the structure of Chinese optimized prompts."""
        prompts = get_chinese_optimized_prompts()
        
        assert isinstance(prompts, dict)
        assert 'summarization_system' in prompts
        assert 'keywords_system' in prompts
        assert 'timestamps_system' in prompts
        
        # Check that prompts are in Chinese
        for prompt in prompts.values():
            assert is_chinese_text(prompt)
    
    def test_chinese_summarization_prompt_content(self):
        """Test Chinese summarization prompt content."""
        prompts = get_chinese_optimized_prompts()
        summarization_prompt = prompts['summarization_system']
        
        # Should contain key Chinese terms
        assert '专业' in summarization_prompt  # Professional
        assert '总结' in summarization_prompt  # Summarize
        assert '中文' in summarization_prompt  # Chinese
        assert '指导原则' in summarization_prompt  # Guidelines
    
    def test_chinese_keywords_prompt_content(self):
        """Test Chinese keywords prompt content."""
        prompts = get_chinese_optimized_prompts()
        keywords_prompt = prompts['keywords_system']
        
        # Should contain key Chinese terms
        assert '关键词' in keywords_prompt  # Keywords
        assert '提取' in keywords_prompt  # Extract
        assert '重要' in keywords_prompt  # Important
    
    def test_chinese_timestamps_prompt_content(self):
        """Test Chinese timestamps prompt content."""
        prompts = get_chinese_optimized_prompts()
        timestamps_prompt = prompts['timestamps_system']
        
        # Should contain key Chinese terms
        assert '时间戳' in timestamps_prompt  # Timestamps
        assert '分析' in timestamps_prompt  # Analyze
        assert '重要' in timestamps_prompt  # Important


class TestChineseContentOptimization:
    """Test Chinese content optimization for LLM processing."""
    
    def test_optimize_chinese_content_summarization(self):
        """Test Chinese content optimization for summarization."""
        chinese_text = "这是一个需要总结的中文文本内容"
        
        result = optimize_chinese_content_for_llm(chinese_text, "summarization")
        
        assert 'system_prompt' in result
        assert 'user_prompt' in result
        assert 'optimized_text' in result
        
        # Check that system prompt is in Chinese
        assert is_chinese_text(result['system_prompt'])
        
        # Check that user prompt contains the text
        assert chinese_text in result['user_prompt']
        assert '总结' in result['user_prompt']  # Summarize
    
    def test_optimize_chinese_content_keywords(self):
        """Test Chinese content optimization for keyword extraction."""
        chinese_text = "这是一个需要提取关键词的中文文本"
        
        result = optimize_chinese_content_for_llm(chinese_text, "keywords")
        
        assert 'system_prompt' in result
        assert 'user_prompt' in result
        
        # Check for keyword-specific terms
        assert '关键词' in result['user_prompt']  # Keywords
        assert '提取' in result['user_prompt']  # Extract
    
    def test_optimize_chinese_content_timestamps(self):
        """Test Chinese content optimization for timestamp generation."""
        chinese_text = "这是一个需要生成时间戳的中文转录文本"
        
        result = optimize_chinese_content_for_llm(chinese_text, "timestamps")
        
        assert 'system_prompt' in result
        assert 'user_prompt' in result
        
        # Check for timestamp-specific terms
        assert '时间戳' in result['user_prompt']  # Timestamps
        assert '转录' in result['user_prompt']  # Transcript
        assert '评分' in result['user_prompt']  # Rating
    
    def test_optimize_chinese_content_unknown_task(self):
        """Test Chinese content optimization for unknown task type."""
        chinese_text = "这是一个测试文本"
        
        result = optimize_chinese_content_for_llm(chinese_text, "unknown_task")
        
        assert 'system_prompt' in result
        assert 'user_prompt' in result
        
        # Should fallback to general processing
        assert '处理' in result['user_prompt']  # Process
    
    def test_optimize_chinese_content_encoding(self):
        """Test that optimization includes proper encoding."""
        chinese_text = "测试编码处理"
        
        with patch('src.utils.language_detector.ensure_chinese_encoding') as mock_encoding:
            mock_encoding.return_value = "优化后的文本"
            
            result = optimize_chinese_content_for_llm(chinese_text, "summarization")
            
            mock_encoding.assert_called_once_with(chinese_text)
            assert result['optimized_text'] == "优化后的文本"


class TestYouTubeLanguageDetectorChinese:
    """Test YouTube language detector specifically for Chinese content."""
    
    @pytest.fixture
    def detector(self):
        """Create YouTube language detector instance."""
        return YouTubeLanguageDetector()
    
    def test_chinese_character_counting(self, detector):
        """Test Chinese character counting functionality."""
        chinese_text = "这是中文测试"
        english_text = "This is English"
        mixed_text = "这是 mixed 内容"
        
        chinese_count = detector._count_chinese_characters(chinese_text)
        english_count = detector._count_chinese_characters(english_text)
        mixed_count = detector._count_chinese_characters(mixed_text)
        
        assert chinese_count == 5  # 5 Chinese characters
        assert english_count == 0  # No Chinese characters
        assert mixed_count == 3  # 3 Chinese characters in mixed text
    
    def test_chinese_text_content_analysis(self, detector):
        """Test text content analysis for Chinese text."""
        chinese_text = "这是一个中文视频教程的内容"
        
        scores = detector._analyze_text_content(chinese_text)
        
        # Should have high scores for Chinese language codes
        assert scores[LanguageCode.CHINESE_GENERIC] > 0.3
        assert scores[LanguageCode.CHINESE_SIMPLIFIED] > 0.3
        
        # Should have low scores for English
        assert scores[LanguageCode.ENGLISH] < 0.3
    
    def test_chinese_title_pattern_detection(self, detector):
        """Test Chinese title pattern detection."""
        chinese_title = "Python编程教程 - 完整指南"
        
        scores = detector._analyze_text_content(chinese_title)
        
        # Should detect Chinese content and title patterns
        assert scores[LanguageCode.CHINESE_GENERIC] > 0.3
    
    def test_mixed_language_analysis(self, detector):
        """Test analysis of mixed Chinese-English content."""
        mixed_text = "Python编程教程 complete tutorial guide"
        
        scores = detector._analyze_text_content(mixed_text)
        
        # Both Chinese and English should have some scores
        assert scores[LanguageCode.CHINESE_GENERIC] > 0.1
        assert scores[LanguageCode.ENGLISH] > 0.1


class TestChineseIntegrationScenarios:
    """Test complete Chinese language processing scenarios."""
    
    def test_end_to_end_chinese_detection_and_optimization(self):
        """Test complete flow from detection to optimization."""
        chinese_content = "这是一个关于人工智能技术发展的详细介绍视频"
        
        # Step 1: Detect language
        detected_language = detect_language(chinese_content)
        assert detected_language in ['zh', 'zh-CN', 'zh-TW']
        
        # Step 2: Validate Chinese content
        is_chinese = is_chinese_text(chinese_content)
        assert is_chinese is True
        
        # Step 3: Ensure proper encoding
        encoded_content = ensure_chinese_encoding(chinese_content)
        assert encoded_content == chinese_content
        
        # Step 4: Optimize for LLM processing
        optimized = optimize_chinese_content_for_llm(encoded_content, "summarization")
        
        assert optimized['optimized_text'] == chinese_content
        assert is_chinese_text(optimized['system_prompt'])
        assert chinese_content in optimized['user_prompt']
    
    def test_chinese_content_with_special_characters(self):
        """Test Chinese content with special characters and punctuation."""
        chinese_with_punctuation = "这是测试！包含标点符号：「引号」、句号。"
        
        # Should still be detected as Chinese
        assert is_chinese_text(chinese_with_punctuation) is True
        
        # Should handle encoding properly
        encoded = ensure_chinese_encoding(chinese_with_punctuation)
        assert encoded == chinese_with_punctuation
        
        # Should optimize correctly
        optimized = optimize_chinese_content_for_llm(chinese_with_punctuation, "keywords")
        assert chinese_with_punctuation in optimized['user_prompt']
    
    def test_chinese_content_edge_cases(self):
        """Test edge cases for Chinese content processing."""
        # Very short Chinese text
        short_chinese = "中文"
        assert is_chinese_text(short_chinese) is True
        
        # Single Chinese character
        single_char = "中"
        assert is_chinese_text(single_char) is True
        
        # Chinese with numbers and English
        mixed_content = "Python 3.9 中文教程 2024"
        # Should still detect Chinese if ratio is high enough
        chinese_chars = sum(1 for c in mixed_content if '\u4e00' <= c <= '\u9fff')
        total_chars = len(mixed_content)
        if chinese_chars / total_chars >= 0.1:
            assert is_chinese_text(mixed_content) is True


if __name__ == '__main__':
    pytest.main([__file__])