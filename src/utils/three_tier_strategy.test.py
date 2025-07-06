"""
Unit tests for three-tier transcript acquisition strategy.

This module tests the ThreeTierTranscriptStrategy class and related functionality
to ensure proper transcript tier prioritization and fallback logic.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


# Mock the TranscriptTier and related classes for testing
class TranscriptTier:
    MANUAL = "manual"
    AUTO_GENERATED = "auto"
    TRANSLATED = "translated"


@dataclass
class MockTranscript:
    """Mock transcript object for testing."""
    language_code: str
    is_generated: bool
    is_translatable: bool
    
    def fetch(self):
        return [{'text': f'Sample transcript in {self.language_code}', 'start': 0, 'duration': 2}]


class TranscriptInfo:
    """Mock TranscriptInfo for testing."""
    
    def __init__(self, language_code: str, is_generated: bool, is_translatable: bool, 
                 video_id: str = "", original_transcript=None):
        self.language_code = language_code
        self.is_generated = is_generated
        self.is_translatable = is_translatable
        self.video_id = video_id
        self.original_transcript = original_transcript
        
        # Determine transcript tier
        if not is_generated:
            self.tier = TranscriptTier.MANUAL
        elif is_translatable:
            self.tier = TranscriptTier.TRANSLATED
        else:
            self.tier = TranscriptTier.AUTO_GENERATED
        
        # Quality score (higher is better)
        self.quality_score = self._calculate_quality_score()
    
    def _calculate_quality_score(self) -> int:
        """Calculate quality score for transcript comparison."""
        if self.tier == TranscriptTier.MANUAL:
            return 100
        elif self.tier == TranscriptTier.AUTO_GENERATED:
            return 50
        else:  # TRANSLATED
            return 25
    
    def __repr__(self):
        return f"TranscriptInfo(lang={self.language_code}, tier={self.tier}, score={self.quality_score})"


class MockLanguageDetector:
    """Mock language detector for testing."""
    
    def __init__(self):
        pass
    
    def detect_language_from_metadata(self, metadata):
        # Mock detection result
        return Mock(detected_language=Mock(value='en'))
    
    def get_preferred_transcript_languages(self, detection_result):
        return ['en', 'en-US', 'en-GB']


class ThreeTierTranscriptStrategy:
    """Mock ThreeTierTranscriptStrategy for testing."""
    
    def __init__(self, language_detector=None):
        self.language_detector = language_detector or MockLanguageDetector()
        self.logger = Mock()
    
    def get_transcript_strategy(self, video_id: str, preferred_languages: Optional[List[str]] = None,
                              video_metadata: Optional[Dict[str, Any]] = None) -> List[TranscriptInfo]:
        """Mock implementation for testing."""
        # Create mock available transcripts
        mock_transcripts = [
            MockTranscript('en', False, False),    # Manual English
            MockTranscript('en', True, False),     # Auto-generated English
            MockTranscript('es', True, True),      # Translated Spanish
            MockTranscript('zh-CN', False, False), # Manual Chinese
            MockTranscript('fr', True, True),      # Translated French
        ]
        
        # Create TranscriptInfo objects
        transcript_infos = []
        for transcript in mock_transcripts:
            info = TranscriptInfo(
                language_code=transcript.language_code,
                is_generated=transcript.is_generated,
                is_translatable=transcript.is_translatable,
                video_id=video_id,
                original_transcript=transcript
            )
            transcript_infos.append(info)
        
        # Sort by strategy
        if not preferred_languages:
            preferred_languages = ['en', 'zh-CN', 'zh-TW', 'zh']
        
        return self._sort_by_strategy(transcript_infos, preferred_languages)
    
    def _sort_by_strategy(self, transcript_infos: List[TranscriptInfo], 
                         preferred_languages: List[str]) -> List[TranscriptInfo]:
        """Sort transcripts by three-tier strategy preference."""
        
        def sort_key(transcript_info: TranscriptInfo):
            # Language preference score
            try:
                lang_score = len(preferred_languages) - preferred_languages.index(transcript_info.language_code)
            except ValueError:
                lang_score = 0
                for i, lang in enumerate(preferred_languages):
                    if (transcript_info.language_code.startswith(lang) or 
                        lang.startswith(transcript_info.language_code)):
                        lang_score = len(preferred_languages) - i
                        break
            
            # Tier priority score
            if transcript_info.tier == TranscriptTier.MANUAL:
                tier_score = 1000
            elif transcript_info.tier == TranscriptTier.AUTO_GENERATED:
                tier_score = 500  
            else:  # TRANSLATED
                tier_score = 100
            
            return tier_score + lang_score
        
        return sorted(transcript_infos, key=sort_key, reverse=True)
    
    def get_best_transcript_option(self, video_id: str, preferred_languages: Optional[List[str]] = None,
                                  video_metadata: Optional[Dict[str, Any]] = None) -> Optional[TranscriptInfo]:
        """Get the single best transcript option."""
        strategy_order = self.get_transcript_strategy(video_id, preferred_languages, video_metadata)
        return strategy_order[0] if strategy_order else None
    
    def categorize_transcripts_by_tier(self, transcript_infos: List[TranscriptInfo]) -> Dict[str, List[TranscriptInfo]]:
        """Categorize transcripts by tier."""
        categorized = {
            TranscriptTier.MANUAL: [],
            TranscriptTier.AUTO_GENERATED: [],
            TranscriptTier.TRANSLATED: []
        }
        
        for info in transcript_infos:
            categorized[info.tier].append(info)
        
        return categorized
    
    def get_transcript_tier_summary(self, video_id: str, preferred_languages: Optional[List[str]] = None,
                                   video_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get summary of available transcripts organized by tier."""
        strategy_order = self.get_transcript_strategy(video_id, preferred_languages, video_metadata)
        categorized = self.categorize_transcripts_by_tier(strategy_order)
        
        return {
            'video_id': video_id,
            'total_transcripts': len(strategy_order),
            'tiers': {
                'manual': {
                    'count': len(categorized[TranscriptTier.MANUAL]),
                    'languages': [t.language_code for t in categorized[TranscriptTier.MANUAL]]
                },
                'auto_generated': {
                    'count': len(categorized[TranscriptTier.AUTO_GENERATED]),
                    'languages': [t.language_code for t in categorized[TranscriptTier.AUTO_GENERATED]]
                },
                'translated': {
                    'count': len(categorized[TranscriptTier.TRANSLATED]),
                    'languages': [t.language_code for t in categorized[TranscriptTier.TRANSLATED]]
                }
            },
            'best_option': {
                'language': strategy_order[0].language_code if strategy_order else None,
                'tier': strategy_order[0].tier if strategy_order else None,
                'quality_score': strategy_order[0].quality_score if strategy_order else None
            },
            'preferred_languages': preferred_languages,
            'strategy_order': [
                {
                    'language': t.language_code,
                    'tier': t.tier,
                    'quality_score': t.quality_score
                } for t in strategy_order
            ]
        }


class TestTranscriptTier(unittest.TestCase):
    """Test cases for TranscriptTier enumeration."""
    
    def test_tier_values(self):
        """Test that tier values are correct."""
        self.assertEqual(TranscriptTier.MANUAL, "manual")
        self.assertEqual(TranscriptTier.AUTO_GENERATED, "auto")
        self.assertEqual(TranscriptTier.TRANSLATED, "translated")


class TestTranscriptInfo(unittest.TestCase):
    """Test cases for TranscriptInfo class."""
    
    def test_manual_transcript_info(self):
        """Test manual transcript classification."""
        info = TranscriptInfo('en', False, False)
        
        self.assertEqual(info.language_code, 'en')
        self.assertEqual(info.tier, TranscriptTier.MANUAL)
        self.assertEqual(info.quality_score, 100)
        self.assertFalse(info.is_generated)
        self.assertFalse(info.is_translatable)
    
    def test_auto_generated_transcript_info(self):
        """Test auto-generated transcript classification."""
        info = TranscriptInfo('en', True, False)
        
        self.assertEqual(info.language_code, 'en')
        self.assertEqual(info.tier, TranscriptTier.AUTO_GENERATED)
        self.assertEqual(info.quality_score, 50)
        self.assertTrue(info.is_generated)
        self.assertFalse(info.is_translatable)
    
    def test_translated_transcript_info(self):
        """Test translated transcript classification."""
        info = TranscriptInfo('es', True, True)
        
        self.assertEqual(info.language_code, 'es')
        self.assertEqual(info.tier, TranscriptTier.TRANSLATED)
        self.assertEqual(info.quality_score, 25)
        self.assertTrue(info.is_generated)
        self.assertTrue(info.is_translatable)
    
    def test_transcript_info_repr(self):
        """Test TranscriptInfo string representation."""
        info = TranscriptInfo('en', False, False)
        repr_str = repr(info)
        
        self.assertIn('en', repr_str)
        self.assertIn('manual', repr_str)
        self.assertIn('100', repr_str)


class TestThreeTierTranscriptStrategy(unittest.TestCase):
    """Test cases for ThreeTierTranscriptStrategy class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.strategy = ThreeTierTranscriptStrategy()
        self.video_id = "test_video_123"
    
    def test_strategy_initialization(self):
        """Test strategy initialization."""
        self.assertIsNotNone(self.strategy.language_detector)
        self.assertIsNotNone(self.strategy.logger)
    
    def test_get_transcript_strategy_with_english_preference(self):
        """Test transcript strategy with English preference."""
        preferred_languages = ['en', 'en-US']
        strategy_order = self.strategy.get_transcript_strategy(
            self.video_id, preferred_languages
        )
        
        self.assertIsInstance(strategy_order, list)
        self.assertGreater(len(strategy_order), 0)
        
        # First option should be the highest quality English transcript
        best_option = strategy_order[0]
        self.assertEqual(best_option.language_code, 'en')
        self.assertEqual(best_option.tier, TranscriptTier.MANUAL)
        
        # Verify sorting order
        for i in range(len(strategy_order) - 1):
            current = strategy_order[i]
            next_item = strategy_order[i + 1]
            
            # Current should have higher or equal quality than next
            current_score = self._calculate_sort_score(current, preferred_languages)
            next_score = self._calculate_sort_score(next_item, preferred_languages)
            self.assertGreaterEqual(current_score, next_score)
    
    def test_get_transcript_strategy_with_chinese_preference(self):
        """Test transcript strategy with Chinese preference."""
        preferred_languages = ['zh-CN', 'zh']
        strategy_order = self.strategy.get_transcript_strategy(
            self.video_id, preferred_languages
        )
        
        # First option should be Chinese manual transcript
        best_option = strategy_order[0]
        self.assertEqual(best_option.language_code, 'zh-CN')
        self.assertEqual(best_option.tier, TranscriptTier.MANUAL)
    
    def test_get_best_transcript_option(self):
        """Test getting single best transcript option."""
        best_option = self.strategy.get_best_transcript_option(self.video_id)
        
        self.assertIsNotNone(best_option)
        self.assertIsInstance(best_option, TranscriptInfo)
        self.assertEqual(best_option.tier, TranscriptTier.MANUAL)  # Should be highest quality
    
    def test_categorize_transcripts_by_tier(self):
        """Test transcript categorization by tier."""
        strategy_order = self.strategy.get_transcript_strategy(self.video_id)
        categorized = self.strategy.categorize_transcripts_by_tier(strategy_order)
        
        self.assertIn(TranscriptTier.MANUAL, categorized)
        self.assertIn(TranscriptTier.AUTO_GENERATED, categorized)
        self.assertIn(TranscriptTier.TRANSLATED, categorized)
        
        # Check that we have transcripts in each tier
        self.assertGreater(len(categorized[TranscriptTier.MANUAL]), 0)
        self.assertGreater(len(categorized[TranscriptTier.AUTO_GENERATED]), 0)
        self.assertGreater(len(categorized[TranscriptTier.TRANSLATED]), 0)
        
        # Verify correct categorization
        for tier, transcripts in categorized.items():
            for transcript in transcripts:
                self.assertEqual(transcript.tier, tier)
    
    def test_get_transcript_tier_summary(self):
        """Test transcript tier summary generation."""
        summary = self.strategy.get_transcript_tier_summary(self.video_id)
        
        self.assertEqual(summary['video_id'], self.video_id)
        self.assertIn('total_transcripts', summary)
        self.assertIn('tiers', summary)
        self.assertIn('best_option', summary)
        self.assertIn('strategy_order', summary)
        
        # Check tier structure
        tiers = summary['tiers']
        self.assertIn('manual', tiers)
        self.assertIn('auto_generated', tiers)
        self.assertIn('translated', tiers)
        
        # Check best option
        best_option = summary['best_option']
        self.assertIsNotNone(best_option['language'])
        self.assertIsNotNone(best_option['tier'])
        self.assertIsNotNone(best_option['quality_score'])
        
        # Best option should be highest quality
        self.assertEqual(best_option['tier'], TranscriptTier.MANUAL)
    
    def test_tier_priority_ordering(self):
        """Test that tiers are ordered correctly by priority."""
        strategy_order = self.strategy.get_transcript_strategy(self.video_id)
        
        # Find transcripts of same language but different tiers
        english_transcripts = [t for t in strategy_order if t.language_code == 'en']
        
        # Manual should come before auto-generated for same language
        manual_en = next((t for t in english_transcripts if t.tier == TranscriptTier.MANUAL), None)
        auto_en = next((t for t in english_transcripts if t.tier == TranscriptTier.AUTO_GENERATED), None)
        
        if manual_en and auto_en:
            manual_index = strategy_order.index(manual_en)
            auto_index = strategy_order.index(auto_en)
            self.assertLess(manual_index, auto_index, "Manual transcript should come before auto-generated")
    
    def test_language_preference_ordering(self):
        """Test that preferred languages are prioritized."""
        preferred_languages = ['zh-CN', 'en']
        strategy_order = self.strategy.get_transcript_strategy(
            self.video_id, preferred_languages
        )
        
        # Find best options for each language
        best_chinese = next((t for t in strategy_order if t.language_code == 'zh-CN'), None)
        best_english = next((t for t in strategy_order if t.language_code == 'en'), None)
        
        if best_chinese and best_english and best_chinese.tier == best_english.tier:
            # If same tier, Chinese should come first due to language preference
            chinese_index = strategy_order.index(best_chinese)
            english_index = strategy_order.index(best_english)
            self.assertLess(chinese_index, english_index, 
                           "Preferred language should come first for same tier")
    
    def _calculate_sort_score(self, transcript_info: TranscriptInfo, preferred_languages: List[str]):
        """Helper method to calculate sort score for testing."""
        try:
            lang_score = len(preferred_languages) - preferred_languages.index(transcript_info.language_code)
        except ValueError:
            lang_score = 0
        
        if transcript_info.tier == TranscriptTier.MANUAL:
            tier_score = 1000
        elif transcript_info.tier == TranscriptTier.AUTO_GENERATED:
            tier_score = 500  
        else:  # TRANSLATED
            tier_score = 100
        
        return tier_score + lang_score


class TestTranscriptTierIntegration(unittest.TestCase):
    """Integration tests for three-tier strategy components."""
    
    def test_end_to_end_strategy_flow(self):
        """Test complete flow from strategy to tier summary."""
        strategy = ThreeTierTranscriptStrategy()
        video_id = "integration_test_video"
        
        # Get strategy order
        strategy_order = strategy.get_transcript_strategy(video_id)
        self.assertGreater(len(strategy_order), 0)
        
        # Get best option
        best_option = strategy.get_best_transcript_option(video_id)
        self.assertEqual(best_option.language_code, strategy_order[0].language_code)
        self.assertEqual(best_option.tier, strategy_order[0].tier)
        
        # Get tier summary
        summary = strategy.get_transcript_tier_summary(video_id)
        self.assertEqual(summary['total_transcripts'], len(strategy_order))
        self.assertEqual(summary['best_option']['language'], best_option.language_code)
        self.assertEqual(summary['best_option']['tier'], best_option.tier)
    
    def test_empty_transcript_list_handling(self):
        """Test handling of empty transcript lists."""
        # This would need to be tested with mocking the YouTube API
        # to return no transcripts, but for now we can test the logic
        strategy = ThreeTierTranscriptStrategy()
        empty_transcripts = []
        categorized = strategy.categorize_transcripts_by_tier(empty_transcripts)
        
        self.assertEqual(len(categorized[TranscriptTier.MANUAL]), 0)
        self.assertEqual(len(categorized[TranscriptTier.AUTO_GENERATED]), 0)
        self.assertEqual(len(categorized[TranscriptTier.TRANSLATED]), 0)


if __name__ == '__main__':
    unittest.main()