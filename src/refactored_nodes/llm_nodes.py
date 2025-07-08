"""
LLM-based processing nodes for text summarization and analysis.

This module contains nodes that use Large Language Models for processing
video transcripts, including summarization and keyword extraction.
"""

import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from .validation_nodes import BaseProcessingNode, Store
from ..utils.smart_llm_client import SmartLLMClient, TaskRequirements, detect_task_requirements

logger = logging.getLogger(__name__)


class SummarizationNode(BaseProcessingNode):
    """
    Node for generating AI-powered summaries of video transcripts.
    
    This node takes transcript data and generates a professional summary
    of approximately 500 words using configured LLM providers.
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0):
        super().__init__("SummarizationNode", max_retries, retry_delay)
        self.smart_llm_client = None
        self.target_word_count = 500
    
    def prep(self, store: Store) -> Dict[str, Any]:
        """
        Prepare for text summarization by validating input and setting up LLM client.
        
        Args:
            store: Data store containing transcript data
            
        Returns:
            Dict containing prep results and LLM configuration
        """
        self.logger.info("Starting summarization preparation")
        
        try:
            # Validate required input
            is_valid, missing_keys = self._validate_store_data(store, ['transcript_data'])
            if not is_valid:
                raise ValueError(f"Missing required data: {missing_keys}")
            
            transcript_data = store['transcript_data']
            transcript_text = transcript_data.get('transcript_text', '')
            
            if not transcript_text or not transcript_text.strip():
                raise ValueError("No transcript text available for summarization")
            
            # Check transcript length
            word_count = len(transcript_text.split())
            if word_count < 50:
                raise ValueError(f"Transcript too short for summarization ({word_count} words)")
            
            # Initialize Smart LLM client
            try:
                self.smart_llm_client = SmartLLMClient()
            except Exception as e:
                raise ValueError(f"Failed to initialize Smart LLM client: {str(e)}")
            
            prep_result = {
                'transcript_text': transcript_text,
                'original_word_count': word_count,
                'target_word_count': self.target_word_count,
                'video_id': transcript_data.get('video_id', 'unknown'),
                'video_title': store.get('video_metadata', {}).get('title', 'Unknown'),
                'prep_timestamp': datetime.utcnow().isoformat(),
                'prep_status': 'success'
            }
            
            self.logger.info(f"Summarization prep successful for {word_count} words")
            return prep_result
            
        except Exception as e:
            error_info = self._handle_error(e, "Summarization preparation failed")
            return {
                'prep_status': 'failed',
                'error': error_info.__dict__,
                'prep_timestamp': datetime.utcnow().isoformat()
            }
    
    def exec(self, store: Store, prep_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute text summarization with retry logic.
        
        Args:
            store: Data store
            prep_result: Results from prep phase
            
        Returns:
            Dict containing generated summary and metadata
        """
        self.logger.info("Starting summarization execution")
        
        if prep_result.get('prep_status') != 'success':
            return {
                'exec_status': 'failed',
                'error': 'Prep phase failed',
                'exec_timestamp': datetime.utcnow().isoformat()
            }
        
        transcript_text = prep_result['transcript_text']
        video_title = prep_result['video_title']
        target_word_count = prep_result['target_word_count']
        last_error = None
        
        for retry_count in range(self.max_retries + 1):
            try:
                self._retry_with_delay(retry_count)
                
                # Get language detection results for language-specific processing
                language_detection_result = store.get('language_detection_result', {})
                detected_language = language_detection_result.get('detected_language', 'en')
                is_chinese = language_detection_result.get('is_chinese', False)
                
                # Apply language-specific text preprocessing if needed
                processed_text = transcript_text
                if is_chinese:
                    try:
                        # Import here to avoid circular dependencies
                        from ..utils.language_detector import ensure_chinese_encoding
                        processed_text = ensure_chinese_encoding(transcript_text)
                        self.logger.debug("Applied Chinese encoding preservation")
                    except Exception as e:
                        self.logger.debug(f"Could not apply Chinese encoding: {str(e)}")
                
                # Create task requirements for smart model selection
                task_requirements = detect_task_requirements(
                    text=processed_text,
                    task_type="summarization",
                    quality_level="medium"
                )
                
                # Add language information to task requirements
                if detected_language:
                    task_requirements.language = detected_language
                    task_requirements.is_chinese = is_chinese
                
                # Generate summary with smart client and language-aware optimization
                if is_chinese:
                    # Use Chinese-optimized processing
                    summary_result = self.smart_llm_client.generate_text_with_chinese_optimization(
                        text=processed_text,
                        task_requirements=task_requirements,
                        max_tokens=800,  # Allow extra tokens for 500-word summary
                        temperature=0.3  # Lower temperature for consistent summaries
                    )
                else:
                    # Use standard processing for English/other languages
                    summary_result = self.smart_llm_client.generate_text_with_fallback(
                        prompt=processed_text,
                        task_requirements=task_requirements,
                        max_tokens=800,
                        temperature=0.3
                    )
                
                # Extract and validate summary text
                summary_text = summary_result['text']
                summary_word_count = len(summary_text.split())
                
                if not summary_text or summary_word_count < 50:
                    raise ValueError(f"Generated summary too short ({summary_word_count} words)")
                
                # Calculate summary statistics
                summary_stats = self._calculate_summary_stats(
                    summary_text, 
                    transcript_text,
                    target_word_count
                )
                
                # Calculate compression ratio
                compression_ratio = prep_result['original_word_count'] / max(summary_word_count, 1)
                
                exec_result = {
                    'exec_status': 'success',
                    'summary_text': summary_text,
                    'summary_word_count': summary_word_count,
                    'target_word_count': target_word_count,
                    'original_word_count': prep_result['original_word_count'],
                    'compression_ratio': compression_ratio,
                    'summary_stats': summary_stats,
                    'video_id': prep_result['video_id'],
                    'video_title': video_title,
                    'llm_metadata': summary_result,
                    'exec_timestamp': datetime.utcnow().isoformat(),
                    'retry_count': retry_count
                }
                
                self.logger.info(f"Summarization successful: {summary_word_count} words")
                return exec_result
                
            except Exception as e:
                last_error = self._handle_error(e, f"Summarization execution failed", retry_count)
                if retry_count >= self.max_retries:
                    break
        
        return {
            'exec_status': 'failed',
            'error': last_error.__dict__ if last_error else 'Unknown error',
            'exec_timestamp': datetime.utcnow().isoformat(),
            'retry_count': self.max_retries
        }
    
    def post(self, store: Store, prep_result: Dict[str, Any], exec_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process summary and update store.
        
        Args:
            store: Data store
            prep_result: Results from prep phase
            exec_result: Results from exec phase
            
        Returns:
            Dict containing final processing results
        """
        self.logger.info("Starting summarization post-processing")
        
        try:
            if exec_result.get('exec_status') != 'success':
                return {
                    'post_status': 'failed',
                    'error': 'Execution phase failed',
                    'post_timestamp': datetime.utcnow().isoformat()
                }
            
            # Extract summary information
            summary_text = exec_result['summary_text']
            summary_word_count = exec_result['summary_word_count']
            video_id = exec_result['video_id']
            video_title = exec_result['video_title']
            
            # Apply language-specific post-processing
            language_detection_result = store.get('language_detection_result', {})
            is_chinese = language_detection_result.get('is_chinese', False)
            
            # Ensure Chinese content encoding is preserved
            if is_chinese and summary_text:
                try:
                    from ..utils.language_detector import ensure_chinese_encoding
                    summary_text = ensure_chinese_encoding(summary_text)
                    self.logger.debug("Applied Chinese encoding preservation to summary")
                except Exception as e:
                    self.logger.debug(f"Could not apply Chinese encoding to summary: {str(e)}")
            
            # Prepare store data
            store_data = {
                'summary_data': {
                    'summary_text': summary_text,
                    'word_count': summary_word_count,
                    'target_word_count': exec_result['target_word_count'],
                    'compression_ratio': exec_result['compression_ratio'],
                    'stats': exec_result['summary_stats'],
                    'generated_at': exec_result['exec_timestamp']
                },
                'summary_metadata': {
                    'video_id': video_id,
                    'video_title': video_title,
                    'original_word_count': exec_result['original_word_count'],
                    'processing_duration': self._calculate_duration(
                        prep_result.get('prep_timestamp', ''),
                        exec_result.get('exec_timestamp', '')
                    ),
                    'retry_count': exec_result.get('retry_count', 0),
                    'llm_provider': exec_result.get('llm_metadata', {}).get('provider', 'unknown'),
                    'llm_model': exec_result.get('llm_metadata', {}).get('model', 'unknown')
                }
            }
            
            # Update store
            self._safe_store_update(store, store_data)
            
            post_result = {
                'post_status': 'success',
                'summary_ready': True,
                'summary_length': summary_word_count,
                'video_id': video_id,
                'video_title': video_title,
                'compression_ratio': exec_result['compression_ratio'],
                'post_timestamp': datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Summary post-processing successful for video {video_id}")
            return post_result
            
        except Exception as e:
            error_info = self._handle_error(e, "Summary post-processing failed")
            return {
                'post_status': 'failed',
                'error': error_info.__dict__,
                'post_timestamp': datetime.utcnow().isoformat()
            }
    
    def _calculate_summary_stats(
        self, 
        summary_text: str, 
        original_text: str, 
        target_word_count: int
    ) -> Dict[str, Any]:
        """Calculate statistics for the generated summary."""
        if not summary_text:
            return {}
        
        summary_words = summary_text.split()
        original_words = original_text.split()
        
        # Calculate readability metrics
        sentences = re.split(r'[.!?]+', summary_text)
        sentences = [s for s in sentences if s.strip()]
        
        avg_sentence_length = len(summary_words) / max(len(sentences), 1)
        
        # Calculate target accuracy
        target_accuracy = min(100, (target_word_count / max(len(summary_words), 1)) * 100)
        
        return {
            'char_count': len(summary_text),
            'word_count': len(summary_words),
            'sentence_count': len(sentences),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'target_accuracy': round(target_accuracy, 2),
            'compression_ratio': len(original_words) / max(len(summary_words), 1),
            'estimated_reading_time_minutes': round(len(summary_words) / 200, 1)  # ~200 WPM
        }
    
    def _calculate_duration(self, start_time: str, end_time: str) -> float:
        """Calculate duration between two timestamps."""
        try:
            start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            return (end - start).total_seconds()
        except Exception:
            return 0.0


class KeywordExtractionNode(BaseProcessingNode):
    """
    Node for extracting relevant keywords from video content.
    
    This node analyzes video transcripts and summaries to extract 5-8 relevant
    keywords that best represent the video content for categorization and search.
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.5):
        super().__init__("KeywordExtractionNode", max_retries, retry_delay)
        self.smart_llm_client = None
        self.default_keyword_count = 6  # Middle of 5-8 range
    
    def prep(self, store: Store) -> Dict[str, Any]:
        """
        Prepare for keyword extraction by validating input data.
        
        Args:
            store: Data store containing transcript and summary data
            
        Returns:
            Dict containing prep results and configuration
        """
        self.logger.info("Starting keyword extraction preparation")
        
        try:
            # Validate required input - we need at least transcript data
            required_keys = ['transcript_data']
            is_valid, missing_keys = self._validate_store_data(store, required_keys)
            if not is_valid:
                raise ValueError(f"Missing required data: {missing_keys}")
            
            transcript_data = store['transcript_data']
            transcript_text = transcript_data.get('transcript_text', '')
            
            if not transcript_text or not transcript_text.strip():
                raise ValueError("No transcript text available for keyword extraction")
            
            # Get summary data if available (preferred for keyword extraction)
            summary_data = store.get('summary_data', {})
            summary_text = summary_data.get('summary_text', '')
            
            # Use summary if available, otherwise use transcript
            extraction_text = summary_text if summary_text else transcript_text
            
            # Validate text length
            word_count = len(extraction_text.split())
            if word_count < 20:
                raise ValueError(f"Text too short for keyword extraction ({word_count} words)")
            
            # Initialize Smart LLM client
            try:
                self.smart_llm_client = SmartLLMClient()
            except Exception as e:
                raise ValueError(f"Failed to initialize Smart LLM client: {str(e)}")
            
            # Get video metadata for context
            video_metadata = store.get('video_metadata', {})
            video_title = video_metadata.get('title', 'Unknown')
            
            prep_result = {
                'extraction_text': extraction_text,
                'text_source': 'summary' if summary_text else 'transcript',
                'text_word_count': word_count,
                'video_id': transcript_data.get('video_id', 'unknown'),
                'video_title': video_title,
                'target_keyword_count': self.default_keyword_count,
                'prep_timestamp': datetime.utcnow().isoformat(),
                'prep_status': 'success'
            }
            
            self.logger.info(f"Keyword extraction prep successful for {word_count} words")
            return prep_result
            
        except Exception as e:
            error_info = self._handle_error(e, "Keyword extraction preparation failed")
            return {
                'prep_status': 'failed',
                'error': error_info.__dict__,
                'prep_timestamp': datetime.utcnow().isoformat()
            }
    
    def exec(self, store: Store, prep_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute keyword extraction with retry logic.
        
        Args:
            store: Data store
            prep_result: Results from prep phase
            
        Returns:
            Dict containing extracted keywords and metadata
        """
        self.logger.info("Starting keyword extraction execution")
        
        if prep_result.get('prep_status') != 'success':
            return {
                'exec_status': 'failed',
                'error': 'Prep phase failed',
                'exec_timestamp': datetime.utcnow().isoformat()
            }
        
        extraction_text = prep_result['extraction_text']
        video_title = prep_result['video_title']
        target_count = prep_result['target_keyword_count']
        last_error = None
        
        for retry_count in range(self.max_retries + 1):
            try:
                self._retry_with_delay(retry_count)
                
                # Get language detection results
                language_detection_result = store.get('language_detection_result', {})
                detected_language = language_detection_result.get('detected_language', 'en')
                is_chinese = language_detection_result.get('is_chinese', False)
                
                # Apply language-specific preprocessing
                processed_text = extraction_text
                if is_chinese:
                    try:
                        from ..utils.language_detector import ensure_chinese_encoding
                        processed_text = ensure_chinese_encoding(extraction_text)
                        self.logger.debug("Applied Chinese encoding preservation")
                    except Exception as e:
                        self.logger.debug(f"Could not apply Chinese encoding: {str(e)}")
                
                # Create task requirements for smart model selection
                task_requirements = detect_task_requirements(
                    text=processed_text,
                    task_type="keyword_extraction",
                    quality_level="medium"
                )
                
                # Add language information
                if detected_language:
                    task_requirements.language = detected_language
                    task_requirements.is_chinese = is_chinese
                
                # Generate keywords with smart client
                keywords_result = self._generate_keywords(
                    processed_text, 
                    video_title, 
                    target_count,
                    task_requirements,
                    is_chinese
                )
                
                # Extract and validate keywords
                keywords = keywords_result['keywords']
                if not keywords or len(keywords) < 2:
                    raise ValueError(f"Generated too few keywords ({len(keywords)})")
                
                # Enhance keywords with metadata
                enhanced_keywords = self._enhance_keywords(
                    keywords, 
                    video_title, 
                    prep_result['text_source']
                )
                
                # Calculate keyword statistics
                keyword_stats = self._calculate_keyword_stats(
                    enhanced_keywords,
                    processed_text
                )
                
                exec_result = {
                    'exec_status': 'success',
                    'keywords': enhanced_keywords,
                    'keyword_count': len(enhanced_keywords),
                    'target_count': target_count,
                    'keyword_stats': keyword_stats,
                    'video_id': prep_result['video_id'],
                    'video_title': video_title,
                    'text_source': prep_result['text_source'],
                    'llm_metadata': keywords_result,
                    'exec_timestamp': datetime.utcnow().isoformat(),
                    'retry_count': retry_count
                }
                
                self.logger.info(f"Keyword extraction successful: {len(enhanced_keywords)} keywords")
                return exec_result
                
            except Exception as e:
                last_error = self._handle_error(e, f"Keyword extraction execution failed", retry_count)
                if retry_count >= self.max_retries:
                    break
        
        return {
            'exec_status': 'failed',
            'error': last_error.__dict__ if last_error else 'Unknown error',
            'exec_timestamp': datetime.utcnow().isoformat(),
            'retry_count': self.max_retries
        }
    
    def post(self, store: Store, prep_result: Dict[str, Any], exec_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process keywords and update store.
        
        Args:
            store: Data store
            prep_result: Results from prep phase
            exec_result: Results from exec phase
            
        Returns:
            Dict containing final processing results
        """
        self.logger.info("Starting keyword extraction post-processing")
        
        try:
            if exec_result.get('exec_status') != 'success':
                return {
                    'post_status': 'failed',
                    'error': 'Execution phase failed',
                    'post_timestamp': datetime.utcnow().isoformat()
                }
            
            # Extract keyword information
            keywords = exec_result['keywords']
            keyword_count = exec_result['keyword_count']
            video_id = exec_result['video_id']
            video_title = exec_result['video_title']
            
            # Apply language-specific post-processing
            language_detection_result = store.get('language_detection_result', {})
            is_chinese = language_detection_result.get('is_chinese', False)
            
            # Ensure Chinese content encoding is preserved
            if is_chinese and keywords:
                try:
                    from ..utils.language_detector import ensure_chinese_encoding
                    for keyword in keywords:
                        keyword['keyword'] = ensure_chinese_encoding(keyword['keyword'])
                    self.logger.debug("Applied Chinese encoding preservation to keywords")
                except Exception as e:
                    self.logger.debug(f"Could not apply Chinese encoding to keywords: {str(e)}")
            
            # Prepare store data
            store_data = {
                'keywords_data': {
                    'keywords': keywords,
                    'count': keyword_count,
                    'target_count': exec_result['target_count'],
                    'stats': exec_result['keyword_stats'],
                    'text_source': exec_result['text_source'],
                    'generated_at': exec_result['exec_timestamp']
                },
                'keywords_metadata': {
                    'video_id': video_id,
                    'video_title': video_title,
                    'processing_duration': self._calculate_duration(
                        prep_result.get('prep_timestamp', ''),
                        exec_result.get('exec_timestamp', '')
                    ),
                    'retry_count': exec_result.get('retry_count', 0),
                    'llm_provider': exec_result.get('llm_metadata', {}).get('provider', 'unknown'),
                    'llm_model': exec_result.get('llm_metadata', {}).get('model', 'unknown')
                }
            }
            
            # Update store
            self._safe_store_update(store, store_data)
            
            post_result = {
                'post_status': 'success',
                'keywords_ready': True,
                'keyword_count': keyword_count,
                'video_id': video_id,
                'video_title': video_title,
                'post_timestamp': datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Keywords post-processing successful for video {video_id}")
            return post_result
            
        except Exception as e:
            error_info = self._handle_error(e, "Keywords post-processing failed")
            return {
                'post_status': 'failed',
                'error': error_info.__dict__,
                'post_timestamp': datetime.utcnow().isoformat()
            }
    
    def _generate_keywords(
        self, 
        text: str, 
        video_title: str, 
        count: int,
        task_requirements: TaskRequirements,
        is_chinese: bool = False
    ) -> Dict[str, Any]:
        """Generate keywords using the smart LLM client."""
        
        # Create keyword extraction prompt
        prompt = f"""
        Extract {count} most relevant keywords from the following text about a video titled "{video_title}".
        
        Text: {text}
        
        Requirements:
        - Return {count} keywords or short phrases
        - Each keyword should be 1-3 words maximum
        - Keywords should be relevant for search and categorization
        - Avoid common words like "the", "and", "video", "content"
        - Focus on specific topics, concepts, or themes
        - Return only the keywords, one per line
        """
        
        # Generate keywords with appropriate method based on language
        if is_chinese:
            result = self.smart_llm_client.generate_text_with_chinese_optimization(
                text=prompt,
                task_requirements=task_requirements,
                max_tokens=200,
                temperature=0.2
            )
        else:
            result = self.smart_llm_client.generate_text_with_fallback(
                prompt=prompt,
                task_requirements=task_requirements,
                max_tokens=200,
                temperature=0.2
            )
        
        # Parse keywords from response
        keywords_text = result['text']
        keywords = []
        
        for line in keywords_text.split('\n'):
            keyword = line.strip()
            # Remove numbering or bullet points
            keyword = re.sub(r'^\d+\.\s*', '', keyword)
            keyword = re.sub(r'^\-\s*', '', keyword)
            keyword = re.sub(r'^\*\s*', '', keyword)
            
            if keyword and len(keyword.split()) <= 3:
                keywords.append(keyword)
        
        return {
            'keywords': keywords,
            'count': len(keywords),
            'requested_count': count,
            'generation_metadata': result
        }

    def _enhance_keywords(
        self, 
        keywords: List[str], 
        video_title: str, 
        text_source: str
    ) -> List[Dict[str, Any]]:
        """Enhance keywords with additional metadata."""
        enhanced = []
        
        for i, keyword in enumerate(keywords):
            keyword_clean = keyword.strip()
            word_count = len(keyword_clean.split())
            
            # Determine keyword type
            if word_count == 1:
                keyword_type = 'single_word'
            elif word_count == 2:
                keyword_type = 'phrase'
            else:
                keyword_type = 'long_phrase'
            
            # Calculate relevance score based on position (earlier = more relevant)
            relevance_score = max(1, 10 - i)  # Score from 10 (first) to 1 (last)
            
            enhanced_keyword = {
                'keyword': keyword_clean,
                'position': i + 1,
                'type': keyword_type,
                'word_count': word_count,
                'relevance_score': relevance_score,
                'text_source': text_source,
                'video_title': video_title,
                'length': len(keyword_clean),
                'is_phrase': word_count > 1
            }
            
            enhanced.append(enhanced_keyword)
        
        return enhanced
    
    def _calculate_keyword_stats(
        self, 
        keywords: List[Dict[str, Any]], 
        source_text: str
    ) -> Dict[str, Any]:
        """Calculate statistics for extracted keywords."""
        if not keywords:
            return {}
        
        # Basic counts
        total_keywords = len(keywords)
        single_words = sum(1 for kw in keywords if kw['type'] == 'single_word')
        phrases = sum(1 for kw in keywords if kw['type'] == 'phrase')
        long_phrases = sum(1 for kw in keywords if kw['type'] == 'long_phrase')
        
        # Length statistics
        keyword_lengths = [len(kw['keyword']) for kw in keywords]
        avg_length = sum(keyword_lengths) / total_keywords
        
        # Relevance statistics
        relevance_scores = [kw['relevance_score'] for kw in keywords]
        avg_relevance = sum(relevance_scores) / total_keywords
        
        # Coverage analysis (how many keywords appear in source text)
        coverage_count = 0
        source_lower = source_text.lower()
        for kw in keywords:
            if kw['keyword'].lower() in source_lower:
                coverage_count += 1
        
        coverage_percent = (coverage_count / total_keywords) * 100
        
        return {
            'total_count': total_keywords,
            'single_word_count': single_words,
            'phrase_count': phrases,
            'long_phrase_count': long_phrases,
            'avg_length': round(avg_length, 2),
            'min_length': min(keyword_lengths),
            'max_length': max(keyword_lengths),
            'avg_relevance_score': round(avg_relevance, 2),
            'coverage_percent': round(coverage_percent, 2),
            'covered_keywords': coverage_count,
            'phrase_ratio': round((phrases + long_phrases) / total_keywords * 100, 2)
        }
    
    def _calculate_duration(self, start_time: str, end_time: str) -> float:
        """Calculate duration between two timestamps."""
        try:
            start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            return (end - start).total_seconds()
        except Exception:
            return 0.0