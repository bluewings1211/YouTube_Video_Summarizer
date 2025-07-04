# Product Requirements Document: Enhanced YouTube Transcript Processing with Local LLM Support

## 1. Introduction/Overview

This feature enhances the current YouTube transcript processing system by addressing critical issues with rate limiting, improving transcript acquisition success rates, and integrating local LLM capabilities. The enhancement builds upon the existing PocketFlow-based youtube-summarizer-wave project to provide more reliable and cost-effective video content summarization.

**Problem Statement:** The current system frequently encounters YouTube API rate limits and fails to acquire transcripts for videos that only have auto-generated captions, resulting in poor user experience and limited functionality coverage.

**Goal:** Create a robust, reliable transcript processing system that can handle the majority of YouTube videos while reducing dependency on expensive cloud-based LLM services.

## 2. Goals

1. **Improve Transcript Acquisition Success Rate**: Achieve 95%+ success rate for videos with available captions (manual, auto-generated, or translated)
2. **Eliminate Rate Limit Failures**: Implement proxy rotation and intelligent retry mechanisms to prevent rate limit errors
3. **Reduce Operational Costs**: Replace cloud LLM services with local Ollama models as the default option
4. **Support Multilingual Content**: Automatically detect and process English and Chinese content appropriately
5. **Maintain User Experience**: Provide seamless integration with existing PocketFlow workflow

## 3. User Stories

**As a general user**, I want to summarize YouTube videos without encountering "transcript unavailable" errors, so that I can get insights from any video with speech content.

**As a cost-conscious user**, I want to use local AI models for summarization, so that I can reduce API costs while maintaining good quality results.

**As a user processing Chinese content**, I want the system to automatically detect the language and provide summaries in the original language, so that I don't lose context through translation.

**As a user with limited technical knowledge**, I want the system to automatically handle rate limits and retry mechanisms, so that I don't need to manually troubleshoot failed requests.

**As a power user**, I want to choose between local and cloud models, so that I can balance between cost and performance based on my needs.

## 4. Functional Requirements

### 4.1 Enhanced Transcript Acquisition
1. The system must implement a three-tier transcript acquisition strategy:
   - **Tier 1**: Manual/human-created transcripts
   - **Tier 2**: Auto-generated transcripts
   - **Tier 3**: Auto-translated transcripts (when original language differs from target)

2. The system must automatically detect video language (English or Chinese) and acquire appropriate transcripts

3. The system must support fallback mechanisms - if higher-tier transcripts are unavailable, automatically attempt lower-tier options

### 4.2 Rate Limit Management
4. The system must implement proxy rotation mechanism similar to the YoutubeSummarizer project approach

5. The system must enforce a minimum 10-second interval between transcript requests

6. The system must implement intelligent retry logic with exponential backoff for failed requests

7. The system must handle TranscriptsDisabled and NoTranscriptFound exceptions gracefully

### 4.3 Local LLM Integration
8. The system must integrate Ollama as the default LLM provider for content summarization

9. The system must provide user option to switch between local (Ollama) and cloud-based models

10. The system must support Chinese language processing for local model summarization

11. The system must maintain feature parity between local and cloud model outputs

### 4.4 Language Processing
12. The system must automatically detect video language (English/Chinese) without user input

13. For Chinese videos, the system must preserve original language for summarization

14. For English videos, the system must provide English summaries

15. The system must handle mixed-language content appropriately

### 4.5 Error Handling & User Feedback
16. The system must provide clear error messages when all transcript acquisition attempts fail

17. The system must log detailed error information for debugging purposes

18. The system must continue processing remaining videos in batch operations when individual videos fail

## 5. Non-Goals (Out of Scope)

- Support for languages other than English and Chinese
- Real-time progress indicators for long-running operations
- Audio-to-transcript conversion using AI models
- Integration with video hosting platforms other than YouTube
- Advanced transcript editing or correction features
- User authentication or account management
- Custom LLM model training or fine-tuning

## 6. Design Considerations

### 6.1 PocketFlow Integration
- Maintain compatibility with existing Node-based architecture
- Preserve current workflow and data passing mechanisms
- Ensure minimal changes to existing user interface

### 6.2 Configuration Management
- Store proxy configuration in environment variables or secure config files
- Provide easy model selection mechanism (environment variable or config file)
- Maintain backward compatibility with existing configuration

### 6.3 Performance Considerations
- Implement connection pooling for proxy requests
- Cache successful transcript requests to avoid redundant API calls
- Optimize memory usage for large transcript processing

## 7. Technical Considerations

### 7.1 Dependencies
- Integration with existing `youtube-transcript-api` library
- Ollama Python/JavaScript client library
- Proxy management library (requests with proxy support)
- Language detection library (optional, can use YouTube metadata)

### 7.2 Infrastructure Requirements
- Local Ollama installation and model management
- Proxy service subscription (similar to YoutubeSummarizer setup)
- Sufficient local compute resources for Ollama operations

### 7.3 Security Considerations
- Secure storage of proxy credentials
- Rate limiting to prevent abuse
- Input validation for YouTube URLs

## 8. Success Metrics

### 8.1 Primary Metrics
- **Transcript Acquisition Success Rate**: Target >95% for videos with any available captions
- **Rate Limit Error Reduction**: Target <1% of requests result in rate limit errors
- **Cost Reduction**: Target 80%+ reduction in LLM API costs through local model usage

### 8.2 Secondary Metrics
- **Processing Time**: Maintain reasonable processing times despite rate limiting controls
- **User Satisfaction**: Measure through reduced error reports and support requests
- **System Reliability**: Target 99%+ uptime for transcript processing functionality

## 9. Open Questions

1. **Proxy Service Selection**: Should we use the same proxy service as YoutubeSummarizer, or explore alternative options?

2. **Ollama Model Selection**: Which specific Ollama models should be recommended/supported for optimal performance vs. resource usage?

3. **Fallback Strategy**: If local Ollama is unavailable, should the system automatically fallback to cloud models or return an error?

4. **Batch Processing**: How should rate limiting be handled in batch operations - sequential processing or intelligent queuing?

5. **Configuration Management**: Should model selection be a per-request option or a global configuration setting?

6. **Error Recovery**: For partially successful batch operations, should the system provide options to retry only failed items?