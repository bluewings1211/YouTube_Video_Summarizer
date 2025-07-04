## Relevant Files

- `src/utils/youtube_api.py` - Enhanced YouTube transcript acquisition with three-tier strategy and proxy support
- `src/utils/youtube_api.test.py` - Unit tests for enhanced YouTube API functionality
- `src/utils/ollama_client.py` - Ollama client integration for local LLM processing
- `src/utils/ollama_client.test.py` - Unit tests for Ollama client functionality
- `src/utils/proxy_manager.py` - Proxy rotation and rate limiting management
- `src/utils/proxy_manager.test.py` - Unit tests for proxy management
- `src/utils/language_detector.py` - Language detection utility for English/Chinese content
- `src/utils/language_detector.test.py` - Unit tests for language detection
- `src/utils/config_manager.py` - Configuration management for model selection and proxy settings
- `src/utils/config_manager.test.py` - Unit tests for configuration management
- `src/nodes.py` - Updated PocketFlow nodes with enhanced transcript processing
- `src/nodes.test.py` - Unit tests for updated PocketFlow nodes
- `requirements.txt` - Updated dependencies including Ollama client and additional libraries
- `.env.example` - Example environment configuration for proxy and model settings
- `docs/setup-guide.md` - Setup guide for Ollama and proxy configuration

### Notes

- Unit tests should typically be placed alongside the code files they are testing (e.g., `youtube_api.py` and `youtube_api.test.py` in the same directory).
- Use `npx jest [optional/path/to/test/file]` to run tests. Running without a path executes all tests found by the Jest configuration.

## Tasks

- [ ] 1.0 Enhanced YouTube Transcript Acquisition System
  - [x] 1.1 Analyze current youtube_api.py and understand existing transcript acquisition logic
  - [x] 1.2 Implement language detection function to identify English vs Chinese videos using YouTube metadata
  - [ ] 1.3 Create three-tier transcript acquisition strategy (manual > auto-generated > translated)
  - [ ] 1.4 Add intelligent fallback logic to automatically try next tier when current tier fails
  - [ ] 1.5 Implement specific exception handling for TranscriptsDisabled, NoTranscriptFound, and other YouTube API errors
  - [ ] 1.6 Add comprehensive logging for transcript acquisition attempts, successes, and failures
  - [ ] 1.7 Update existing get_video_info function to use new enhanced acquisition system
  - [ ] 1.8 Create comprehensive unit tests covering all transcript acquisition scenarios and edge cases

- [ ] 2.0 Proxy Rotation and Rate Limiting Infrastructure
  - [ ] 2.1 Create proxy configuration management system using environment variables and secure storage
  - [ ] 2.2 Implement proxy rotation mechanism with multiple proxy endpoints (similar to YoutubeSummarizer approach)
  - [ ] 2.3 Add rate limiting with enforced minimum 10-second intervals between transcript requests
  - [ ] 2.4 Implement exponential backoff retry logic for failed requests with configurable max attempts
  - [ ] 2.5 Add proxy health checking and automatic proxy switching when proxies fail
  - [ ] 2.6 Integrate proxy support into all YouTube transcript API calls with connection pooling
  - [ ] 2.7 Add comprehensive error handling and detailed logging for proxy operations and failures
  - [ ] 2.8 Create unit tests for proxy rotation, rate limiting, and failure recovery functionality

- [ ] 3.0 Ollama Local LLM Integration
  - [ ] 3.1 Research and select appropriate Ollama models for text summarization (recommend llama3, mistral, or similar)
  - [ ] 3.2 Create Ollama client wrapper with connection management and error handling
  - [ ] 3.3 Implement configurable model selection logic (local Ollama vs cloud models)
  - [ ] 3.4 Add Chinese language processing support ensuring proper encoding and model compatibility
  - [ ] 3.5 Ensure output format parity between Ollama responses and existing cloud model responses
  - [ ] 3.6 Add graceful error handling and fallback mechanisms when Ollama is unavailable
  - [ ] 3.7 Update all existing LLM call points in PocketFlow nodes to use new unified client
  - [ ] 3.8 Create comprehensive unit tests for Ollama integration, including Chinese content processing

- [ ] 4.0 Language Detection and Processing Logic
  - [ ] 4.1 Implement automatic language detection using YouTube video metadata and transcript analysis
  - [ ] 4.2 Create language-specific processing workflows for English and Chinese content
  - [ ] 4.3 Add support for mixed-language content detection and appropriate handling strategies
  - [ ] 4.4 Integrate language detection with three-tier transcript acquisition strategy
  - [ ] 4.5 Ensure proper language preservation for Chinese content throughout summarization pipeline
  - [ ] 4.6 Add language preference configuration options for user control
  - [ ] 4.7 Create unit tests covering language detection accuracy and processing logic for both languages

- [ ] 5.0 Configuration Management and Environment Setup
  - [ ] 5.1 Create comprehensive environment variable system for proxy credentials and endpoints
  - [ ] 5.2 Add model selection configuration with support for local/cloud preference and specific model names
  - [ ] 5.3 Implement secure credential storage system for proxy authentication details
  - [ ] 5.4 Ensure complete backward compatibility with existing configuration systems
  - [ ] 5.5 Create configuration validation system with helpful error messages for invalid settings
  - [ ] 5.6 Update requirements.txt with all new dependencies (ollama client, proxy libraries, etc.)
  - [ ] 5.7 Create detailed setup documentation for Ollama installation and model management
  - [ ] 5.8 Create example configuration files (.env.example) and environment setup templates