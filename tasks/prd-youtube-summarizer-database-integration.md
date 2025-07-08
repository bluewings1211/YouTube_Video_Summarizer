# Product Requirements Document: YouTube Summarizer Database Integration & History APIs

## Introduction/Overview

This feature enhances the existing YouTube Summarizer service by adding persistent data storage and history management capabilities. Currently, the service operates as a stateless API that processes YouTube videos and returns summaries without storing results. This PRD outlines the implementation of database integration to store processing results and provide history APIs for querying past summarizations.

The primary goal is to enable users to track their video processing history, retrieve past results, and avoid reprocessing the same content unnecessarily.

## Goals

1. **Data Persistence**: Store all video processing results (metadata, transcripts, summaries, keywords) in a PostgreSQL database
2. **History Management**: Provide APIs to query past processing results with pagination and filtering
3. **Code Quality**: Refactor oversized files to improve maintainability while preserving PocketFlow patterns
4. **Performance**: Enable efficient retrieval of historical data without impacting current processing performance
5. **User Experience**: Allow users to access their complete processing history through well-designed APIs

## User Stories

1. **As a user**, I want to view a list of all videos I've previously processed so that I can track my summarization history
2. **As a user**, I want to search and filter my processing history by date, keywords, or video title so that I can quickly find specific results
3. **As a user**, I want to retrieve detailed information about a specific processed video so that I can access the full summary, timestamps, and keywords without reprocessing
4. **As a user**, I want pagination support for my history so that I can efficiently browse through large numbers of processed videos
5. **As a developer**, I want clean, maintainable code modules so that I can easily extend functionality
6. **As a developer**, I want preserved PocketFlow patterns so that the workflow orchestration remains consistent

## Functional Requirements

### Database Requirements
1. The system must use PostgreSQL as the primary database
2. The system must store video metadata including video_id, title, duration, and processing timestamps
3. The system must store complete transcript text in plain text format
4. The system must store generated summaries, keywords, and timestamped segments
5. The system must track processing parameters and workflow execution details
6. The system must support efficient queries for history retrieval

### History API Requirements
7. The system must provide `GET /api/v1/history/videos` endpoint that returns paginated list of processed videos
8. The system must support filtering by date range, keywords, and video title in the history endpoint
9. The system must provide `GET /api/v1/history/videos/{video_id}` endpoint for detailed video information
10. The system must include pagination metadata (total count, page size, current page) in list responses
11. The system must return proper HTTP status codes and error messages for history queries

### Integration Requirements
12. The system must modify existing workflow to persist results after successful processing
13. The system must maintain existing API response format while adding database storage
14. The system must handle database failures gracefully without breaking the summarization workflow
15. The system must check for existing processing results before starting new workflows

### Code Refactoring Requirements
16. The system must refactor `src/nodes.py` into smaller, focused modules (target: <500 lines per module)
17. The system must refactor `src/flow.py` into manageable components (target: <800 lines per module)
18. The system must refactor `src/utils/youtube_api.py` into focused utility modules (target: <1000 lines per module)
19. The system must preserve all existing PocketFlow patterns and functionality during refactoring
20. The system must maintain backward compatibility with existing tests

## Non-Goals (Out of Scope)

1. **User Authentication**: No user management or authentication system (personal use only)
2. **Real-time Updates**: No WebSocket or real-time notification features
3. **Data Analytics**: No analytics dashboard or reporting features
4. **Content Recommendations**: No AI-powered content recommendation system
5. **External Integrations**: No third-party service integrations beyond existing YouTube API
6. **Mobile App**: No mobile application development
7. **Caching Layer**: No Redis or advanced caching implementation
8. **Multi-language Support**: No internationalization beyond existing language detection

## Technical Considerations

### Database Design
- Use SQLAlchemy ORM for database operations
- Implement proper indexing for efficient history queries
- Use JSON columns for storing complex data structures (timestamps, keywords)
- Set up database migrations for schema management

### API Design
- Follow RESTful conventions for history endpoints
- Implement proper HTTP status codes and error handling
- Use Pydantic models for request/response validation
- Include comprehensive OpenAPI documentation

### Performance Considerations
- Implement database connection pooling
- Use async database operations where possible
- Add database query optimization for history retrieval
- Consider query result caching for frequently accessed data

### Error Handling
- Implement graceful degradation when database is unavailable
- Provide meaningful error messages for API consumers
- Log database errors for troubleshooting
- Maintain service availability during database maintenance

## Success Metrics

1. **Code Quality**: Reduce average file size by 60% (from 2000+ lines to <800 lines)
2. **API Response Time**: History API endpoints respond within 200ms for typical queries
3. **Data Integrity**: 100% of successful processing results are stored in database
4. **Error Rate**: <1% error rate for history API queries
5. **User Adoption**: Enable tracking of processing history usage patterns

## Open Questions

1. **Database Schema**: Should we implement soft deletes for processed videos or allow permanent deletion?
2. **Data Retention**: What is the desired retention period for processed video data?
3. **Duplicate Handling**: How should the system handle reprocessing of the same video URL?
4. **Migration Strategy**: Should existing users have a way to import historical data if available?
5. **Backup Strategy**: What backup and recovery procedures should be implemented?
6. **Performance Monitoring**: What database performance metrics should be tracked?