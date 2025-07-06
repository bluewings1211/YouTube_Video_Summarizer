# Architecture and Flow: Transcript Acquisition

This document outlines the process flow for acquiring YouTube video transcripts within this application. The flowchart below details the steps, decision points, and error handling mechanisms, particularly focusing on the flow *without* a proxy enabled.

## Transcript Acquisition Flowchart

The following diagram illustrates the robust, multi-stage process designed to reliably fetch the best possible transcript while handling various potential failures.

```mermaid
graph TD
    A[Start: Provide YouTube URL] --> B{1. Validate URL & Extract Video ID};
    B --> |URL Invalid| C[Fail: Throw InvalidYouTubeURLError];
    B --> |URL Valid| D{2. Check Video for Support};
    
    D --> E[2a. Extract Video Metadata];
    E --> |Extraction Fails| F[Fail: Throw YouTubeTranscriptError];
    E --> |Extraction Succeeds| G{2b. Analyze Video Characteristics};
    G --> |Video is Too Long / Private / Live| H[Fail: Throw specific UnsupportedVideoTypeError];
    G --> |Checks Pass| I{3. Build Three-Tier Strategy};

    I --> J[3a. Fetch List of All Available Transcripts];
    J --> |API Request Fails| K{3b. Enter Retry Process (RetryManager)};
    K --> |Retry Succeeds| J;
    K --> |Retry Ultimately Fails| L[Fail: Throw YouTubeTranscriptError];
    J --> |No Transcripts Available at All| M[Fail: Throw NoTranscriptAvailableError];
    J --> |List Retrieved Successfully| N[3c. Sort Transcripts by Quality & Language Preference];
    
    N --> O{4. Attempt to Fetch Transcript Based on Strategy};
    O --> P[4a. Select Highest Priority Option (e.g., Manual English)];
    P --> Q{4b. Call API to Fetch Full Transcript Content};
    
    Q --> |API Request Fails (e.g., Rate Limit)| R{4c. Enter Retry Process (RetryManager)};
    R --> |Retry Succeeds| Q;
    R --> |Retry Ultimately Fails| S{4d. Fallback: Attempt Next Option};
    
    Q --> |Fetch Succeeds| T[Success: Return Transcript Result];
    
    S --> |Next option exists| P;
    S --> |All options have been attempted and failed| U[Fail: Throw NoTranscriptAvailableError];

    subgraph "Core Error Handling & Retries (Managed by RetryManager)"
        direction LR
        K;
        R;
    end

    subgraph "Initial Video Filtering (Managed by YouTubeVideoMetadataExtractor)"
        direction TB
        D; E; G;
    end

    subgraph "Intelligent Transcript Selection (Managed by ThreeTierTranscriptStrategy)"
        direction TB
        I; J; N;
    end
```

### Flow Explanation

1.  **Validation and Filtering (Nodes B, D, G):**
    *   The process starts with a YouTube URL. The system first validates its format and extracts the unique `video_id`.
    *   It then fetches the video's metadata to perform a preliminary screening. This is a critical step to avoid wasting resources on videos that are guaranteed to fail.
    *   **Error Handling:** If the video is private, a live stream, or exceeds the configured maximum duration, the process terminates early with a specific, informative error.

2.  **Three-Tier Strategy Building (Nodes I, J, N):**
    *   If the video passes the initial checks, the system queries the YouTube API to get a list of all available transcripts (e.g., manual English, auto-generated English, auto-generated Japanese).
    *   **Error Handling:** During this query, if a network error or temporary API issue occurs, the `RetryManager` automatically performs exponential backoff retries. It waits for a short interval, retries, and doubles the waiting time on subsequent failures, up to a maximum attempt limit. The process only fails if the retries are exhausted.
    *   Once the list is retrieved, the `ThreeTierTranscriptStrategy` sorts them based on a quality pyramid:
        *   **Tier 1 (Highest Priority):** Manually created transcripts in preferred languages.
        *   **Tier 2:** Auto-generated transcripts in preferred languages.
        *   **Tier 3:** Transcripts in other languages (which can be translated).

3.  **Strategic Fetching (Nodes O, P, Q):**
    *   The system picks the highest-priority option from the sorted list and attempts to fetch its full content.
    *   **Error Handling:** This is the step most likely to encounter rate limits. If an error occurs here, the `RetryManager` again initiates the retry process.

4.  **Intelligent Fallback (Node S):**
    *   If fetching the highest-priority option (e.g., "Manual English") ultimately fails even after retries, the system **does not give up**.
    *   It intelligently falls back to try the **next option** in the list (e.g., "Auto-generated English").
    *   This continues until a transcript is successfully fetched or all available options have been exhausted.

---

## Proxy Architecture: `ProxyRotationManager`

When a proxy is configured, every API call in the flow (Nodes E, J, Q) is routed through the `ProxyRotationManager`. This section details its specific behavior regarding hosts and ports.

### Key Design Principle: URL as an Independent Entity

The `ProxyRotationManager` is designed to treat each full Proxy URL provided in the configuration as a **single, independent, and complete entity**. 

This leads to two important points:

1.  **No Automatic Port Rotation:** The manager **does not** automatically cycle through different ports for a single given hostname. There is no configuration parameter to specify a port range (e.g., `8001-8005`).

2.  **Rotation Unit is the Full URL:** The rotation and health checks are performed on the entire URL string (`scheme://host:port`) listed in the configuration.

### How to Achieve Multi-Port Rotation for a Single Host

To achieve the effect of rotating through multiple ports on the same proxy server, you must **explicitly list each host-port combination** as a separate entry in the `PROXY_URLS` environment variable, separated by commas.

**Correct Configuration Example:**

```ini
# .env.development or .env.production

PROXY_ENABLED=true

# Each URL, including its port, is treated as a separate proxy for rotation.
PROXY_URLS=http://your-proxy-server.com:8001,http://your-proxy-server.com:8002,http://your-proxy-server.com:8003,https://another-proxy.com:9000
```

When configured this way, the `ProxyRotationManager` will:

*   Recognize four independent proxy entries in its pool.
*   Individually monitor the health of each one. If `http://your-proxy-server.com:8002` becomes unresponsive, it will be temporarily removed from rotation, while the other three continue to be used.
*   Rotate requests among the currently healthy proxies in the list.

This explicit, URL-based approach provides maximum flexibility and precise, granular control over the health and usage of each individual proxy endpoint.