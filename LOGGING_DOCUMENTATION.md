# Query Timing Logging System 📊

## Overview

A comprehensive query timing logging system has been implemented to track the performance of each component in your multi-modal potato disease RAG system. All logs are saved to `logs/query_timing.log`.

## Architecture Components with Timing

### 1. **Query Processing** (`src/query_processor.py`)
Tracks the time taken for preprocessing user queries:

- **STANDALONE_QUESTION**: Time to convert follow-up questions to standalone questions using LLM
  - Measures: duration, history turns, generation status
  
- **QUERY_EXPANSION**: Time to generate alternative query phrasings
  - Measures: duration, expanded count, total queries
  
- **FULL_PREPROCESSING**: Total time for the complete preprocessing pipeline
  - Measures: duration, chat history presence, expanded queries count

### 2. **Retrieval** (`src/retrieval.py`)
Tracks document retrieval and reranking operations:

- **QUERY_PREPROCESS**: Query enhancement with domain knowledge
  - Measures: duration, enhancements applied, domains used
  
- **SEMANTIC_RETRIEVAL**: Vector-based semantic search
  - Measures: duration, documents retrieved
  
- **ENSEMBLE_RETRIEVAL**: Hybrid semantic + BM25 search
  - Measures: duration, documents retrieved
  
- **FALLBACK_RETRIEVAL**: Secondary retrieval when initial results are insufficient
  - Measures: duration, documents retrieved, reason
  
- **DOCUMENT_RERANKING**: Re-ranking documents by relevance
  - Measures: duration, documents processed, documents returned
  
- **RETRIEVAL** (Overall): Total retrieval latency
  - Measures: documents returned, methods used

### 3. **Generation** (`src/generation.py`)
Tracks LLM response generation and memory operations:

- **MEMORY_SYNC**: Synchronizing chat history with memory buffer
  - Measures: duration, history items
  
- **CHAIN_INVOCATION**: Full chain invoke (non-streaming)
  - Measures: duration, answer length, source documents count
  
- **MEMORY_SYNC_STREAM**: Memory sync during streaming
  - Measures: duration, history items
  
- **STREAMING_CHUNKS**: Chunk-by-chunk streaming output
  - Measures: duration, chunk count, answer length
  
- **GENERATION**: Total generation time
  - Measures: duration, chunk count (optional), token count (optional)

### 4. **API Layer** (`backend/main.py`)
Tracks end-to-end request processing:

- **DB_ADD_USER_MESSAGE**: Database insert for user message
  - Measures: duration
  
- **HISTORY_RETRIEVAL**: Fetching chat history from database
  - Measures: duration, history items count
  
- **AI_CHAIN_INVOCATION**: LLM chain processing
  - Measures: duration
  
- **DB_ADD_AI_MESSAGE**: Database insert for AI response
  - Measures: duration, response length
  
- **STREAMING_COMPLETE**: Full streaming operation
  - Measures: duration, chunks sent, response length, first chunk time
  
- **QUERY_START**: Query processing begins (request header)
  - Query ID and question preview
  
- **QUERY_COMPLETE**: Query processing ends (request summary)
  - Query ID, status, total time, timing breakdown

## Log Format

All logs follow this format:
```
YYYY-MM-DD HH:MM:SS | LEVEL | LOGGER_NAME | MESSAGE
```

### Example Log Entry

```
2025-02-22 10:30:45,123 | INFO | retrieval | TIMING | SEMANTIC_RETRIEVAL | Duration: 342.15ms | docs_retrieved=8
```

## Log Metrics Interpretation

### Performance Indicators

1. **Query Processing** typically takes: 50-300ms
   - Depends on LLM availability and chat history length

2. **Retrieval** typically takes: 200-600ms
   - Includes semantic search, BM25, reranking
   - Slower with more documents to process

3. **Generation** typically takes: 1000-5000ms
   - First chunk usually appears in 300-1000ms
   - Total depends on response length

4. **Total Request** typically takes: 1500-7000ms
   - Sum of retrieval + generation + overhead

## Analyzing Logs

### View Real-time Logs
```bash
tail -f logs/query_timing.log
```

### View All Logs for a Query
```bash
grep "QUERY_START" logs/query_timing.log
```

### Extract Timing Summary
```bash
grep "TIMING_BREAKDOWN" logs/query_timing.log -A 20
```

### Find Slow Requests
```bash
grep "QUERY_COMPLETE.*FAILED\|QUERY_COMPLETE.*\|[5-9][0-9][0-9][0-9]ms" logs/query_timing.log
```

### Component Performance Analysis
```bash
grep "TIMING" logs/query_timing.log | grep "SEMANTIC_RETRIEVAL\|EXPANSION\|GENERATION"
```

## Configuration

### Log File Location
- Default: `logs/query_timing.log`
- Configurable in `src/logging_utils.py`

### Log Rotation
- Max size: 10MB per file
- Backup count: 5 files
- Old files are automatically archived

### Log Level
- Default: DEBUG (logs everything)
- Change in `setup_logger()` function

## Identifying Bottlenecks

Based on the timing breakdowns, you can identify which component is causing latency:

1. **High RETRIEVAL time** → Optimize vector search or reduce document count
2. **High QUERY_EXPANSION time** → Consider disabling query expansion or using faster LLM
3. **High GENERATION time** → Use faster LLM model or reduce response length
4. **High MEMORY_SYNC time** → Reduce chat history size

## API Response Timings

### Non-streaming Endpoint
Returns `timings` object with:
```json
{
  "timings": {
    "total_ms": 3500
  }
}
```

### WebSocket Streaming
Returns `timings` object with:
```json
{
  "timings": {
    "first_chunk_ms": 450,
    "total_ms": 2800
  }
}
```

## Integration with Monitoring

The logging can be integrated with external monitoring systems:

1. Parse `query_timing.log` with tools like ELK Stack
2. Track metrics with Prometheus
3. Alert on high latency values
4. Dashboard visualization

## Best Practices

1. **Regular Analysis**: Check logs weekly to identify trends
2. **Baseline Metrics**: Establish baseline performance metrics
3. **Alert Thresholds**: Set up alerts for requests exceeding thresholds
4. **Archive Old Logs**: Regularly archive logs for historical analysis
5. **Query Optimization**: Use timing data to optimize queries

## Troubleshooting

### No logs appearing
- Check if `logs/` directory exists and is writable
- Verify logger is initialized before use
- Check console output for initialization messages

### Incomplete timing data
- Some components may skip logging if execution is very fast
- Check for exceptions in ERROR level logs

### High variance in timings
- Network latency can affect API response times
- LLM inference time varies based on prompt complexity
- Document count affects retrieval time

## Future Enhancements

Potential improvements to the logging system:
- [ ] Add request ID tracking across all logs
- [ ] Implement distributed tracing
- [ ] Add custom timing annotations via decorators
- [ ] Create real-time dashboard
- [ ] Add performance alerts
