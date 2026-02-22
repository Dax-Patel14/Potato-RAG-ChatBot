# Quick Start Guide - Query Timing Logs 🚀

## What Changed?

Your system now automatically logs the time taken by each component. No configuration needed!

---

## Where to Find Logs

All logs are saved to: **`logs/query_timing.log`**

---

## How to View Logs

### Option 1: Real-Time Monitoring (Linux/Mac)
```bash
tail -f logs/query_timing.log
```

### Option 2: Real-Time Monitoring (Windows PowerShell)
```powershell
Get-Content logs/query_timing.log -Wait
```

### Option 3: View Full Log
```bash
cat logs/query_timing.log
```

### Option 4: Open in Editor
Just open `logs/query_timing.log` in VS Code

---

## What Gets Logged?

Every query is logged with:
1. **Query Start** - When processing begins
2. **Component Timings** - Each step's duration (in milliseconds)
3. **Query Complete** - Final summary with breakdown

---

## Example Log Entry

```
==================================================================================================================
QUERY_START | ID: a7c2e1b3 | Question: What are the symptoms of late blight?...
==================================================================================================================
TIMING | QUERY_PREPROCESS | Duration: 1.23ms | enhancements=1
TIMING | SEMANTIC_RETRIEVAL | Duration: 348.56ms | docs_retrieved=8
TIMING | DOCUMENT_RERANKING | Duration: 48.32ms | docs_processed=9
TIMING | MEMORY_SYNC | Duration: 12.45ms | history_items=2
TIMING | CHAIN_INVOCATION | Duration: 2345.67ms
TIMING | DB_ADD_AI_MESSAGE | Duration: 14.23ms | response_length=2847
==================================================================================================================
QUERY_COMPLETE | ID: a7c2e1b3 | Status: SUCCESS
TOTAL_TIME | 4466.78ms
TIMING_BREAKDOWN:
  CHAIN_INVOCATION                | 2345.67ms  (52.5%)
  SEMANTIC_RETRIEVAL              | 348.56ms   (7.8%)
  DOCUMENT_RERANKING              | 48.32ms    (1.1%)
  MEMORY_SYNC                     | 12.45ms    (0.3%)
==================================================================================================================
```

---

## Components Being Tracked

### Query Processing (`src/query_processor.py`)
- ⏱️ Generate standalone questions from chat context
- ⏱️ Expand queries with alternative phrasings
- ⏱️ Enhance with domain knowledge

### Retrieval (`src/retrieval.py`)
- ⏱️ Preprocess query with enhancements
- ⏱️ Semantic vector search
- ⏱️ BM25 keyword search
- ⏱️ Rerank documents by relevance

### Generation (`src/generation.py`)
- ⏱️ Synchronize chat history
- ⏱️ Invoke LLM chain
- ⏱️ Stream response chunks

### API Endpoints (`backend/main.py`)
- ⏱️ database operations
- ⏱️ History retrieval
- ⏱️ WebSocket streaming

---

## Finding Your Bottleneck

Look at the **TIMING_BREAKDOWN** section - it shows components from slowest to fastest!

**Example**: If you see:
```
CHAIN_INVOCATION    | 2345.67ms  (52.5%)
SEMANTIC_RETRIEVAL  | 348.56ms   (7.8%)
```

This means **LLM generation** is your bottleneck (using 52% of time).

---

## Common Scenarios

### Scenario 1: Slow Response Overall (>6 seconds)
```
Look at TIMING_BREAKDOWN:
- If CHAIN_INVOCATION is high → Use faster LLM model
- If SEMANTIC_RETRIEVAL is high → Reduce document count
- If both are high → Optimize both
```

### Scenario 2: Slow First Response
```
Watch for first_chunk_ms timing in WebSocket logs:
- If > 1000ms → LLM is slow
- If < 300ms → Good performance
```

### Scenario 3: Variable Performance
```
Check multiple queries' logs:
- If consistent → System is stable
- If varies widely → May have network issues
```

---

## Useful grep Commands

### Find all query starts
```bash
grep "QUERY_START" logs/query_timing.log
```

### Find slow queries (> 5 seconds)
```bash
grep "5[0-9][0-9][0-9]ms\|[6-9][0-9][0-9][0-9]ms" logs/query_timing.log
```

### Find specific query by ID
```bash
grep "a7c2e1b3" logs/query_timing.log
```

### Get all timing breakdowns
```bash
grep "TIMING_BREAKDOWN" logs/query_timing.log -A 10
```

### Find failed queries
```bash
grep "FAILED" logs/query_timing.log
```

### Count total queries
```bash
grep -c "QUERY_START" logs/query_timing.log
```

---

## Performance Expectations

**Typical timings:**
- Query processing: 50-250ms
- Retrieval: 200-600ms  
- LLM generation: 1-5 seconds
- **Total: 1.5-7 seconds**

If consistently slower, check bottleneck in TIMING_BREAKDOWN!

---

## File Details

| File | Purpose |
|------|---------|
| `logs/query_timing.log` | Main log file (auto-created) |
| `LOGGING_DOCUMENTATION.md` | Full technical documentation |
| `SAMPLE_LOGS.md` | Real example outputs |
| `IMPLEMENTATION_SUMMARY.md` | What was changed |
| `test_query_timing.py` | Test script |

---

## Test It Out

Run the test script to see logging in action:
```bash
python test_query_timing.py
```

Then check `logs/query_timing.log` to see the output!

---

## Next Steps

1. ✅ **Look at a few logs** - Get familiar with the format
2. ✅ **Identify bottleneck** - What's using most time?
3. ✅ **Optimize** - Based on the bottleneck
4. ✅ **Re-measure** - Compare new logs vs old ones
5. ✅ **Iterate** - Keep optimizing!

---

## Questions?

📖 **Full Documentation**: Read `LOGGING_DOCUMENTATION.md`
📋 **Examples**: Read `SAMPLE_LOGS.md`
🧪 **Test Code**: Read `test_query_timing.py`

---

**Happy Optimizing! 🚀**
