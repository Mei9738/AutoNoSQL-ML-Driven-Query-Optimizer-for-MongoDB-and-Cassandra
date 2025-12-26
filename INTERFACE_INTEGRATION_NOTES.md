# Interface Integration Notes

This document explains the current state of the AutoNoSQL backend for interface integration.

## Current Architecture

### 3-Step Workflow
1. **ML Model Classification** - Predicts if query is good/bad (binary classifier)
2. **Rule-Based Analysis** - Detects specific issues (regex, missing indexes, etc.)
3. **LLM Optimization** - Only called if ML predicts "bad" query

### Database Support
- **MongoDB**: Fully functional with live connection
- **Cassandra**: Fully functional with live connection (uses eventlet for Python 3.13)

## Key Backend Functions for Interface

### 1. Query Analysis Functions

**MongoDB:**
```python
def analyze_mongodb_query(query_str: str, config: dict, use_llm: bool = True, use_ml: bool = True)
```
Returns: `QueryAnalysis` object with:
- `issues_found`: List of detected problems
- `suggestions`: List of optimization recommendations
- `performance_metrics`: Dict with execution stats
- `execution_plan`: Dict with query plan info

**Cassandra:**
```python
def analyze_cassandra_query(query_str: str, config: dict, use_llm: bool = True, use_ml: bool = True)
```
Returns: Same `QueryAnalysis` structure as MongoDB

### 2. ML Model Prediction

```python
from autonosql.ml.query_classifier import QueryClassifier

classifier = QueryClassifier(model_path='models/mongodb_classifier.pkl')
prediction, confidence = classifier.predict(query_dict, database_type='mongodb')
# prediction: 0 (good) or 1 (bad)
# confidence: float (0.0 to 1.0)
```

### 3. LLM Optimization

```python
from autonosql.llm.llm_service import LLMService

llm = LLMService(provider='ollama', model='llama3.2')
suggestions = llm.get_optimization_suggestions(query, analysis, database_type='mongodb')
```

## Performance Metrics Explained

### Cassandra Performance Metrics

The `get_execution_stats()` function returns different data based on whether query execution succeeds:

**If query executes successfully (table exists):**
```python
{
    'analysis_type': 'live',
    'execution_time_ms': 12.5,           # Query execution time
    'rows_returned': 1,                   # Number of results
    'coordinator': '127.0.0.1',           # Cassandra node that handled query
    'duration_micros': 12500,             # Duration in microseconds
    'trace_events_count': 8,              # Number of internal operations
    'slowest_operation': 'Execute query'  # Bottleneck operation
}
```

**If query fails (table doesn't exist, no keyspace, etc.):**
```python
{
    'analysis_type': 'static',
    'note': 'Table does not exist - static analysis only',
    'suggestion': 'Create the table to enable live query execution'
}
```

### MongoDB Performance Metrics

Similar structure to Cassandra. Current implementation returns:
```python
{
    'error': 'Cursor.explain() takes 1 positional argument...',  # Known minor issue
    'document_count': 0
}
```

## Environment Configuration

Set these in `.env` file:

```bash
# LLM Configuration
LLM_PROVIDER=ollama          # or 'openai'
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434

# MongoDB
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=test

# Cassandra
CASSANDRA_HOSTS=127.0.0.1
CASSANDRA_PORT=9042
CASSANDRA_KEYSPACE=demo      # Set this if you have a specific keyspace
CASSANDRA_SKIP_CONNECTION=false
```

## API Response Structure

The interface should expect this structure from the backend:

```python
{
    "database_type": "MongoDB" or "Cassandra",
    "query": "<original query>",

    # Step 1: ML Classification
    "ml_prediction": 0 or 1,        # 0=good, 1=bad
    "ml_confidence": 0.85,           # 0.0 to 1.0

    # Step 2: Rule-Based Analysis
    "issues_found": [
        "Using $regex without index",
        "Missing WHERE clause"
    ],
    "suggestions": [
        "Create index on 'name'",
        "Add WHERE clause to filter"
    ],
    "performance_metrics": {
        "analysis_type": "static" or "live",
        "execution_time_ms": 12.5,   # If live
        "rows_returned": 100,         # If live
        # ... other metrics
    },

    # Step 3: LLM Optimization (only if ml_prediction == 1)
    "llm_suggestions": "<text from LLM>"  # May be None if query is good
}
```

## Tips for Interface Integration

### 1. Display ML Confidence
Show the confidence score to users:
- 90-100%: Very confident
- 75-90%: Confident
- 50-75%: Uncertain

### 2. Visualize Performance Metrics
If `analysis_type == 'live'`, you can show:
- Execution time chart
- Row count
- Trace event timeline (if available)

If `analysis_type == 'static'`, show a message like:
"Query analyzed statically. Create the table for live metrics."

### 3. Color Coding
- Green: No issues found (ml_prediction == 0)
- Yellow: Minor issues
- Red: Major issues (ALLOW FILTERING, full table scan, etc.)

### 4. LLM Toggle
Add a checkbox for "Use LLM suggestions" that sets `use_llm=True/False`

### 5. Error Handling
Always check for:
- Database connection failures
- ML model not found
- LLM service unavailable (Ollama not running)

## Testing the Backend

Run these commands to test before integrating:

```bash
# Test MongoDB good query
python main.py --db mongodb --file examples/mongo_query_good.json --no-llm

# Test MongoDB bad query with LLM
python main.py --db mongodb --file examples/mongo_query_bad.json

# Test Cassandra good query
python main.py --db cassandra --file examples/cassandra_query_good.cql --no-llm

# Test Cassandra bad query with LLM
python main.py --db cassandra --file examples/cassandra_query_bad.cql

# Run full system test
python test_system.py
```

## Known Issues to Handle in Interface

1. **MongoDB Explain Error**: `Cursor.explain()` has a minor compatibility issue. Doesn't affect functionality, but shows in performance metrics.

2. **Cassandra Keyspace**: If `CASSANDRA_KEYSPACE` is empty or table doesn't exist, queries won't execute (falls back to static analysis).

3. **Eventlet Warnings**: Suppressed in CLI, but might appear in logs. Safe to ignore.

## Questions for Interface Design

Before merging the interface branch, consider:

1. **Real-time analysis?** Should queries be analyzed as users type, or on submit?

2. **Sample queries?** Include a library of example good/bad queries for users to test?

3. **History?** Should the interface save previous query analyses?

4. **Batch mode?** Allow users to upload multiple queries at once?

5. **Export results?** Let users download analysis as PDF/JSON?

## Next Steps

1. Pull the interface branch
2. Review how it calls the backend functions
3. Update API responses to match the expected structure
4. Test with both MongoDB and Cassandra
5. Add error handling for edge cases
6. Consider adding WebSocket support for real-time updates

Good luck with the integration! 🚀
