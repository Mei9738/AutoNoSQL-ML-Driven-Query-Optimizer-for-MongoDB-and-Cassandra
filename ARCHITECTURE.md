# AutoNoSQL Architecture

## System Overview

AutoNoSQL implements a hybrid approach combining Machine Learning classification with rule-based analysis and LLM-powered optimization.

## Workflow

```
Query Input
    ↓
┌─────────────────────────────────────┐
│  Step 1: ML Classification          │
│  - Extract features from query      │
│  - Random Forest prediction         │
│  - Output: Good (0) or Bad (1)      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Step 2: Rule-Based Analysis        │
│  - Parse query structure            │
│  - Check execution plan             │
│  - Verify indexes                   │
│  - Detect anti-patterns             │
└─────────────────────────────────────┘
    ↓
    ├──→ If ML predicts "Good" (0) → Skip LLM, Done
    │
    └──→ If ML predicts "Bad" (1)  → Continue to Step 3
            ↓
        ┌─────────────────────────────────────┐
        │  Step 3: LLM Optimization           │
        │  - Build context from analysis      │
        │  - Query LLM (Ollama or OpenAI)     │
        │  - Generate human-readable advice   │
        └─────────────────────────────────────┘
```

## Components

### 1. ML Query Classifier (`autonosql/ml/query_classifier.py`)

**Purpose**: Predict if a query needs optimization before running expensive analysis

**Algorithm**: Random Forest Classifier
- **Training**: 20 example queries per database (10 good, 10 bad)
- **Features Extracted**:

  **MongoDB**:
  - has_regex: Query uses regex patterns
  - has_ne_nin: Uses $ne or $nin operators
  - has_or: Uses $or operator
  - filter_field_count: Number of fields in filter
  - has_sort, has_limit, has_projection
  - no_filter: Empty filter (scans all)

  **Cassandra**:
  - has_allow_filtering: Uses ALLOW FILTERING
  - select_all: Uses SELECT *
  - has_where: Has WHERE clause
  - has_in_clause: Uses IN operator
  - in_clause_size: Number of values in IN
  - no_where: Missing WHERE clause

**Output**:
- Prediction: 0 (good) or 1 (needs optimization)
- Confidence: Probability score (0.0 to 1.0)

### 2. Rule-Based Analyzers

#### MongoDB Analyzer (`autonosql/analyzers/mongodb_analyzer.py`)

- Connects to MongoDB
- Uses `.explain()` to get execution plan
- Detects:
  - Collection scans (COLLSCAN)
  - Regex queries without indexes
  - Inefficient operators ($ne, $nin)
  - Missing indexes on large collections

#### Cassandra Analyzer (`autonosql/analyzers/cassandra_analyzer.py`)

- Connects to Cassandra
- Parses CQL queries
- Detects:
  - ALLOW FILTERING usage
  - SELECT * patterns
  - Missing WHERE clauses
  - Non-partition key queries
  - Large IN clauses

### 3. LLM Service (`autonosql/llm/llm_service.py`)

**Purpose**: Generate human-readable optimization suggestions

**Providers Supported**:
- **Ollama** (local, free): Llama 3.2, Phi-3, etc.
- **OpenAI** (cloud, paid): GPT-3.5-turbo, GPT-4

**How it works**:
1. Receives query + analysis results
2. Builds contextual prompt with:
   - Query text
   - Issues found
   - Performance metrics
   - Initial suggestions
3. Sends to LLM
4. Returns formatted optimization advice

**When called**:
- Only when ML model predicts query is bad (1)
- OR when ML model unavailable and rules find issues

## Why This Architecture?

### Efficiency
- **ML Model is Fast**: Predictions in milliseconds
- **LLM is Slow**: Can take seconds, costs money (if using OpenAI)
- **Solution**: Only call LLM when needed

### Intelligence
- **ML Model**: Learns patterns from training data
- **Rules**: Domain-specific knowledge
- **LLM**: Natural language explanations and creative solutions

### Scalability
- ML model can handle thousands of queries/second
- LLM calls are minimized to only problematic queries
- Can process large query logs efficiently

## Training the ML Model

### Data Format

```json
{
  "queries": [
    {
      "query": {...},
      "label": 0,  // 0 = good, 1 = bad
      "description": "Why this is labeled this way"
    }
  ]
}
```

### Training Process

1. Load training data
2. Extract features for each query
3. Train Random Forest (100 trees, max depth 10)
4. Evaluate on training set
5. Save model to disk

### Adding More Training Data

To improve accuracy:
1. Add more queries to `training_data/mongodb_queries.json` or `cassandra_queries.json`
2. Re-run `python train_model.py --db mongodb` or `--db cassandra`
3. Model automatically saved to `models/`

## Extension Points

### Adding New Databases

1. Create new analyzer in `autonosql/analyzers/`
2. Inherit from `BaseQueryAnalyzer`
3. Implement:
   - `connect()`, `disconnect()`
   - `analyze_query()`
   - `get_execution_stats()`
   - `check_indexes()`
4. Add feature extraction to `QueryFeatureExtractor`
5. Create training data
6. Update main.py

### Adding New ML Models

Current: Random Forest

To use different model:
1. Update `QueryClassifier.__init__()` to use different sklearn model
2. Examples: GradientBoosting, SVM, Neural Network
3. Re-train with `train_model.py`

### Custom LLM Prompts

Edit `LLMService._build_prompt()` to customize how context is sent to LLM

## Performance Characteristics

**ML Classification**: <10ms per query
**Rule-Based Analysis**: 50-200ms (depends on database connection)
**LLM Generation**: 2-10 seconds (Ollama) or 1-3 seconds (OpenAI)

**Total for Good Query**: ~60-210ms (no LLM call)
**Total for Bad Query**: ~2-10 seconds (includes LLM)

This makes the system practical for real-time query optimization!
