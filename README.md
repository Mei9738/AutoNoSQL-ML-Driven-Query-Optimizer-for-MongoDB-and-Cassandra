# AutoNoSQL: ML-Driven Query Optimizer

A machine learning driven query optimizer for MongoDB and Cassandra that combines ML classification with LLM-powered optimization suggestions.

## How It Works

AutoNoSQL uses a **3-step intelligent workflow**:

1. **ML Model Classification** - A trained Random Forest classifier predicts if a query needs optimization
2. **Rule-Based Analysis** - Analyzes query structure, execution plans, and indexes
3. **LLM Optimization** - Only called if ML model predicts the query is bad, provides human-readable suggestions

This approach is efficient: the LLM (which can be slow/expensive) is only used when necessary!

## Features

- **ML-Powered Classification**: Trained model predicts query quality before expensive analysis
- **Multi-Database Support**: Analyze queries for both MongoDB and Cassandra
- **LLM Optimization**: Uses local LLM (Ollama) or OpenAI for intelligent suggestions
- **Rule-Based Analysis**: Detects missing indexes, inefficient patterns, and optimization opportunities
- **Easy CLI Interface**: Simple command-line tool for query optimization

## Setup

### Prerequisites

- Python 3.8+
- MongoDB (running locally)
- Cassandra (running locally)
- Ollama (for local LLM) OR OpenAI API key

### Installation

1. Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and configure:
```bash
cp .env.example .env
```

3. Edit `.env` with your settings:
   - For Ollama: Set `LLM_PROVIDER=ollama` and install Ollama from https://ollama.ai
   - For OpenAI: Set `LLM_PROVIDER=openai` and add your API key

### Install Ollama (Recommended for local LLM)

```bash
# Download from https://ollama.ai
# Then pull a small model:
ollama pull llama3.2
```

## Usage

### Step 1: Train the ML Models

Before using the optimizer, train the ML models:

```bash
# Train MongoDB classifier
python train_model.py --db mongodb

# Train Cassandra classifier
python train_model.py --db cassandra
```

This creates trained models in the `models/` directory.

### Step 2: Analyze Queries

```bash
# Analyze a MongoDB query
python main.py --db mongodb --file examples/mongo_query_bad.json

# Analyze a Cassandra query
python main.py --db cassandra --file examples/cassandra_query_bad.cql

# Or pass query directly
python main.py --db mongodb --query '{"collection":"users","operation":"find","filter":{"age":{"$gt":25}}}'
```

The tool will:
1. Use the ML model to classify the query
2. Run rule-based analysis
3. Call the LLM only if the query needs optimization

### Step 3: Test the System

```bash
# Run all system tests
python test_system.py

# Or run the demo
python demo.py
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models
python train_model.py --db mongodb
python train_model.py --db cassandra

# 3. Test it
python main.py --db mongodb --file examples/mongo_query_bad.json
```

## Project Structure

```
AutoNoSQL/
├── autonosql/
│   ├── analyzers/       # Query analyzers (MongoDB, Cassandra)
│   ├── llm/            # LLM integration (Ollama/OpenAI)
│   ├── ml/             # ML query classifier
│   └── utils/
├── training_data/       # Labeled query examples
├── models/             # Trained ML models
├── examples/           # Example queries
├── main.py            # CLI entry point
├── train_model.py     # Train ML models
└── test_system.py     # System tests
```

## Performance

- **ML Classification**: <10ms per query
- **Good Query Total**: ~100ms (skips LLM!)
- **Bad Query Total**: ~3-10s (includes LLM suggestions)
- **Efficiency Gain**: ~50% reduction in LLM calls

## Documentation

- `README.md` - This file (overview and quick start)
- `ARCHITECTURE.md` - Technical architecture details
- `DATASET_GUIDE.md` - Training data guide
- `TEST_RESULTS.md` - Test results and benchmarks

## License

MIT
