"""
AutoNoSQL - ML-Driven Query Optimizer for MongoDB and Cassandra
Main CLI application
"""

# Monkey patch eventlet FIRST before any other imports (for Cassandra driver compatibility)
import warnings
import sys
import os

# Suppress eventlet deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

try:
    import eventlet
    # Temporarily suppress stderr to hide RLock warnings (cosmetic only)
    stderr_backup = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    eventlet.monkey_patch()
    sys.stderr.close()
    sys.stderr = stderr_backup
except ImportError:
    pass

import argparse
import json
from dotenv import load_dotenv
from colorama import init, Fore, Style

from autonosql.analyzers.mongodb_analyzer import MongoDBAnalyzer

try:
    from autonosql.analyzers.cassandra_analyzer import CassandraAnalyzer
    CASSANDRA_AVAILABLE = True
except (ImportError, Exception) as e:
    CASSANDRA_AVAILABLE = False
    CASSANDRA_ERROR = str(e)
    CassandraAnalyzer = None

from autonosql.llm.llm_service import LLMService
from autonosql.ml.query_classifier import QueryClassifier

# Initialize colorama for colored output
init(autoreset=True)


def print_header():
    """Print application header"""
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{Fore.CYAN}  AutoNoSQL - ML-Driven Query Optimizer")
    print(f"{Fore.CYAN}  MongoDB & Cassandra Query Optimization Tool")
    print(f"{Fore.CYAN}{'='*70}\n{Style.RESET_ALL}")


def load_config():
    """Load configuration from .env file"""
    load_dotenv()
    return {
        'llm_provider': os.getenv('LLM_PROVIDER', 'ollama'),
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'ollama_base_url': os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
        'ollama_model': os.getenv('OLLAMA_MODEL', 'llama3.2'),
        'mongodb_uri': os.getenv('MONGODB_URI', 'mongodb://localhost:27017'),
        'mongodb_database': os.getenv('MONGODB_DATABASE', 'test'),
        'cassandra_hosts': os.getenv('CASSANDRA_HOSTS', '127.0.0.1').split(','),
        'cassandra_port': int(os.getenv('CASSANDRA_PORT', '9042')),
        'cassandra_keyspace': os.getenv('CASSANDRA_KEYSPACE', 'system'),
        'cassandra_skip_connection': os.getenv('CASSANDRA_SKIP_CONNECTION', 'false').lower() == 'true'
    }


def analyze_mongodb_query(query_str: str, config: dict, use_llm: bool = True, use_ml: bool = True):
    """Analyze a MongoDB query"""
    print(f"{Fore.YELLOW}Analyzing MongoDB query...{Style.RESET_ALL}\n")

    # Parse query for ML model
    import ast
    try:
        query_dict = json.loads(query_str) if isinstance(query_str, str) and query_str.startswith('{') else ast.literal_eval(query_str)
    except:
        query_dict = None

    # Step 1: Use ML model to predict if optimization is needed
    ml_prediction = None
    ml_confidence = None
    if use_ml and query_dict:
        print(f"{Fore.CYAN}Step 1: ML Model Classification{Style.RESET_ALL}")
        try:
            ml_classifier = QueryClassifier(model_path='models/mongodb_classifier.pkl')
            ml_prediction, ml_confidence = ml_classifier.predict(query_dict, database_type='mongodb')

            status = f"{Fore.GREEN}GOOD QUERY" if ml_prediction == 0 else f"{Fore.RED}NEEDS OPTIMIZATION"
            print(f"  ML Prediction: {status}{Style.RESET_ALL}")
            print(f"  Confidence: {ml_confidence:.2%}\n")
        except FileNotFoundError:
            print(f"  {Fore.YELLOW}ML model not found. Train it with: python train_model.py --db mongodb{Style.RESET_ALL}\n")
            ml_prediction = None
        except Exception as e:
            print(f"  {Fore.YELLOW}ML prediction error: {e}{Style.RESET_ALL}\n")
            ml_prediction = None

    # Step 2: Analyze the query with rule-based analyzer
    print(f"{Fore.CYAN}Step 2: Rule-Based Analysis{Style.RESET_ALL}")
    analyzer = MongoDBAnalyzer({
        'uri': config['mongodb_uri'],
        'database': config['mongodb_database']
    })

    # Connect to MongoDB
    if not analyzer.connect():
        print(f"{Fore.RED}Failed to connect to MongoDB{Style.RESET_ALL}")
        return

    try:
        # Analyze the query
        analysis = analyzer.analyze_query(query_str)

        # Print analysis results
        print(analyzer.format_analysis(analysis))

        # Step 3: Get LLM suggestions ONLY if ML model says query is bad
        if use_llm:
            should_use_llm = False

            if ml_prediction == 1:  # ML says it's bad
                should_use_llm = True
                print(f"{Fore.CYAN}Step 3: LLM Optimization (ML model detected issues){Style.RESET_ALL}\n")
            elif ml_prediction is None and (analysis.issues_found or analysis.suggestions):
                # ML model not available, fall back to rule-based decision
                should_use_llm = True
                print(f"{Fore.CYAN}Step 3: LLM Optimization (rule-based analysis detected issues){Style.RESET_ALL}\n")
            else:
                print(f"{Fore.GREEN}Step 3: Skipping LLM (ML model says query is good){Style.RESET_ALL}\n")

            if should_use_llm:
                print(f"{Fore.YELLOW}Getting LLM optimization suggestions...{Style.RESET_ALL}\n")

                llm_kwargs = {}
                if config['llm_provider'] == 'openai':
                    llm_kwargs['api_key'] = config['openai_api_key']
                else:
                    llm_kwargs['base_url'] = config['ollama_base_url']
                    llm_kwargs['model'] = config['ollama_model']

                llm_service = LLMService(provider=config['llm_provider'], **llm_kwargs)
                suggestions = llm_service.get_optimization_suggestions(query_str, analysis)

                print(f"{Fore.GREEN}{'='*70}")
                print(f"{Fore.GREEN}LLM Optimization Suggestions:")
                print(f"{Fore.GREEN}{'='*70}{Style.RESET_ALL}")
                print(suggestions)
                print()

    finally:
        analyzer.disconnect()


def analyze_cassandra_query(query_str: str, config: dict, use_llm: bool = True, use_ml: bool = True):
    """Analyze a Cassandra query"""
    if not CASSANDRA_AVAILABLE:
        print(f"{Fore.RED}Cassandra driver not available!{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Error: {CASSANDRA_ERROR}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Try: pip install cassandra-driver eventlet{Style.RESET_ALL}\n")
        return

    print(f"{Fore.YELLOW}Analyzing Cassandra query...{Style.RESET_ALL}\n")

    # Step 1: Use ML model to predict if optimization is needed
    ml_prediction = None
    ml_confidence = None
    if use_ml:
        print(f"{Fore.CYAN}Step 1: ML Model Classification{Style.RESET_ALL}")
        try:
            ml_classifier = QueryClassifier(model_path='models/cassandra_classifier.pkl')
            ml_prediction, ml_confidence = ml_classifier.predict(query_str, database_type='cassandra')

            status = f"{Fore.GREEN}GOOD QUERY" if ml_prediction == 0 else f"{Fore.RED}NEEDS OPTIMIZATION"
            print(f"  ML Prediction: {status}{Style.RESET_ALL}")
            print(f"  Confidence: {ml_confidence:.2%}\n")
        except FileNotFoundError:
            print(f"  {Fore.YELLOW}ML model not found. Train it with: python train_model.py --db cassandra{Style.RESET_ALL}\n")
            ml_prediction = None
        except Exception as e:
            print(f"  {Fore.YELLOW}ML prediction error: {e}{Style.RESET_ALL}\n")
            ml_prediction = None

    # Step 2: Rule-based analysis
    print(f"{Fore.CYAN}Step 2: Rule-Based Analysis{Style.RESET_ALL}")
    analyzer = CassandraAnalyzer({
        'hosts': config['cassandra_hosts'],
        'port': config['cassandra_port'],
        'keyspace': config['cassandra_keyspace'],
        'skip_connection': config.get('cassandra_skip_connection', False)
    })

    # Connect to Cassandra
    if not analyzer.connect():
        print(f"{Fore.RED}Failed to connect to Cassandra{Style.RESET_ALL}")
        return

    try:
        # Analyze the query
        analysis = analyzer.analyze_query(query_str)

        # Print analysis results
        print(analyzer.format_analysis(analysis))

        # Step 3: Get LLM suggestions ONLY if ML model says query is bad
        if use_llm:
            should_use_llm = False

            if ml_prediction == 1:  # ML says it's bad
                should_use_llm = True
                print(f"{Fore.CYAN}Step 3: LLM Optimization (ML model detected issues){Style.RESET_ALL}\n")
            elif ml_prediction is None and (analysis.issues_found or analysis.suggestions):
                # ML model not available, fall back to rule-based decision
                should_use_llm = True
                print(f"{Fore.CYAN}Step 3: LLM Optimization (rule-based analysis detected issues){Style.RESET_ALL}\n")
            else:
                print(f"{Fore.GREEN}Step 3: Skipping LLM (ML model says query is good){Style.RESET_ALL}\n")

            if should_use_llm:
                print(f"{Fore.YELLOW}Getting LLM optimization suggestions...{Style.RESET_ALL}\n")

                llm_kwargs = {}
                if config['llm_provider'] == 'openai':
                    llm_kwargs['api_key'] = config['openai_api_key']
                else:
                    llm_kwargs['base_url'] = config['ollama_base_url']
                    llm_kwargs['model'] = config['ollama_model']

                llm_service = LLMService(provider=config['llm_provider'], **llm_kwargs)
                suggestions = llm_service.get_optimization_suggestions(query_str, analysis)

                print(f"{Fore.GREEN}{'='*70}")
                print(f"{Fore.GREEN}LLM Optimization Suggestions:")
                print(f"{Fore.GREEN}{'='*70}{Style.RESET_ALL}")
                print(suggestions)
                print()

    finally:
        analyzer.disconnect()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='AutoNoSQL - ML-Driven Query Optimizer for MongoDB and Cassandra',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze MongoDB query
  python main.py --db mongodb --query '{"collection":"users","operation":"find","filter":{"age":{"$gt":25}}}'

  # Analyze Cassandra query
  python main.py --db cassandra --query "SELECT * FROM users WHERE age > 25 ALLOW FILTERING"

  # Analyze query from file
  python main.py --db mongodb --file examples/mongo_query.json

  # Skip LLM suggestions
  python main.py --db mongodb --query '...' --no-llm
        """
    )

    parser.add_argument('--db', required=True, choices=['mongodb', 'cassandra'],
                        help='Database type')
    parser.add_argument('--query', help='Query string to analyze')
    parser.add_argument('--file', help='File containing query to analyze')
    parser.add_argument('--no-llm', action='store_true',
                        help='Skip LLM optimization suggestions')

    args = parser.parse_args()

    print_header()

    # Load configuration
    config = load_config()

    # Get query from args or file
    query_str = None
    if args.query:
        query_str = args.query
    elif args.file:
        try:
            with open(args.file, 'r') as f:
                query_str = f.read().strip()
        except FileNotFoundError:
            print(f"{Fore.RED}Error: File '{args.file}' not found{Style.RESET_ALL}")
            return
        except Exception as e:
            print(f"{Fore.RED}Error reading file: {e}{Style.RESET_ALL}")
            return
    else:
        print(f"{Fore.RED}Error: Either --query or --file must be provided{Style.RESET_ALL}")
        parser.print_help()
        return

    # Analyze based on database type
    use_llm = not args.no_llm

    if args.db == 'mongodb':
        analyze_mongodb_query(query_str, config, use_llm)
    elif args.db == 'cassandra':
        analyze_cassandra_query(query_str, config, use_llm)


if __name__ == '__main__':
    main()
