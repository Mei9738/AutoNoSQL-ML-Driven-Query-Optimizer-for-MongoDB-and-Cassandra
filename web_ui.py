"""
AutoNoSQL - Web UI for Query Optimization
Simple Flask server for browser-based query analysis
"""

# Monkey patch eventlet FIRST before any other imports (for Cassandra driver compatibility)
import warnings
import sys

warnings.filterwarnings('ignore', category=DeprecationWarning)

try:
    import eventlet
    # Temporarily suppress stderr to hide RLock warnings (cosmetic only)
    stderr_backup = sys.stderr
    sys.stderr = open(os.devnull, 'w') if 'os' in dir() else sys.stderr
    eventlet.monkey_patch()
    if stderr_backup != sys.stderr:
        sys.stderr.close()
        sys.stderr = stderr_backup
except ImportError:
    pass

import json
import ast
import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

from autonosql.analyzers.mongodb_analyzer import MongoDBAnalyzer
try:
    from autonosql.analyzers.cassandra_analyzer import CassandraAnalyzer
    CASSANDRA_AVAILABLE = True
except (ImportError, Exception) as e:
    CASSANDRA_AVAILABLE = False
    CASSANDRA_ERROR = str(e)
    CassandraAnalyzer = None

from autonosql.ml.query_classifier import QueryClassifier
from autonosql.llm.llm_service import LLMService

app = Flask(__name__)
load_dotenv()


def get_config():
    """Load configuration from environment"""
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


@app.route('/')
def index():
    """Main page with query input"""
    return render_template('index.html', cassandra_available=CASSANDRA_AVAILABLE)


@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze a query and return results"""
    data = request.json
    db_type = data.get('db_type', 'mongodb')
    query_str = data.get('query', '').strip()
    use_llm = data.get('use_llm', True)
    
    if not query_str:
        return jsonify({'error': 'No query provided'}), 400
    
    config = get_config()
    
    try:
        if db_type == 'mongodb':
            result = analyze_mongodb(query_str, config, use_llm)
        else:
            result = analyze_cassandra(query_str, config, use_llm)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def analyze_mongodb(query_str: str, config: dict, use_llm: bool):
    """Analyze MongoDB query"""
    result = {
        'db_type': 'MongoDB',
        'query': query_str,
        'ml_prediction': None,
        'ml_confidence': None,
        'issues': [],
        'suggestions': [],
        'execution_stats': {},
        'llm_suggestions': None
    }
    
    # Parse query for ML model
    try:
        query_dict = json.loads(query_str) if query_str.startswith('{') else ast.literal_eval(query_str)
    except:
        query_dict = None
    
    # Step 1: ML Classification
    if query_dict:
        try:
            classifier = QueryClassifier(model_path='models/mongodb_classifier.pkl')
            prediction, confidence = classifier.predict(query_dict, database_type='mongodb')
            result['ml_prediction'] = 'GOOD' if prediction == 0 else 'NEEDS OPTIMIZATION'
            result['ml_confidence'] = round(confidence * 100, 1)
        except FileNotFoundError:
            result['ml_prediction'] = 'MODEL NOT TRAINED'
        except Exception as e:
            result['ml_prediction'] = f'Error: {str(e)}'
    
    # Step 2: Rule-based Analysis
    analyzer = MongoDBAnalyzer({
        'uri': config['mongodb_uri'],
        'database': config['mongodb_database']
    })
    
    if not analyzer.connect():
        result['error'] = 'Failed to connect to MongoDB'
        return result
    
    try:
        analysis = analyzer.analyze_query(query_str)
        result['issues'] = analysis.issues_found
        result['suggestions'] = analysis.suggestions
        result['execution_stats'] = getattr(analysis, 'execution_stats', {})
        
        # Step 3: LLM suggestions if needed
        if use_llm and (result['ml_prediction'] == 'NEEDS OPTIMIZATION' or 
                        (result['ml_prediction'] in [None, 'MODEL NOT TRAINED'] and analysis.issues_found)):
            try:
                llm_kwargs = {}
                if config['llm_provider'] == 'openai':
                    llm_kwargs['api_key'] = config['openai_api_key']
                else:
                    llm_kwargs['base_url'] = config['ollama_base_url']
                    llm_kwargs['model'] = config['ollama_model']
                
                llm_service = LLMService(provider=config['llm_provider'], **llm_kwargs)
                result['llm_suggestions'] = llm_service.get_optimization_suggestions(query_str, analysis)
            except Exception as e:
                result['llm_suggestions'] = f'LLM Error: {str(e)}'
    finally:
        analyzer.disconnect()
    
    return result


def analyze_cassandra(query_str: str, config: dict, use_llm: bool):
    """Analyze Cassandra query"""
    if not CASSANDRA_AVAILABLE:
        return {'error': f'Cassandra driver not available: {CASSANDRA_ERROR}'}
    
    result = {
        'db_type': 'Cassandra',
        'query': query_str,
        'ml_prediction': None,
        'ml_confidence': None,
        'issues': [],
        'suggestions': [],
        'execution_stats': {},
        'llm_suggestions': None
    }
    
    # Step 1: ML Classification
    try:
        classifier = QueryClassifier(model_path='models/cassandra_classifier.pkl')
        prediction, confidence = classifier.predict(query_str, database_type='cassandra')
        result['ml_prediction'] = 'GOOD' if prediction == 0 else 'NEEDS OPTIMIZATION'
        result['ml_confidence'] = round(confidence * 100, 1)
    except FileNotFoundError:
        result['ml_prediction'] = 'MODEL NOT TRAINED'
    except Exception as e:
        result['ml_prediction'] = f'Error: {str(e)}'
    
    # Step 2: Rule-based Analysis
    analyzer = CassandraAnalyzer({
        'hosts': config['cassandra_hosts'],
        'port': config['cassandra_port'],
        'keyspace': config['cassandra_keyspace'],
        'skip_connection': config.get('cassandra_skip_connection', False)
    })
    
    if not analyzer.connect():
        result['error'] = 'Failed to connect to Cassandra'
        return result
    
    try:
        analysis = analyzer.analyze_query(query_str)
        result['issues'] = analysis.issues_found
        result['suggestions'] = analysis.suggestions
        result['execution_stats'] = getattr(analysis, 'execution_stats', {})
        
        # Step 3: LLM suggestions if needed
        if use_llm and (result['ml_prediction'] == 'NEEDS OPTIMIZATION' or 
                        (result['ml_prediction'] in [None, 'MODEL NOT TRAINED'] and analysis.issues_found)):
            try:
                llm_kwargs = {}
                if config['llm_provider'] == 'openai':
                    llm_kwargs['api_key'] = config['openai_api_key']
                else:
                    llm_kwargs['base_url'] = config['ollama_base_url']
                    llm_kwargs['model'] = config['ollama_model']
                
                llm_service = LLMService(provider=config['llm_provider'], **llm_kwargs)
                result['llm_suggestions'] = llm_service.get_optimization_suggestions(query_str, analysis)
            except Exception as e:
                result['llm_suggestions'] = f'LLM Error: {str(e)}'
    finally:
        analyzer.disconnect()
    
    return result


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  AutoNoSQL Web UI")
    print("  Open http://localhost:5000 in your browser")
    print("="*60 + "\n")
    app.run(debug=True, port=5000)
