"""
Quick demo script to test AutoNoSQL functionality
Run this after setting up your environment
"""

import os
from dotenv import load_dotenv
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

def check_env_file():
    """Check if .env file exists"""
    if not os.path.exists('.env'):
        print(f"{Fore.YELLOW}Warning: .env file not found!")
        print(f"{Fore.YELLOW}Creating .env from .env.example...{Style.RESET_ALL}")

        if os.path.exists('.env.example'):
            with open('.env.example', 'r') as src, open('.env', 'w') as dst:
                dst.write(src.read())
            print(f"{Fore.GREEN}.env file created! Please edit it with your settings.{Style.RESET_ALL}\n")
        else:
            print(f"{Fore.RED}Error: .env.example not found!{Style.RESET_ALL}")
            return False
    return True

def test_mongodb():
    """Test MongoDB connection and analysis"""
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}Testing MongoDB Analysis")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")

    from autonosql.analyzers.mongodb_analyzer import MongoDBAnalyzer

    load_dotenv()
    uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
    database = os.getenv('MONGODB_DATABASE', 'test')

    analyzer = MongoDBAnalyzer({'uri': uri, 'database': database})

    if analyzer.connect():
        print(f"{Fore.GREEN}MongoDB connection successful!{Style.RESET_ALL}\n")

        # Test query
        test_query = {
            "collection": "test_collection",
            "operation": "find",
            "filter": {"name": {"$regex": "^test"}, "age": {"$ne": 30}}
        }

        print("Analyzing test query...")
        analysis = analyzer.analyze_query(str(test_query))
        print(analyzer.format_analysis(analysis))

        analyzer.disconnect()
        return True
    else:
        print(f"{Fore.RED}MongoDB connection failed!{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Make sure MongoDB is running on {uri}{Style.RESET_ALL}\n")
        return False

def test_cassandra():
    """Test Cassandra connection and analysis"""
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}Testing Cassandra Analysis")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")

    from autonosql.analyzers.cassandra_analyzer import CassandraAnalyzer

    load_dotenv()
    hosts = os.getenv('CASSANDRA_HOSTS', '127.0.0.1').split(',')
    port = int(os.getenv('CASSANDRA_PORT', '9042'))
    keyspace = os.getenv('CASSANDRA_KEYSPACE', 'system')

    analyzer = CassandraAnalyzer({'hosts': hosts, 'port': port, 'keyspace': keyspace})

    if analyzer.connect():
        print(f"{Fore.GREEN}Cassandra connection successful!{Style.RESET_ALL}\n")

        # Test query
        test_query = "SELECT * FROM system.peers WHERE peer IN ('127.0.0.1', '127.0.0.2') ALLOW FILTERING"

        print("Analyzing test query...")
        analysis = analyzer.analyze_query(test_query)
        print(analyzer.format_analysis(analysis))

        analyzer.disconnect()
        return True
    else:
        print(f"{Fore.RED}Cassandra connection failed!{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Make sure Cassandra is running on {hosts}:{port}{Style.RESET_ALL}\n")
        return False

def test_llm():
    """Test LLM integration"""
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}Testing LLM Integration")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")

    load_dotenv()
    provider = os.getenv('LLM_PROVIDER', 'ollama')

    print(f"LLM Provider: {provider}")

    try:
        from autonosql.llm.llm_service import LLMService
        from autonosql.base_analyzer import QueryAnalysis

        llm_kwargs = {}
        if provider == 'openai':
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                print(f"{Fore.YELLOW}Warning: OPENAI_API_KEY not set in .env{Style.RESET_ALL}")
                return False
            llm_kwargs['api_key'] = api_key
        else:
            llm_kwargs['base_url'] = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
            llm_kwargs['model'] = os.getenv('OLLAMA_MODEL', 'llama3.2')

        llm_service = LLMService(provider=provider, **llm_kwargs)

        # Create a dummy analysis
        test_analysis = QueryAnalysis(
            query="SELECT * FROM users WHERE age > 25 ALLOW FILTERING",
            database_type="Cassandra",
            issues_found=["Query uses ALLOW FILTERING (full table scan)"],
            suggestions=["Add appropriate indexes or redesign query"],
            performance_metrics={},
            execution_plan={}
        )

        print(f"{Fore.YELLOW}Getting LLM suggestions (this may take a moment)...{Style.RESET_ALL}\n")
        suggestions = llm_service.get_optimization_suggestions(test_analysis.query, test_analysis)

        print(f"{Fore.GREEN}LLM Response:{Style.RESET_ALL}")
        print(suggestions)
        print()

        return True

    except Exception as e:
        print(f"{Fore.RED}LLM test failed: {e}{Style.RESET_ALL}")
        if provider == 'ollama':
            print(f"{Fore.YELLOW}Make sure Ollama is running: ollama serve{Style.RESET_ALL}")
        return False

def main():
    """Run all tests"""
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}  AutoNoSQL - Quick Start Test")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")

    # Check environment
    if not check_env_file():
        return

    results = {
        'MongoDB': False,
        'Cassandra': False,
        'LLM': False
    }

    # Run tests
    try:
        results['MongoDB'] = test_mongodb()
    except Exception as e:
        print(f"{Fore.RED}MongoDB test error: {e}{Style.RESET_ALL}")

    try:
        results['Cassandra'] = test_cassandra()
    except Exception as e:
        print(f"{Fore.RED}Cassandra test error: {e}{Style.RESET_ALL}")

    try:
        results['LLM'] = test_llm()
    except Exception as e:
        print(f"{Fore.RED}LLM test error: {e}{Style.RESET_ALL}")

    # Print summary
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}  Test Summary")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")

    for component, success in results.items():
        status = f"{Fore.GREEN}PASS{Style.RESET_ALL}" if success else f"{Fore.RED}FAIL{Style.RESET_ALL}"
        print(f"{component:15} {status}")

    print()

    if all(results.values()):
        print(f"{Fore.GREEN}All tests passed! AutoNoSQL is ready to use.{Style.RESET_ALL}")
        print(f"\nTry running:")
        print(f"  python main.py --db mongodb --file examples/mongo_query_bad.json")
    else:
        print(f"{Fore.YELLOW}Some tests failed. Check the errors above and refer to SETUP_GUIDE.md{Style.RESET_ALL}")

if __name__ == '__main__':
    main()
