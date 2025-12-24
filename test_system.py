"""
Simple system test - verifies everything is working
Run this to test your AutoNoSQL system
"""

import os
import sys
from colorama import init, Fore, Style

init(autoreset=True)


def test_imports():
    """Test 1: Check all imports work"""
    print(f"{Fore.CYAN}Test 1: Checking imports...{Style.RESET_ALL}")
    try:
        from autonosql.analyzers.mongodb_analyzer import MongoDBAnalyzer
        from autonosql.ml.query_classifier import QueryClassifier
        from autonosql.llm.llm_service import LLMService
        print(f"{Fore.GREEN}  PASS - All imports successful{Style.RESET_ALL}")
        return True
    except Exception as e:
        print(f"{Fore.RED}  FAIL - Import error: {e}{Style.RESET_ALL}")
        return False


def test_models_exist():
    """Test 2: Check if ML models are trained"""
    print(f"\n{Fore.CYAN}Test 2: Checking ML models...{Style.RESET_ALL}")
    mongodb_model = os.path.exists('models/mongodb_classifier.pkl')
    cassandra_model = os.path.exists('models/cassandra_classifier.pkl')

    if mongodb_model:
        print(f"{Fore.GREEN}  PASS - MongoDB model found{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}  WARN - MongoDB model not found (run: python train_model.py --db mongodb){Style.RESET_ALL}")

    if cassandra_model:
        print(f"{Fore.GREEN}  PASS - Cassandra model found{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}  WARN - Cassandra model not found (run: python train_model.py --db cassandra){Style.RESET_ALL}")

    return mongodb_model or cassandra_model


def test_mongodb_connection():
    """Test 3: Check MongoDB connection"""
    print(f"\n{Fore.CYAN}Test 3: Testing MongoDB connection...{Style.RESET_ALL}")
    try:
        from autonosql.analyzers.mongodb_analyzer import MongoDBAnalyzer
        analyzer = MongoDBAnalyzer({'uri': 'mongodb://localhost:27017', 'database': 'test'})
        if analyzer.connect():
            analyzer.disconnect()
            print(f"{Fore.GREEN}  PASS - MongoDB connected successfully{Style.RESET_ALL}")
            return True
        else:
            print(f"{Fore.YELLOW}  WARN - MongoDB not running or not accessible{Style.RESET_ALL}")
            return False
    except Exception as e:
        print(f"{Fore.RED}  FAIL - MongoDB error: {e}{Style.RESET_ALL}")
        return False


def test_ml_prediction():
    """Test 4: Test ML model prediction"""
    print(f"\n{Fore.CYAN}Test 4: Testing ML prediction...{Style.RESET_ALL}")
    try:
        from autonosql.ml.query_classifier import QueryClassifier

        # Good query
        classifier = QueryClassifier(model_path='models/mongodb_classifier.pkl')
        good_query = {"collection": "users", "operation": "find", "filter": {"_id": "123"}}
        prediction, confidence = classifier.predict(good_query, database_type='mongodb')

        print(f"  Good query prediction: {'GOOD' if prediction == 0 else 'BAD'} ({confidence*100:.1f}% confidence)")

        # Bad query
        bad_query = {"collection": "users", "operation": "find", "filter": {"name": {"$regex": "^test"}}}
        prediction, confidence = classifier.predict(bad_query, database_type='mongodb')

        print(f"  Bad query prediction:  {'GOOD' if prediction == 0 else 'BAD'} ({confidence*100:.1f}% confidence)")
        print(f"{Fore.GREEN}  PASS - ML predictions working{Style.RESET_ALL}")
        return True
    except FileNotFoundError:
        print(f"{Fore.YELLOW}  SKIP - Model not trained yet{Style.RESET_ALL}")
        return False
    except Exception as e:
        print(f"{Fore.RED}  FAIL - Prediction error: {e}{Style.RESET_ALL}")
        return False


def test_ollama():
    """Test 5: Check if Ollama is available"""
    print(f"\n{Fore.CYAN}Test 5: Testing Ollama connection...{Style.RESET_ALL}")
    try:
        import requests
        response = requests.get('http://localhost:11434/api/version', timeout=2)
        if response.status_code == 200:
            version = response.json().get('version', 'unknown')
            print(f"{Fore.GREEN}  PASS - Ollama running (version {version}){Style.RESET_ALL}")
            return True
        else:
            print(f"{Fore.YELLOW}  WARN - Ollama not responding{Style.RESET_ALL}")
            return False
    except:
        print(f"{Fore.YELLOW}  WARN - Ollama not running (optional - LLM won't work){Style.RESET_ALL}")
        return False


def test_full_workflow():
    """Test 6: Test complete workflow"""
    print(f"\n{Fore.CYAN}Test 6: Testing complete workflow...{Style.RESET_ALL}")
    try:
        from autonosql.analyzers.mongodb_analyzer import MongoDBAnalyzer
        from autonosql.ml.query_classifier import QueryClassifier

        # Connect to MongoDB
        analyzer = MongoDBAnalyzer({'uri': 'mongodb://localhost:27017', 'database': 'test'})
        if not analyzer.connect():
            print(f"{Fore.YELLOW}  SKIP - MongoDB not available{Style.RESET_ALL}")
            return False

        # Test query
        query = '{"collection": "users", "operation": "find", "filter": {"email": "test@example.com"}}'

        # Step 1: ML prediction
        import ast
        query_dict = ast.literal_eval(query)
        classifier = QueryClassifier(model_path='models/mongodb_classifier.pkl')
        prediction, confidence = classifier.predict(query_dict, database_type='mongodb')
        print(f"  Step 1 - ML: {'GOOD' if prediction == 0 else 'BAD'} ({confidence*100:.1f}%)")

        # Step 2: Analysis
        analysis = analyzer.analyze_query(query)
        print(f"  Step 2 - Analysis: {len(analysis.issues_found)} issues, {len(analysis.suggestions)} suggestions")

        # Step 3: Would call LLM if needed
        if prediction == 1:
            print(f"  Step 3 - LLM: Would be called (query needs optimization)")
        else:
            print(f"  Step 3 - LLM: Skipped (query is good)")

        analyzer.disconnect()
        print(f"{Fore.GREEN}  PASS - Complete workflow working{Style.RESET_ALL}")
        return True

    except FileNotFoundError:
        print(f"{Fore.YELLOW}  SKIP - Model not trained{Style.RESET_ALL}")
        return False
    except Exception as e:
        print(f"{Fore.RED}  FAIL - Workflow error: {e}{Style.RESET_ALL}")
        return False


def main():
    print(f"\n{Fore.GREEN}{'='*70}")
    print(f"{Fore.GREEN}  AutoNoSQL System Test")
    print(f"{Fore.GREEN}{'='*70}{Style.RESET_ALL}\n")

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("ML Models", test_models_exist()))
    results.append(("MongoDB Connection", test_mongodb_connection()))
    results.append(("ML Predictions", test_ml_prediction()))
    results.append(("Ollama/LLM", test_ollama()))
    results.append(("Full Workflow", test_full_workflow()))

    # Summary
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{Fore.CYAN}  Test Summary")
    print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = f"{Fore.GREEN}PASS{Style.RESET_ALL}" if result else f"{Fore.YELLOW}FAIL{Style.RESET_ALL}"
        print(f"  {name:20s} {status}")

    print(f"\n  Total: {passed}/{total} tests passed\n")

    if passed == total:
        print(f"{Fore.GREEN}All tests passed! System is ready!{Style.RESET_ALL}")
    elif passed >= 4:
        print(f"{Fore.YELLOW}Core system working, some optional features unavailable{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}System not ready. Check errors above.{Style.RESET_ALL}")

    print(f"\n{Fore.CYAN}Next steps:{Style.RESET_ALL}")
    print(f"  1. Train models: python train_model.py --db mongodb")
    print(f"  2. Test queries:  python main.py --db mongodb --file examples/mongo_query_bad.json")
    print(f"  3. Run demo:      python demo.py\n")


if __name__ == '__main__':
    main()
