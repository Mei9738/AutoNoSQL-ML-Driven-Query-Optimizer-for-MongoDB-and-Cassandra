"""
Train the ML query classifier model
"""

import json
import argparse
from autonosql.ml.query_classifier import QueryClassifier
from colorama import init, Fore, Style

init(autoreset=True)


def load_training_data(file_path: str):
    """Load training data from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)

    queries = [item['query'] for item in data['queries']]
    labels = [item['label'] for item in data['queries']]

    return queries, labels


def main():
    parser = argparse.ArgumentParser(description='Train ML query classifier')
    parser.add_argument('--db', required=True, choices=['mongodb', 'cassandra'],
                        help='Database type to train for')
    parser.add_argument('--data', help='Path to training data JSON file')
    parser.add_argument('--output', help='Path to save trained model')

    args = parser.parse_args()

    # Determine data file path
    if args.data:
        data_file = args.data
    else:
        data_file = f'training_data/{args.db}_queries.json'

    # Determine output path
    if args.output:
        model_path = args.output
    else:
        model_path = f'models/{args.db}_classifier.pkl'

    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{Fore.CYAN}Training ML Query Classifier - {args.db.upper()}")
    print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")

    # Load training data
    print(f"Loading training data from {data_file}...")
    queries, labels = load_training_data(data_file)

    print(f"Loaded {len(queries)} training samples")
    print(f"  - Good queries: {labels.count(0)}")
    print(f"  - Bad queries (need optimization): {labels.count(1)}\n")

    # Initialize classifier
    classifier = QueryClassifier(model_path=model_path)

    # Train the model
    print(f"{Fore.YELLOW}Training model...{Style.RESET_ALL}\n")
    classifier.train(queries, labels, database_type=args.db)

    # Evaluate on training data (for demonstration)
    print(f"\n{Fore.YELLOW}Evaluating on training data:{Style.RESET_ALL}")
    classifier.evaluate(queries, labels, database_type=args.db)

    # Save the model
    print(f"\n{Fore.YELLOW}Saving model...{Style.RESET_ALL}")
    classifier.save_model(model_path)

    print(f"\n{Fore.GREEN}Training complete!{Style.RESET_ALL}")
    print(f"Model saved to: {model_path}\n")

    # Test with example queries
    print(f"{Fore.CYAN}Testing with example queries:{Style.RESET_ALL}\n")

    if args.db == 'mongodb':
        test_queries = [
            {"collection": "users", "operation": "find", "filter": {"_id": "123"}},
            {"collection": "users", "operation": "find", "filter": {"name": {"$regex": "^test"}, "age": {"$ne": 30}}}
        ]
    else:
        test_queries = [
            "SELECT user_id, name FROM users WHERE user_id = '123'",
            "SELECT * FROM users WHERE age > 25 ALLOW FILTERING"
        ]

    for i, query in enumerate(test_queries, 1):
        prediction, confidence = classifier.predict(query, database_type=args.db)

        status = f"{Fore.GREEN}GOOD" if prediction == 0 else f"{Fore.RED}BAD (needs optimization)"
        print(f"Query {i}: {status}{Style.RESET_ALL}")
        print(f"  Confidence: {confidence:.2%}")
        if args.db == 'mongodb':
            print(f"  Query: {query}")
        else:
            print(f"  Query: {query[:60]}...")
        print()


if __name__ == '__main__':
    main()
