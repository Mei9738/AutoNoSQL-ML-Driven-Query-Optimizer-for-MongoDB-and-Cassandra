"""
Interactive script to add more training data
Use this to expand your dataset with custom queries
"""

import json
import os
from colorama import init, Fore, Style

init(autoreset=True)


def load_training_data(db_type):
    """Load existing training data"""
    file_path = f'training_data/{db_type}_queries.json'
    with open(file_path, 'r') as f:
        return json.load(f)


def save_training_data(db_type, data):
    """Save training data"""
    file_path = f'training_data/{db_type}_queries.json'
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"{Fore.GREEN}✓ Saved to {file_path}{Style.RESET_ALL}")


def add_mongodb_query():
    """Add a MongoDB query"""
    print(f"\n{Fore.CYAN}Add MongoDB Query{Style.RESET_ALL}")
    print("Enter query as JSON dict, e.g.: {\"collection\":\"users\",\"operation\":\"find\",\"filter\":{\"age\":30}}")

    query_str = input("Query: ").strip()

    try:
        query = json.loads(query_str)
    except:
        import ast
        try:
            query = ast.literal_eval(query_str)
        except:
            print(f"{Fore.RED}Invalid JSON format{Style.RESET_ALL}")
            return None

    print("\nIs this query GOOD (0) or BAD/needs optimization (1)?")
    label_str = input("Label (0 or 1): ").strip()

    try:
        label = int(label_str)
        if label not in [0, 1]:
            raise ValueError
    except:
        print(f"{Fore.RED}Label must be 0 or 1{Style.RESET_ALL}")
        return None

    description = input("Description (why is it good/bad?): ").strip()

    return {
        "query": query,
        "label": label,
        "description": description
    }


def add_cassandra_query():
    """Add a Cassandra query"""
    print(f"\n{Fore.CYAN}Add Cassandra Query{Style.RESET_ALL}")
    print("Enter CQL query, e.g.: SELECT * FROM users WHERE user_id = '123'")

    query = input("Query: ").strip()

    print("\nIs this query GOOD (0) or BAD/needs optimization (1)?")
    label_str = input("Label (0 or 1): ").strip()

    try:
        label = int(label_str)
        if label not in [0, 1]:
            raise ValueError
    except:
        print(f"{Fore.RED}Label must be 0 or 1{Style.RESET_ALL}")
        return None

    description = input("Description (why is it good/bad?): ").strip()

    return {
        "query": query,
        "label": label,
        "description": description
    }


def show_stats(data):
    """Show statistics"""
    queries = data['queries']
    good = sum(1 for q in queries if q['label'] == 0)
    bad = sum(1 for q in queries if q['label'] == 1)

    print(f"\n{Fore.CYAN}Current Dataset:{Style.RESET_ALL}")
    print(f"  Total queries: {len(queries)}")
    print(f"  Good queries:  {good} ({good/len(queries)*100:.1f}%)")
    print(f"  Bad queries:   {bad} ({bad/len(queries)*100:.1f}%)")


def main():
    print(f"\n{Fore.GREEN}{'='*60}")
    print(f"{Fore.GREEN}  AutoNoSQL - Add Training Data")
    print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}\n")

    # Choose database
    print("Which database?")
    print("  1. MongoDB")
    print("  2. Cassandra")
    choice = input("\nChoice (1 or 2): ").strip()

    if choice == '1':
        db_type = 'mongodb'
        add_func = add_mongodb_query
    elif choice == '2':
        db_type = 'cassandra'
        add_func = add_cassandra_query
    else:
        print(f"{Fore.RED}Invalid choice{Style.RESET_ALL}")
        return

    # Load existing data
    data = load_training_data(db_type)
    show_stats(data)

    # Add queries
    print(f"\n{Fore.YELLOW}Add new queries (press Ctrl+C to finish){Style.RESET_ALL}")

    try:
        while True:
            new_query = add_func()
            if new_query:
                data['queries'].append(new_query)
                print(f"{Fore.GREEN}✓ Added query!{Style.RESET_ALL}")
                show_stats(data)

            cont = input("\nAdd another? (y/n): ").strip().lower()
            if cont != 'y':
                break

    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Interrupted{Style.RESET_ALL}")

    # Save
    if input(f"\nSave changes? (y/n): ").strip().lower() == 'y':
        save_training_data(db_type, data)
        print(f"\n{Fore.GREEN}✓ Done! Remember to retrain the model:{Style.RESET_ALL}")
        print(f"  python train_model.py --db {db_type}")
    else:
        print("Changes discarded")


if __name__ == '__main__':
    main()
