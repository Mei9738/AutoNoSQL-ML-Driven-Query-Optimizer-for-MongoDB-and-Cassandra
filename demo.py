"""
Demo script to showcase AutoNoSQL ML + LLM workflow
Run this to see the difference between good and bad queries
"""

import subprocess
import sys
from colorama import init, Fore, Style

init(autoreset=True)


def run_command(cmd, description):
    """Run a command and print output"""
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{Fore.CYAN}{description}")
    print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0


def main():
    print(f"\n{Fore.GREEN}{'='*70}")
    print(f"{Fore.GREEN}  AutoNoSQL Demo - ML + LLM Query Optimization")
    print(f"{Fore.GREEN}{'='*70}{Style.RESET_ALL}\n")

    print("This demo shows how AutoNoSQL uses ML to gate LLM calls:\n")
    print("1. Good queries -> ML detects -> Skip expensive LLM call")
    print("2. Bad queries -> ML detects -> Call LLM for optimization advice\n")

    input(f"{Fore.YELLOW}Press Enter to start demo...{Style.RESET_ALL}")

    # Test 1: Good MongoDB query
    run_command(
        ["python", "main.py", "--db", "mongodb", "--file", "examples/mongo_query_good.json", "--no-llm"],
        "Test 1: MongoDB GOOD Query (should skip LLM)"
    )

    input(f"\n{Fore.YELLOW}Press Enter to continue...{Style.RESET_ALL}")

    # Test 2: Bad MongoDB query
    run_command(
        ["python", "main.py", "--db", "mongodb", "--file", "examples/mongo_query_bad.json", "--no-llm"],
        "Test 2: MongoDB BAD Query (ML detects issues)"
    )

    print(f"\n{Fore.GREEN}{'='*70}")
    print(f"{Fore.GREEN}  Demo Complete!")
    print(f"{Fore.GREEN}{'='*70}{Style.RESET_ALL}\n")

    print(f"{Fore.CYAN}Key Observations:{Style.RESET_ALL}")
    print(f"  • Good query: ML predicted 'GOOD QUERY' → {Fore.GREEN}Skipped LLM{Style.RESET_ALL}")
    print(f"  • Bad query:  ML predicted 'NEEDS OPTIMIZATION' → {Fore.YELLOW}Would call LLM{Style.RESET_ALL}")
    print(f"\n{Fore.CYAN}Efficiency Gain:{Style.RESET_ALL}")
    print(f"  • Without ML: Every query calls LLM (~3-10 seconds)")
    print(f"  • With ML:    Only bad queries call LLM (~50% reduction)")

    print(f"\n{Fore.YELLOW}To see LLM in action, remove --no-llm flag and ensure Ollama is running{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Example: python main.py --db mongodb --file examples/mongo_query_bad.json{Style.RESET_ALL}\n")


if __name__ == '__main__':
    main()
