"""
run_tests.py
------------
Script to run all EARLY agent tests via pytest.
Automatically loads environment variables from .env to ensure
API keys (like GROQ_API_KEY) are available for DeepEval and LangChain.
"""

import os
import sys
import pytest
from dotenv import load_dotenv

def main():
    # 1. Dynamically resolve paths based on the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Add project root to sys.path so 'from agents...' imports work correctly
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # 2. Load environment variables from the root .env file
    load_dotenv(os.path.join(project_root, ".env"))
    
    if not os.getenv("GROQ_API_KEY"):
        print("⚠️ WARNING: GROQ_API_KEY is not set. Live LLM tests will fail.", file=sys.stderr)
        
    print("🚀 Running EARLY agent tests...\n")
    
    # 3. Run pytest on the agents test directory using the absolute path
    agents_test_dir = os.path.join(script_dir, "agents")
    
    # Forward any additional command-line arguments to pytest
    pytest_args = [agents_test_dir, "-v"] + sys.argv[1:]
    exit_code = pytest.main(pytest_args)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()