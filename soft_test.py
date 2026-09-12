import sys
import os
import benchmark_massive

# Override constants for a quick soft test
benchmark_massive.PROMPT_COUNT = 1
benchmark_massive.RESULTS_FILE = "soft_test_results.json"

# Test on all available models with 1 prompt
# (We retain benchmark_massive.MODELS as is, meaning all 4 models will run)

def mock_load_prompts():
    return ["Explain why hibernating bears don't need to eat, drink, or urinate for months. Keep the answer to 4-5 sentences, final answer only."]

if __name__ == "__main__":
    benchmark_massive.load_prompts = mock_load_prompts
    print("Running soft test for ALL models on 1 prompt...")
    benchmark_massive.main()
