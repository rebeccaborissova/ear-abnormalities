# Runs all test scripts in sequence and print a final summary.

import subprocess
import sys
import os
 
TESTS = [
    ("Preprocessing", "test_preprocessing.py"),
    ("Model Output", "test_model_output.py"),
    ("Accuracy Calculation", "test_accuracy.py"),
    ("Robustness", "test_robustness.py"),
    ("23-Point Model on Adult Dataset", "test_adult_dataset.py"),
]

os.chdir(os.path.dirname(os.path.abspath(__file__)))

results = []
for label, script in TESTS:
    print(f"\n{'='*60}")
    print(f"  Running: {label} ({script})")
    print(f"{'='*60}")
    ret = subprocess.run([sys.executable, script])
    results.append((label, script, ret.returncode))
    
print(f"\n{'='*60}")
print("  FINAL RESULTS")
print(f"{'='*60}")
for label, script, code in results:
    status = "OK" if code == 0 else "ERRORS"
    print(f"  [{status}]  {label}")