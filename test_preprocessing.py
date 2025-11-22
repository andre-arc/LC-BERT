"""
Test script for different preprocessing methods.

Usage:
    python test_preprocessing.py
"""

import utils.preprocessing as preprocessing

# Sample texts to test
test_texts = [
    "The companies are running multiple studies on the effects of global warming.",
    "Children were playing happily in the park while their parents watched them.",
    "He's been working on developing new technologies for sustainable energy.",
]

print("="*80)
print("PREPROCESSING METHODS COMPARISON")
print("="*80)
print()

methods = ['gensim', 'nltk', 'stanza']

for i, text in enumerate(test_texts, 1):
    print(f"Original Text {i}:")
    print(f"  {text}")
    print()

    for method in methods:
        try:
            # Set preprocessing method
            preprocessing.set_preprocessing_method(method)

            # Clean the text
            cleaned = preprocessing.clean([text])

            print(f"  {method.upper():8} → {cleaned[0]}")
        except ImportError as e:
            print(f"  {method.upper():8} → [NOT INSTALLED: {e}]")
        except Exception as e:
            print(f"  {method.upper():8} → [ERROR: {e}]")

    print()
    print("-"*80)
    print()

print("Testing complete!")
print()
print("Current method:", preprocessing.get_preprocessing_method())
