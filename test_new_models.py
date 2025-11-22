"""
Simple test script to verify DistilBERT and ALBERT implementations
"""
import torch
from modules.word_classification import DistilBertForWordClassification, AlbertForWordClassification
from transformers import DistilBertConfig, AlbertConfig

def test_distilbert():
    print("Testing DistilBertForWordClassification...")
    try:
        # Create a config
        config = DistilBertConfig.from_pretrained('distilbert-base-uncased')
        config.num_labels = 4  # AG News has 4 classes

        # Initialize model
        model = DistilBertForWordClassification(config)

        # Create dummy input
        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        labels = torch.randint(0, 4, (batch_size,))

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs[0]
        logits = outputs[1]

        print(f"  ✓ Model initialized successfully")
        print(f"  ✓ Forward pass completed")
        print(f"  ✓ Loss shape: {loss.shape}")
        print(f"  ✓ Logits shape: {logits.shape} (expected: [{batch_size}, {config.num_labels}])")

        assert logits.shape == (batch_size, config.num_labels), "Logits shape mismatch!"
        print("  ✓ DistilBERT test PASSED!\n")
        return True

    except Exception as e:
        print(f"  ✗ DistilBERT test FAILED: {e}\n")
        return False

def test_albert():
    print("Testing AlbertForWordClassification...")
    try:
        # Create a config
        config = AlbertConfig.from_pretrained('albert-base-v2')
        config.num_labels = 4  # AG News has 4 classes

        # Initialize model
        model = AlbertForWordClassification(config)

        # Create dummy input
        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        token_type_ids = torch.zeros(batch_size, seq_length, dtype=torch.long)
        labels = torch.randint(0, 4, (batch_size,))

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )

        loss = outputs[0]
        logits = outputs[1]

        print(f"  ✓ Model initialized successfully")
        print(f"  ✓ Forward pass completed")
        print(f"  ✓ Loss shape: {loss.shape}")
        print(f"  ✓ Logits shape: {logits.shape} (expected: [{batch_size}, {config.num_labels}])")

        assert logits.shape == (batch_size, config.num_labels), "Logits shape mismatch!"
        print("  ✓ ALBERT test PASSED!\n")
        return True

    except Exception as e:
        print(f"  ✗ ALBERT test FAILED: {e}\n")
        return False

def test_model_loading():
    print("Testing model loading functions...")
    try:
        from utils.functions import load_model, load_tokenizer, load_extraction_model

        # Test DistilBERT loading
        print("  Testing DistilBERT loading...")
        args_distilbert = {
            'model_name': 'distilbert-base-uncased',
            'num_labels': 4
        }
        model, tokenizer, _, _ = load_model(args_distilbert)
        print(f"    ✓ DistilBERT model loaded: {type(model).__name__}")
        print(f"    ✓ DistilBERT tokenizer loaded: {type(tokenizer).__name__}")

        # Test ALBERT loading
        print("  Testing ALBERT loading...")
        args_albert = {
            'model_name': 'albert-base-v2',
            'num_labels': 4
        }
        model, tokenizer, _, _ = load_model(args_albert)
        print(f"    ✓ ALBERT model loaded: {type(model).__name__}")
        print(f"    ✓ ALBERT tokenizer loaded: {type(tokenizer).__name__}")

        # Test extraction model loading
        print("  Testing extraction model loading...")
        args_extract_distilbert = {
            'extract_model': 'distilbert-base-uncased'
        }
        model, tokenizer, _, _ = load_extraction_model(args_extract_distilbert)
        print(f"    ✓ DistilBERT extraction model loaded: {type(model).__name__}")

        args_extract_albert = {
            'extract_model': 'albert-base-v2'
        }
        model, tokenizer, _, _ = load_extraction_model(args_extract_albert)
        print(f"    ✓ ALBERT extraction model loaded: {type(model).__name__}")

        print("  ✓ Model loading test PASSED!\n")
        return True

    except Exception as e:
        print(f"  ✗ Model loading test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing DistilBERT and ALBERT implementations")
    print("=" * 60 + "\n")

    results = []

    # Run tests
    results.append(("DistilBERT", test_distilbert()))
    results.append(("ALBERT", test_albert()))
    results.append(("Model Loading", test_model_loading()))

    # Summary
    print("=" * 60)
    print("Test Summary:")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:20s}: {status}")

    all_passed = all(result[1] for result in results)
    print("=" * 60)
    if all_passed:
        print("All tests PASSED! ✓")
    else:
        print("Some tests FAILED! ✗")
    print("=" * 60)
