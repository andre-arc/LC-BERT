from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Add CLS token
subwords = [tokenizer.cls_token_id]
subword_to_word_indices = [-1] # For CLS

print(subwords)
print(subword_to_word_indices)