from transformers import AutoTokenizer


def run():
    print("=" * 60)
    print("PHASE 03 - CODE 01: BERT TOKENIZATION & ENCODING")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    text = "Large language models are powerful."

    tokens = tokenizer.tokenize(text)
    print("Original Text:", text)
    print("\nTokens:")
    print(tokens)

    token_ids = tokenizer.encode(text)
    print("\nToken IDs:")
    print(token_ids)


if __name__ == "__main__":
    run()
