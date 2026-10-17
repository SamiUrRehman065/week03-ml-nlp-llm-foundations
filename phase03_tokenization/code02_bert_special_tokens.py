from transformers import AutoTokenizer


def run():
    print("=" * 60)
    print("PHASE 03 - CODE 02: BERT ENCODING, SPECIAL TOKENS & DECODING")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    text = "Large language models are powerful."

    print("Original text:")
    print(text)

    tokens = tokenizer.tokenize(text)
    print("\nTokens:")
    print(tokens)

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=True
    )

    print("\nToken IDs:")
    print(token_ids)

    print("\nNumber of tokens:")
    print(len(token_ids))

    decoded = tokenizer.decode(token_ids)
    print("\nDecoded text:")
    print(decoded)


if __name__ == "__main__":
    run()
